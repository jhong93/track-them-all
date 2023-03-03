#!/usr/bin/env python3

import os
import argparse
from collections import defaultdict
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

import vipe
import contact
import vitpose
import eva
import yolo

from util.video import get_metadata
from util.box import Box, PersonMeta
from util.body import Pose
from util.sort import Sort
from util.vis import draw
from util.io import store_gz_json, store_json

cv2.setNumThreads(4)


PERSON_CLASS_ID = 0

MIN_CONF = 0.25
MIN_HEIGHT = 0.1
MAX_HEIGHT = 0.8

MIN_TRACK_LEN = 60


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file')
    parser.add_argument('-y', '--use_yolo', action='store_true')
    parser.add_argument('-v', '--visualize_detection', action='store_true')
    parser.add_argument('-vt', '--visualize_tracking', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--out_dir')
    return parser.parse_args()


class VideoAsFrames(Dataset):

    def __init__(self, video_file):
        self.meta = get_metadata(video_file)

        self.vc = None
        assert os.path.exists(video_file)
        self.video_file = video_file

    def __getitem__(self, i):
        if self.vc is None:
            self.vc = cv2.VideoCapture(self.video_file)
            self.frame_num = 0

        ret, frame = self.vc.read()

        frame_num = self.frame_num
        self.frame_num += 1

        if not ret:
            self.vc.release()
            print(self.frame_num, self.video_file)
            frame = np.zeros((2, 2, 3), np.uint8)
        return frame_num, ret, frame

    def __len__(self):
        return self.meta.num_frames


def is_cropped_pose(keyp, width, height):
    low_hip_threshold = height * 0.9
    if min(keyp[[Pose.RHip, Pose.LHip], 1]) > low_hip_threshold:
        return True

    low_shoulder_threshold = height * 0.8
    if min(keyp[[Pose.RShoulder, Pose.LShoulder], 1]) > low_shoulder_threshold:
        return True

    vedge_threshold = width * 0.1
    if (max(keyp[:, 0]) < vedge_threshold
        or min(keyp[:, 0]) > width - vedge_threshold
    ):
        return True
    return False


def detect_people(video_file, use_yolo, limit, visualize=False, heatmap=False):
    infer = yolo.infer if use_yolo else eva.infer

    dataset = VideoAsFrames(video_file)
    loader = DataLoader(dataset)
    height = dataset.meta.height

    all_dets = []
    for i, ret, frame in tqdm(loader, desc='Detect'):
        i = i[0]
        ret = ret[0]
        frame = frame[0].numpy()
        if not ret:
            break
        if limit and i > limit:
            break

        frame_dets = []
        result = infer(frame)

        bboxes_for_pose = []
        for det in result:
            if int(det['class']) != PERSON_CLASS_ID:
                continue
            if det['score'] < MIN_CONF:
                continue

            x, y, w, h = det['xywh']
            h_frac = h / height
            if h_frac < MIN_HEIGHT or h_frac > MAX_HEIGHT:
                continue

            bboxes_for_pose.append([x, y, w, h])
            frame_dets.append(Box(
                x, y, w, h, score=det['score'], payload=det['mask']))

        # Batch the pose computation
        pose, extra_outputs = vitpose.infer_pose(
            frame, bboxes_for_pose, heatmap=heatmap)
        frame_dets_cpy = []
        for i, det in enumerate(frame_dets):
            keyp = pose[i]['keypoints']

            if is_cropped_pose(keyp, dataset.meta.width, dataset.meta.height):
                continue

            heatmap = extra_outputs[0]['heatmap'][i] if heatmap else None
            keyp_vipe = vipe.preprocess_2d_keyp(keyp, flip=False)
            vipe_emb = vipe_model.embed(keyp_vipe)[0]

            # TODO: masks and heatmaps take too much space
            det._payload = PersonMeta(
                pose=keyp,
                pose_heatmap=heatmap,
                mask=det.payload,
                vipe=vipe_emb)
            frame_dets_cpy.append(det)
        frame_dets = frame_dets_cpy
        del frame_dets_cpy

        if visualize:
            frame = draw(frame, frame_dets)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

        all_dets.append(frame_dets)
    if visualize:
        cv2.destroyAllWindows()
    return all_dets


def compute_sort_tracks(all_dets, max_age=1):
    tracker = Sort(max_age=max_age)
    tracks = []
    for frame, dets in enumerate(tqdm(all_dets, 'Track')):
        # Advance state
        if len(dets) == 0:
            tracker.update(np.empty((0, 5)))
        else:
            boxes = []
            for b in dets:
                boxes.append([b.x, b.y, b.x2, b.y2, b.score])

            frame_tracks = tracker.update(np.array(boxes))
            if frame_tracks.shape[0] > 0:
                tmp = []
                for i in range(frame_tracks.shape[0]):
                    x1, y1, x2, y2, obj_id = frame_tracks[i, :].tolist()
                    b = Box(x1, y1, x2 - x1, y2 - y1)

                    # Match with detections
                    best = None
                    best_iou = 0.5
                    for d in dets:
                        iou = d.iou(b)
                        if iou > best_iou:
                            best = d
                            best_iou = iou

                    if best is not None:
                        tmp.append((int(obj_id), best))
                tracks.append((frame, tmp))
    return tracks


def track_people(dets):
    sort_tracks = compute_sort_tracks(dets)
    for frame, track_and_boxes in sort_tracks:
        if track_and_boxes is not None and len(track_and_boxes) > 0:
            for d in dets[frame]:
                best_iou = 0.8
                best_track = None
                for track_id, b in track_and_boxes:
                    if b.iou(d) > best_iou:
                        best_track = track_id

                if best_track is not None:
                    d._track = best_track
    return dets


def group_by_track_and_infer_contact(dets):
    det_by_track = defaultdict(list)
    for t, frame_dets in enumerate(dets):
        for d in frame_dets:
            if d.track is not None:
                det_by_track[d.track].append((t, d))
    print('# tracks:', max(det_by_track.keys()))

    for track in det_by_track:
        dets = det_by_track[track]
        if len(dets) < MIN_TRACK_LEN:
            continue

        meta = []
        pose = []
        heatmap = []
        mask = {}
        vipe = []
        for t, d in dets:
            meta.append({
                't': t,
                'xywh': [d.x, d.y, d.w, d.h],
                'score': d.score
            })
            heatmap.append(d.payload.pose_heatmap)
            mask[str(t)] = d.payload.mask
            vipe.append(d.payload.vipe)
            pose.append(d.payload.pose)

        # Use iterator to save memory
        pose = np.stack(pose)
        if heatmap[0] is not None:
            heatmap = np.stack(heatmap)
        vipe = np.stack(vipe)
        contacts = contact.infer_contact(contact_model, pose)
        yield (track, meta, pose, heatmap, mask, vipe, contacts)


def visualize_people(video_file, dets):
    vc = cv2.VideoCapture(video_file)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in trange(num_frames, desc='Visualize'):
        ret, frame = vc.read()
        if not ret:
            break

        draw(frame, dets[i])
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    vc.release()
    cv2.destroyAllWindows()


def save_results(video_file, out_dir, track_iterator):
    print('Saving results:')
    for track, meta, pose, heatmap, mask, vipe, contacts in track_iterator:
        track_dir = os.path.join(out_dir, '{:06d}'.format(track))

        os.makedirs(track_dir)
        store_gz_json(os.path.join(track_dir, 'meta.json.gz'), meta)
        np.save(os.path.join(track_dir, 'vipe.npy'), vipe)
        np.save(os.path.join(track_dir, 'pose.npy'), pose)
        np.save(os.path.join(track_dir, 'contact.npy'), contacts)

        # NOTE: dont bother with heatmaps
        # np.save(os.path.join(track_dir, 'pose_heatmap.npy'), heatmap)

        # NOTE: find better way to store masks
        np.savez_compressed(os.path.join(track_dir, 'mask.npz'), **mask)

    meta = get_metadata(video_file)
    store_json(os.path.join(out_dir, 'meta.json'), {
        'video_file': os.path.basename(video_file),
        'fps': meta.fps,
        'num_frames': meta.num_frames,
        'width': meta.width,
        'height': meta.height
    })


def main(args):
    if args.use_yolo:
        yolo.init_model()
    else:
        eva.init_model()
    vitpose.init_model()

    global vipe_model
    vipe_model = vipe.load_embedding_model()

    global contact_model
    contact_model = contact.load_contact_model()

    def process(video_file, out_dir):
        detections = detect_people(
            video_file, args.use_yolo, args.limit,
            visualize=args.visualize_detection)
        detections = track_people(detections)
        if args.visualize_tracking:
            visualize_people(video_file, detections)

        track_detections = group_by_track_and_infer_contact(detections)
        if out_dir is not None:
            save_results(video_file, out_dir, track_detections)

    if os.path.isdir(args.video_file):
        video_dir = args.video_file
        for video_file in tqdm(os.listdir(video_dir)):
            if video_file.endswith('.mp4'):
                video_name = os.path.splitext(video_file)[0]
                print('Processing:', video_name)
                process(os.path.join(video_dir, video_file),
                        os.path.join(args.out_dir, video_name))
    else:
        process(args.video_file, args.out_dir)
    print('Done!')


if __name__ == '__main__':
    main(get_args())