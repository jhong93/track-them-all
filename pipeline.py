#!/usr/bin/env python3

import os
import argparse
import random
from collections import defaultdict
from typing import NamedTuple
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vipe import load_embedding_model, preprocess_2d_keyp
import vitpose
import eva
import yolo

from util.video import get_metadata
from util.box import Box
from util.sort import Sort
from util.io import store_gz_json, store_json

cv2.setNumThreads(4)


PERSON_CLASS_ID = 0

MIN_CONF = 0.25
MIN_HEIGHT = 0.1
MAX_HEIGHT = 0.8

MIN_TRACK_LEN = 60


BOX_COLOR = (0, 0, 255)

LEFT_COLOR = (120, 200, 255)
RIGHT_COLOR = (120, 255, 200)

LEFT_BONES = [(5, 0), (7, 5), (9, 7), (11, 5), (13, 11), (15, 13)]
RIGHT_BONES = [(6, 0), (8, 6), (10, 8), (12, 6), (14, 12), (16, 14)]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file')
    parser.add_argument('-y', '--use_yolo', action='store_true')
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--out_dir')
    return parser.parse_args()


class PersonMeta(NamedTuple):
    pose: 'np.ndarray'
    pose_heatmap: 'np.ndarray'
    mask: 'np.ndarray'
    vipe: 'np.ndarray'


def draw(frame, boxes, alpha=0.33, track_colors={}):
    has_mask = False

    for box in boxes:
        if box.track is None:
            box_color = BOX_COLOR
        else:
            if box.track not in track_colors:
                track_colors[box.track] = tuple([
                    random.randint(0, 255) for _ in range(3)])
            box_color = track_colors[box.track]

        cv2.rectangle(
            frame, (int(box.x), int(box.y)),
            (int(box.x + box.w), int(box.y + box.h)),
            box_color, 2)
        cv2.putText(frame, '{:0.3f}'.format(box.score),
                    (int(box.x), int(box.y) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, box_color, 1)

        for bones, color in (
            (LEFT_BONES, LEFT_COLOR),
            (RIGHT_BONES, RIGHT_COLOR),
        ):
            for b1, b2 in bones:
                b1_int = (int(box.payload.pose[b1, 0]),
                          int(box.payload.pose[b1, 1]))
                b2_int = (int(box.payload.pose[b2, 0]),
                          int(box.payload.pose[b2, 1]))
                cv2.line(frame, b1_int, b2_int, color, 2)

        if box.payload.mask is not None:
            if not has_mask:
                mask_frame = np.zeros_like(frame)
            has_mask = True

            x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)
            mask_frame[y:y + h, x:x + w, 2] = box.payload.mask * 255

    if has_mask:
        frame = cv2.addWeighted(frame, 1. - alpha, mask_frame, alpha, 0.)
    return frame


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


def detect_people(video_file, use_yolo, limit, visualize=False):
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
            frame, bboxes_for_pose)
        for i, det in enumerate(frame_dets):
            keyp = pose[i]['keypoints']
            heatmap = extra_outputs[0]['heatmap'][i]
            keyp_vipe = preprocess_2d_keyp(keyp, flip=False)
            vipe = vipe_model.embed(keyp_vipe)[0]

            det._payload = PersonMeta(
                pose=keyp,
                pose_heatmap=heatmap,
                mask=det.payload,
                vipe=vipe)

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

        prev_frame = frame
    return tracks


def track_people(dets):
    sort_tracks = compute_sort_tracks(dets)

    for frame, track_and_boxes in sort_tracks:
        if track_and_boxes is not None and len(track_and_boxes) > 0:
            for d in dets[frame]:
                best_iou = 0.8
                best_track = None
                for track_id, b in track_and_boxes:
                    if b.iou(d) > 0.8:
                        best_track = track_id

                if best_track is not None:
                    d._track = best_track
    return dets


def visualize_people(video_file, dets):
    vc = cv2.VideoCapture(video_file)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in trange(num_frames, desc='Detect'):
        ret, frame = vc.read()
        if not ret:
            break

        draw(frame, dets[i])
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    vc.release()
    if visualize:
        cv2.destroyAllWindows()


def save_results(out_dir, all_dets):
    det_by_track = defaultdict(list)
    for t, frame_dets in enumerate(all_dets):
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

        pose = np.stack(pose)
        heatmap = np.stack(heatmap)
        vipe = np.stack(vipe)

        if out_dir is not None:
            track_dir = os.path.join(out_dir, '{:06d}'.format(track))
            os.makedirs(track_dir)
            store_gz_json(os.path.join(track_dir, 'meta.json.gz'), meta)
            np.save(os.path.join(track_dir, 'vipe.npy'), vipe)
            np.save(os.path.join(track_dir, 'pose.npy'), pose)
            np.save(os.path.join(track_dir, 'pose_heatmap.npy'), heatmap)
            np.savez_compressed(os.path.join(track_dir, 'mask.npz'), **mask)


def main(video_file, use_yolo, visualize, out_dir, limit):
    if use_yolo:
        yolo.init_model()
    else:
        eva.init_model()
    vitpose.init_model()

    global vipe_model
    vipe_model = load_embedding_model('pretrained_vipe')

    def process(video_file, out_dir):
        detections = detect_people(video_file, use_yolo, limit)
        detections = track_people(detections)
        if visualize:
            visualize_people(video_file, detections)
        save_results(out_dir, detections)

    if os.path.isdir(video_file):
        video_dir = video_file
        for video_file in os.listdir(video_dir):
            video_name = os.path.splitext(video_file)[0]
            process(os.path.join(video_file, video_file),
                    os.path.join(video_dir, video_name))
    else:
        process(video_file, out_dir)
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))