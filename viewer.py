#!/usr/bin/env python3

import os
import argparse
import datetime
from io import BytesIO
from typing import NamedTuple
import cv2
import smplx
import torch
from tqdm import tqdm, trange
from PIL import Image
import numpy as np

from flask import Flask, send_file, render_template, request, jsonify

from util.video import get_metadata
from util.box import Box, PersonMeta
from util.io import (load_gz_json, load_json, store_json, load_pickle,
                     decode_png)
from util.vis import draw
from util.renderer_pyrd import Renderer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('track_dir')
    parser.add_argument('video_dir')
    parser.add_argument('-p', '--port', type=int, default=9001)
    return parser.parse_args()


class Track(NamedTuple):
    id: int
    video: str
    track: int
    path: str
    length: int
    min_frame: int
    fps: float
    detections: object
    has_3d: bool


def list_tracks(track_dir):
    tracks = []
    count = 0
    for sub_dir in sorted(os.listdir(track_dir)):
        sub_dir_path = os.path.join(track_dir, sub_dir)
        meta = load_json(os.path.join(sub_dir_path, 'meta.json'))
        for track in sorted(os.listdir(sub_dir_path)):
            track_path = os.path.join(sub_dir_path, track)
            if not os.path.isdir(track_path):
                continue

            det_data = load_gz_json(os.path.join(track_path, 'meta.json.gz'))
            max_t = det_data[-1]['t']
            min_t = det_data[0]['t']

            has_3d = os.path.exists(os.path.join(track_path, 'smpl.npz'))
            tracks.append(Track(
                count,
                meta['video_file'],
                int(track),
                track_path,
                max_t - min_t + 1,
                min_t,
                meta['fps'],
                det_data,
                has_3d))
            count += 1
    return tracks


class TrackInfo(NamedTuple):
    id: int
    video: str
    track: int
    length: int
    label: str
    has_3d: bool


def get_label(track):
    label_path = os.path.join(track.path, 'label.json')
    if os.path.exists(label_path):
        return load_json(label_path)['value']
    else:
        return None


def set_label(track, value):
    label_path = os.path.join(track.path, 'label.json')
    if value is None:
        print('Clear:', label_path)
        if os.path.exists(label_path):
            os.remove(label_path)
    else:
        print('Set:', label_path)
        store_json(label_path, {
            'value': value, 'time': datetime.datetime.now().isoformat()})


def track_iou(t1, t2):
    overlap = min(t1.min_frame + t1.length, t2.min_frame + t2.length) - max(
        t1.min_frame, t2.min_frame)
    if overlap < 0:
        return 0.
    return overlap / (t1.length + t2.length - overlap)


def load_frames_w_track(video_dir, track, out_height=480):
    pose = np.load(os.path.join(track.path, 'pose.npy'), mmap_mode='r')
    mask = load_pickle(os.path.join(track.path, 'mask.pkl'))
    box_dict = {det['t']: [Box(
        *det['xywh'], det['score'], payload=PersonMeta(
            pose=pose[i], mask=decode_png(mask[i]))
    )] for i, det in enumerate(track.detections)}

    frames = []
    vc = cv2.VideoCapture(os.path.join(video_dir, track.video))
    fps = vc.get(cv2.CAP_PROP_FPS)
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_width = int(out_height / height * width)

    vc.set(cv2.CAP_PROP_POS_FRAMES, track.min_frame)
    for i in range(track.min_frame, track.min_frame + track.length + 1):
        ret, frame = vc.read()
        if not ret:
            break

        frame = draw(frame, box_dict.get(i, []))
        frame = cv2.resize(frame, (out_width, out_height))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    vc.release()
    return frames, fps


def load_frames_w_smpl(video_dir, track, out_height=480):
    smpl_model = smplx.create('data', 'smpl')

    smpl_data = np.load(os.path.join(track.path, 'smpl.npz'))
    pose = smpl_data['pose']
    betas = smpl_data['shape']
    trans = smpl_data['global_t']
    focal_l = smpl_data['focal_l']

    pred_vert_arr = []
    for i in trange(len(track.detections), desc='Running SMPL'):
        betas_i = torch.from_numpy(betas[i]).unsqueeze(0)
        pose_i = torch.from_numpy(pose[i][1:]).unsqueeze(0)
        orient_i = torch.from_numpy(pose[i][[0]]).unsqueeze(0)
        trans_i = torch.from_numpy(trans[i]).unsqueeze(0)

        pred_output = smpl_model(
            betas=betas_i,
            body_pose=pose_i,
            global_orient=orient_i,
            pose2rot=False,
            transl=trans_i)
        pred_vertices = pred_output.vertices
        pred_vert_arr.extend(pred_vertices.cpu().numpy())

    vc = cv2.VideoCapture(os.path.join(video_dir, track.video))
    fps = vc.get(cv2.CAP_PROP_FPS)
    gap = int(1000 / fps)
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_width = int(out_height / height * width)

    frames = []
    for i, det in enumerate(tqdm(track.detections, desc='Rendering')):
        t = det['t']
        if t != int(vc.get(cv2.CAP_PROP_POS_FRAMES)):
            vc.set(cv2.CAP_PROP_POS_FRAMES, t)

        _, img_bgr = vc.read()
        img_h, img_w, _ = img_bgr.shape
        renderer = Renderer(
            focal_length=focal_l[i], img_w=img_w, img_h=img_h,
            faces=smpl_model.faces,
            same_mesh_color=True)
        front_view = renderer.render_front_view(
            pred_vert_arr[i:i + 1],
            bg_img_rgb=img_bgr[:, :, ::-1].copy())
        renderer.delete()

        frames.append(cv2.resize(front_view, (out_width, out_height)))
    vc.release()
    return frames, fps


def build_app(args):
    tracks = list_tracks(args.track_dir)

    def has_label(track_dir):
        return os.path.exists(os.path.join(track_dir, 'label.json'))

    def video_path(video_file):
        return os.path.join(args.video_dir, video_file)

    app = Flask(__name__, template_folder='web/templates',
                static_folder='web/static')

    @app.route('/')
    def root():
        limit = request.args.get('limit', type=int, default=500)
        select = request.args.get('select')
        unlabeled_only = select == 'nolabel'

        num_accept = 0
        num_reject = 0
        num_unlabeled = 0
        num_seconds = 0
        num_seconds_accepted = 0

        filtered_tracks = []
        for t in tracks:
            label = get_label(t)
            if label is None:
                num_unlabeled += 1
            elif label == 'accept':
                num_accept += 1
                num_seconds_accepted += t.length / t.fps
            elif label == 'reject':
                num_reject += 1
            num_seconds += t.length / t.fps

            if select is not None:
                if unlabeled_only:
                    if label is not None:
                        continue
                else:
                    if label != select:
                        continue

            filtered_tracks.append(TrackInfo(
                id=t.id, video=t.video, track=t.track, length=t.length,
                label=label, has_3d=t.has_3d))

        filtered_tracks.sort(key=lambda x: -x.length)
        filtered_tracks = filtered_tracks[:limit]
        return render_template(
            'root.html',
            num_tracks=len(tracks),
            num_frames=sum(t.length + 1 for t in tracks),
            num_accepted=num_accept,
            num_rejected=num_reject,
            num_unlabeled=num_unlabeled,
            num_seconds=round(num_seconds),
            num_seconds_accepted=round(num_seconds_accepted),
            tracks=filtered_tracks)

    @app.route('/label/<int:id>', methods=['POST'])
    def _set_label(id):
        value = request.args.get('value')

        track = tracks[id]
        if value not in ['', 'accept', 'reject', 'rejectmany', 'flag']:
            return 'Error', 400
        elif value == 'rejectmany':
            to_set = [track]
            for t in tracks:
                if t.id != id and t.video == track.video:
                    if track_iou(t, track) > 0.25:
                        # Also reject t
                        to_set.append(t)
            for t in to_set:
                set_label(t, 'reject')
            return 'Success: {}'.format(value), 200
        else:
            if value == '':
                value = None
            set_label(track, value)
            return 'Success: {}'.format(value), 200

    @app.route('/label/<int:id>', methods=['GET'])
    def _get_label(id):
        return str(get_label(tracks[id])), 200

    @app.route('/track-preview/<int:id>')
    def get_track_preview(id, n=5):
        out_height = request.args.get('height', type=int, default=240)

        track = tracks[id]
        pose = np.load(os.path.join(track.path, 'pose.npy'), mmap_mode='r')
        mask = load_pickle(os.path.join(track.path, 'mask.pkl'))
        stride = int(len(track.detections) / n)
        dets = track.detections[::stride][:n]

        vc = cv2.VideoCapture(os.path.join(args.video_dir, track.video))
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_width = int(out_height / height * width)

        frames = []
        for i, det in enumerate(dets):
            box = Box(*det['xywh'], det['score'], payload=PersonMeta(
                pose=pose[i * stride],
                mask=decode_png(mask[i * stride])))

            vc.set(cv2.CAP_PROP_POS_FRAMES, det['t'])
            _, frame = vc.read()
            frame = draw(frame, [box], thickness=5)
            frame = cv2.resize(frame, (out_width, out_height))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        img = Image.fromarray(np.hstack(frames))
        f = BytesIO()
        img.save(fp=f, format='JPEG')
        f.seek(0)
        return send_file(f, mimetype='image/jpeg')

    def send_gif(frames, fps):
        imgs = [Image.fromarray(x) for x in frames]

        gif_bytes = BytesIO()
        img0 = imgs[0]
        img0.save(fp=gif_bytes, format='GIF',
                  append_images=imgs[1:], save_all=True,
                  duration=1 / fps, loop=0)
        gif_bytes.seek(0)
        return send_file(gif_bytes, mimetype='image/gif')

    @app.route('/track-gif/<int:id>')
    def get_track_gif(id):
        frames, fps = load_frames_w_track(args.video_dir, tracks[id])
        return send_gif(frames, fps)

    @app.route('/smpl-gif/<int:id>')
    def get_smpl_gif(id):
        frames, fps = load_frames_w_smpl(args.video_dir, tracks[id])
        return send_gif(frames, fps)

    @app.route('/labels')
    @app.route('/labels.json')
    def get_labels():
        result = []
        for t in tracks:
            label = get_label(t)
            if label == 'accept':
                result.append({
                    'video': t.video,
                    'track': t.track,
                    'length': t.length
                })
        return jsonify(result)

    return app


def main(args):
    app = build_app(args)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, port=args.port, host='0.0.0.0',
            threaded=False, processes=os.cpu_count())


if __name__ == '__main__':
    main(get_args())
