#!/usr/bin/env python3

import os
import argparse
import datetime
from io import BytesIO
from typing import NamedTuple
import cv2
from PIL import Image
import numpy as np

from flask import Flask, send_file, render_template, request

from util.video import get_metadata
from util.box import Box, PersonMeta
from util.io import load_gz_json, load_json, store_json
from util.vis import draw


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
    detections: object


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
            tracks.append(Track(
                count,
                meta['video_file'],
                int(track),
                track_path,
                max_t - min_t + 1,
                min_t,
                det_data))
            count += 1

    tracks.sort(key=lambda x: (-x.length, x.video, x.track))
    return tracks


class TrackInfo(NamedTuple):
    id: int
    video: str
    track: int
    length: int
    label: str


def get_label(track):
    label_path = os.path.join(track.path, 'label.json')
    if os.path.exists(label_path):
        return load_json(label_path)['value']
    else:
        return None


def set_label(track, value):
    label_path = os.path.join(track.path, 'label.json')
    if value is not None:
        if os.path.exists(label_path):
            os.remove(label_path)
    else:
        store_json(label_path, {
            'value': value, 'time': datetime.now().isoformat()})


def track_temporal_sim(t1, t2):
    overlap = min(t1.min_frame + t1.length, t2.min_frame + t2.length) - max(
        t1.min_frame, t2.min_frame)
    return max(0, overlap) / t1.length


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
        select = request.args.get('select')
        unlabeled_only = select == 'nolabel'

        filtered_tracks = []
        for t in tracks:
            label = get_label(t)
            if select is not None:
                if unlabeled_only:
                    if label is not None:
                        continue
                else:
                    if label != select:
                        continue
            filtered_tracks.append(TrackInfo(
                id=t.id, video=t.video, track=t.track, length=t.length,
                label=label))

        return render_template(
            'root.html',
            num_tracks=len(tracks),
            num_frames=sum(t.length + 1 for t in tracks),
            tracks=filtered_tracks)

    @app.route('/label/<int:id>', methods=['POST'])
    def label(id):
        value = request.args.get('value')
        track = tracks[id]
        if value not in ['', 'accept', 'reject', 'rejectmany', 'flag']:
            return 'Error', 400
        elif value == 'rejectmany':
            to_set = [track]
            for t in tracks:
                if t.id != id and t.video == track.video:
                    if track_temporal_sim(t, track) > 0.5:
                        # Also reject t
                        to_set.append(t)
            for t in to_set:
                set_label(track, 'reject')
            return 'Success: {}'.format(value), 200
        else:
            if value == '':
                value = None
            set_label(track, value)
            return 'Success: {}'.format(value), 200

    def load_frames(track, out_height=480):
        pose = np.load(os.path.join(track.path, 'pose.npy'), mmap_mode='r')
        box_dict = {det['t']: [Box(
            *det['xywh'], det['score'], payload=PersonMeta(pose=pose[i])
        )] for i, det in enumerate(track.detections)}

        frames = []
        vc = cv2.VideoCapture(os.path.join(args.video_dir, track.video))
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

    @app.route('/track-preview/<int:id>')
    def get_track_preview(id, n=5):
        out_height = request.args.get('height', type=int, default=240)

        track = tracks[id]
        pose = np.load(os.path.join(track.path, 'pose.npy'), mmap_mode='r')
        stride = int(len(track.detections) / n)
        dets = track.detections[::stride][:n]

        vc = cv2.VideoCapture(os.path.join(args.video_dir, track.video))
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_width = int(out_height / height * width)

        frames = []
        for i, det in enumerate(dets):
            box = Box(*det['xywh'], det['score'],
                      payload=PersonMeta(pose=pose[i * stride]))

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

    @app.route('/track-gif/<int:id>')
    def get_track_gif(id):
        frames, fps = load_frames(tracks[id])
        imgs = [Image.fromarray(x) for x in frames]

        gif_bytes = BytesIO()
        img0 = imgs[0]
        img0.save(fp=gif_bytes, format='GIF',
                  append_images=imgs[1:], save_all=True,
                  duration=1 / fps * 5, loop=0)
        gif_bytes.seek(0)
        return send_file(gif_bytes, mimetype='image/gif')

    return app


def main(args):
    app = build_app(args)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, port=args.port, host='0.0.0.0',
            threaded=False, processes=os.cpu_count())


if __name__ == '__main__':
    main(get_args())
