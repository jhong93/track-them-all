#!/usr/bin/env python3

import os
import argparse
import random
import cv2
from tqdm import tqdm

from util.io import load_json, load_gz_json



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('video_dir')
    parser.add_argument('-o', '--out_dir')
    parser.add_argument('-l', '--limit', type=int)
    return parser.parse_args()


def render(track_dir, video_file, out_file=None):
    detections = load_gz_json(os.path.join(track_dir, 'meta.json.gz'))

    vc = cv2.VideoCapture(video_file)
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))

    if out_file is not None:
        vo = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'avc1'),
                             vc.get(cv2.CAP_PROP_FPS), (width, height))

    for i, det in enumerate(tqdm(detections)):
        t = det['t']
        if t != int(vc.get(cv2.CAP_PROP_POS_FRAMES)):
            vc.set(cv2.CAP_PROP_POS_FRAMES, t)

        _, img = vc.read()
        if out_file is not None:
            vo.write(img)
        else:
            cv2.imshow('frame', img)
            cv2.waitKey(1)

    vc.release()
    if out_file is not None:
        vo.release()


def main(args):
    videos = os.listdir(args.data_dir)
    random.shuffle(videos)

    count = 0
    for video in videos:
        video_data_dir = os.path.join(args.data_dir, video)
        video_meta = load_json(os.path.join(video_data_dir, 'meta.json'))

        tracks = os.listdir(video_data_dir)
        random.shuffle(tracks)

        for track in tracks:
            track_data_dir = os.path.join(video_data_dir, track)
            if not os.path.isdir(track_data_dir) or not os.path.exists(
                 os.path.join(track_data_dir, 'smpl.npz')
            ):
                continue

            out_file = None
            if args.out_dir is not None:
                os.makedirs(args.out_dir, exist_ok=True)
                out_file = os.path.join(args.out_dir, '{}__{}.mp4'.format(
                    video, track))
            render(track_data_dir,
                   os.path.join(args.video_dir, video_meta['video_file']),
                   out_file=out_file)

            count += 1
            if args.limit is not None and count >= args.limit:
                break
    print('Done!')


if __name__ == '__main__':
    main(get_args())