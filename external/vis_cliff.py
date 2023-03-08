#!/usr/bin/env python3

import os
import argparse
import json
import gzip
import numpy as np
import cv2
import smplx
import torch
from tqdm import tqdm

from common import constants
from common.imutils import process_image
from common.utils import strip_prefix_if_present, cam_crop2full
from common.utils import estimate_focal_length
from common.renderer_pyrd import Renderer


DEVICE = torch.device('cuda')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('track_dir')
    parser.add_argument('video_dir')
    return parser.parse_args()


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def load_gz_json(fpath):
    with gzip.open(fpath, 'rt', encoding='ascii') as fp:
        return json.load(fp)


def main(args):
    # Setup the SMPL model
    smpl_model = smplx.create(constants.SMPL_MODEL_DIR, 'smpl').to(DEVICE)

    for video in os.listdir(args.data_dir):
        video_dir = os.path.join(args.data_dir, video)

        video_meta = load_json(os.path.join(video_dir, 'meta.json'))
        for track in tqdm(os.listdir(video_dir), desc=video):
            track_dir = os.path.join(video_dir, track)
            if not os.path.isdir(track_dir):
                continue

            label_file = os.path.join(track_dir, 'label.json')
            if (not os.path.exists(label_file)
                or load_json(label_file)['value'] != 'accept'
            ):
                continue

            process_track(
                os.path.join(args.video_dir, video_meta['video_file']),
                track_dir, args.visualize)
    print('Done!')


if __name__ == '__main__':
    main(get_args())