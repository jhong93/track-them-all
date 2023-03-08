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
from common.renderer_pyrd import Renderer


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
    smpl_model = smplx.create(constants.SMPL_MODEL_DIR, 'smpl')

    video_meta = load_json(os.path.join(args.track_dir, '..', 'meta.json'))
    video_file = os.path.join(args.video_dir, video_meta['video_file'])

    detections = load_gz_json(os.path.join(args.track_dir, 'meta.json.gz'))
    smpl_data = np.load(os.path.join(args.track_dir, 'smpl.npz'))

    pose = smpl_data['pose']
    betas = smpl_data['shape']
    trans = smpl_data['global_t']
    focal_l = smpl_data['focal_l']

    pred_vert_arr =[]
    for i in range(len(detections)):
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

    vc = cv2.VideoCapture(video_file)
    gap = int(1000 / vc.get(cv2.CAP_PROP_FPS))
    for i, det in enumerate(tqdm(detections)):
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

        cv2.imshow('front', front_view)
        cv2.waitKey(gap)
    vc.release()
    print('Done!')


if __name__ == '__main__':
    main(get_args())