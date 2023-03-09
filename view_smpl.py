#!/usr/bin/env python3

import os
import argparse
import random
import numpy as np
import cv2
import smplx
import torch
from tqdm import tqdm

from util.renderer_pyrd import Renderer
from util.io import load_gz_json, load_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('video_dir')
    return parser.parse_args()


def render(track_dir, video_file):
    detections = load_gz_json(os.path.join(track_dir, 'meta.json.gz'))
    smpl_data = np.load(os.path.join(track_dir, 'smpl.npz'))

    pose = smpl_data['pose']
    betas = smpl_data['shape']
    trans = smpl_data['global_t']
    focal_l = smpl_data['focal_l']

    pred_vert_arr = []
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

        cv2.imshow('front', cv2.cvtColor(front_view, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
    vc.release()


def main(args):
    # Setup the SMPL model
    global smpl_model
    smpl_model = smplx.create('data', 'smpl')

    videos = os.listdir(args.data_dir)
    random.shuffle(videos)

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

            render(track_data_dir,
                   os.path.join(args.video_dir, video_meta['video_file']))
    print('Done!')


if __name__ == '__main__':
    main(get_args())