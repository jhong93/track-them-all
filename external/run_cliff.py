#!/usr/bin/env python3

import os
import argparse
import json
import gzip
import numpy as np
import cv2
import smplx
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.cliff_hr48.cliff import CLIFF

from common import constants
from common.imutils import process_image
from common.utils import strip_prefix_if_present, cam_crop2full
from common.utils import estimate_focal_length
from common.renderer_pyrd import Renderer


CHECKPOINT_FILE = 'data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt'
DEVICE = torch.device('cuda')

BATCH_SIZE = 32


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('video_dir')
    parser.add_argument('-v', '--visualize', action='store_true')
    return parser.parse_args()


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def load_gz_json(fpath):
    with gzip.open(fpath, 'rt', encoding='ascii') as fp:
        return json.load(fp)


class TrackDataset(Dataset):

    def __init__(self, video_file, detections):
        assert os.path.isfile(video_file)
        self.video_file = video_file
        self.detections = detections
        self.base_frame = min(x['t'] for x in detections)
        self.vc = None

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, idx):
        if self.vc is None:
            self.vc = cv2.VideoCapture(self.video_file)

        det = self.detections[idx]
        t = det['t']
        if t != int(self.vc.get(cv2.CAP_PROP_POS_FRAMES)):
            self.vc.set(cv2.CAP_PROP_POS_FRAMES, t)

        ret, img_bgr = self.vc.read()
        if idx == len(self.detections) - 1:
            self.vc.release()

        img_rgb = img_bgr[:, :, ::-1]
        img_h, img_w, _ = img_rgb.shape
        focal_length = estimate_focal_length(img_h, img_w)

        xywh = det['xywh']
        xyxy = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
        norm_img, center, scale, crop_ul, crop_br, _ = process_image(
            img_rgb, xyxy)

        item = {}
        item['norm_img'] = norm_img
        item['center'] = center
        item['scale'] = scale
        item['crop_ul'] = crop_ul
        item['crop_br'] = crop_br
        item['img_h'] = img_h
        item['img_w'] = img_w
        item['focal_length'] = focal_length
        return item


def process_track(video_file, track_dir, visualize):
    detections = load_gz_json(os.path.join(track_dir, 'meta.json.gz'))
    dataset = TrackDataset(video_file, detections)
    loader = DataLoader(
        dataset, batch_size=min(BATCH_SIZE, len(detections)))

    pred_vert_arr = []
    smpl_pose = []
    smpl_betas = []
    smpl_trans = []
    smpl_joints = []
    cam_focal_l = []

    for batch in tqdm(loader):
        norm_img = batch['norm_img'].to(DEVICE).float()
        center = batch['center'].to(DEVICE).float()
        scale = batch['scale'].to(DEVICE).float()
        img_h = batch['img_h'].to(DEVICE).float()
        img_w = batch['img_w'].to(DEVICE).float()
        focal_length = batch['focal_length'].to(DEVICE).float()

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        # The constants below are used for normalization, and calculated from H36M data.
        # It should be fine if you use the plain Equation (5) in the paper.
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(
            -1) * 2.8  # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (
            0.06 * focal_length)  # [-1, 1]

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_cam_crop = cliff_model(
                norm_img, bbox_info)

        # convert the camera parameters from the crop camera to the full camera
        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)

        pred_output = smpl_model(betas=pred_betas,
                                 body_pose=pred_rotmat[:, 1:],
                                 global_orient=pred_rotmat[:, [0]],
                                 pose2rot=False,
                                 transl=pred_cam_full)
        pred_vertices = pred_output.vertices
        pred_vert_arr.extend(pred_vertices.cpu().numpy())

        smpl_pose.extend(pred_rotmat.cpu().numpy())
        smpl_betas.extend(pred_betas.cpu().numpy())
        smpl_trans.extend(pred_cam_full.cpu().numpy())
        smpl_joints.extend(pred_output.joints.cpu().numpy())
        cam_focal_l.extend(focal_length.cpu().numpy())

    result_path = os.path.join(track_dir, 'smpl.npz')
    print(f'Save results to \"{result_path}\"')
    np.savez_compressed(
        result_path,
        pose=smpl_pose, shape=smpl_betas, global_t=smpl_trans,
        pred_joints=smpl_joints, focal_l=cam_focal_l)

    del smpl_pose
    del smpl_betas
    del smpl_trans
    del smpl_joints
    del cam_focal_l

    if visualize:
        vc = cv2.VideoCapture(video_file)
        gap = int(1000 / vc.get(cv2.CAP_PROP_FPS))
        for i, det in enumerate(detections):
            t = det['t']
            if t != int(vc.get(cv2.CAP_PROP_POS_FRAMES)):
                vc.set(cv2.CAP_PROP_POS_FRAMES, t)

            _, img_bgr = vc.read()
            img_h, img_w, _ = img_bgr.shape
            focal_length = estimate_focal_length(img_h, img_w)
            renderer = Renderer(
                focal_length=focal_length, img_w=img_w, img_h=img_h,
                faces=smpl_model.faces,
                same_mesh_color=True)
            front_view = renderer.render_front_view(
                pred_vert_arr[i:i + 1],
                bg_img_rgb=img_bgr[:, :, ::-1].copy())
            renderer.delete()

            cv2.imshow('front', front_view)
            cv2.waitKey(gap)
        vc.release()


def main(args):
    global cliff_model, smpl_model

    # Create the model instance
    cliff_model = CLIFF(constants.SMPL_MEAN_PARAMS).to(DEVICE)

    # Load the pretrained model
    print('Load the CLIFF checkpoint from path:', CHECKPOINT_FILE)
    state_dict = torch.load(CHECKPOINT_FILE)['model']
    state_dict = strip_prefix_if_present(state_dict, prefix='module.')
    cliff_model.load_state_dict(state_dict, strict=True)
    cliff_model.eval()
    del state_dict

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