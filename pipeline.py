#!/usr/bin/env python3

import os
import argparse
from typing import NamedTuple
import cv2
import numpy as np
from tqdm import trange

from vipe import load_embedding_model, preprocess_2d_keyp
import vitpose
import eva
import yolo

from util.box import Box
from util.io import store_gz_json


PERSON_CLASS_ID = 0

MIN_CONF = 0.5
MIN_HEIGHT = 0.1
MAX_HEIGHT = 0.8


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
    parser.add_argument('--out_dir')
    return parser.parse_args()


class PersonMeta(NamedTuple):
    pose: 'np.ndarray'
    mask: 'np.ndarray'
    vipe: 'np.ndarray'


def draw(frame, boxes, alpha=0.33):
    has_mask = False

    for box in boxes:
        cv2.rectangle(
            frame, (int(box.x), int(box.y)),
            (int(box.x + box.w), int(box.y + box.h)),
            BOX_COLOR, 1)
        cv2.putText(frame, '{:0.3f}'.format(box.score),
                    (int(box.x), int(box.y) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, BOX_COLOR, 2)

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


def detect_people(video_file, use_yolo, visualize):
    infer = yolo.infer if use_yolo else eva.infer

    vc = cv2.VideoCapture(video_file)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    # width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)

    all_dets = []
    for _ in trange(num_frames, desc='Detect'):
        ret, frame = vc.read()
        if not ret:
            break

        frame_dets = []
        result = infer(frame)
        for det in result:
            if int(det['class']) != PERSON_CLASS_ID:
                continue
            if det['score'] < MIN_CONF:
                continue

            x, y, w, h = det['xywh']
            h_frac = h / height
            if h_frac < MIN_HEIGHT or h_frac > MAX_HEIGHT:
                continue

            pose = vitpose.infer_pose(frame, [[x, y, w, h]])
            if len(pose) == 0:
                continue
            keyp = pose[0]['keypoints']

            keyp_vipe = preprocess_2d_keyp(keyp, flip=False)
            vipe = vipe_model.embed(keyp_vipe)[0]

            frame_dets.append(Box(
                x, y, w, h, score=det['score'],
                payload=PersonMeta(pose=keyp, mask=det['mask'], vipe=vipe)))

        if visualize:
            draw(frame, frame_dets)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

        all_dets.append(frame_dets)

    vc.release()
    if visualize:
        cv2.destroyAllWindows()
    return all_dets


def main(video_file, use_yolo, visualize, out_dir):
    if use_yolo:
        yolo.init_model()
    else:
        eva.init_model()
    vitpose.init_model()

    global vipe_model
    vipe_model = load_embedding_model('pretrained_vipe')

    detections = detect_people(video_file, use_yolo, visualize)


if __name__ == '__main__':
    main(**vars(get_args()))