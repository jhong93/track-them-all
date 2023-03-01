import numpy as np
import cv2

from mmpose.apis import (
    inference_top_down_pose_model, init_pose_model, vis_pose_result)
from mmpose.datasets import DatasetInfo


MODEL = None
DATASET = None
DATASET_INFO = None


CONFIG_FILE = 'deps/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py'

CHECKPOINT_FILE = 'vitpose-h-multi-coco.pth'


def init_model(device='cuda'):
    global MODEL, DATASET, DATASET_INFO
    MODEL = init_pose_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)

    DATASET = MODEL.cfg.data['test']['type']
    DATASET_INFO = DatasetInfo(MODEL.cfg.data['test'].get('dataset_info', None))


def infer_pose(img, boxes):
    person_results = [
        {'bbox': np.array([x, y, x + w, y + h])} for x, y, w, h in boxes]
    pose_results, returned_outputs = inference_top_down_pose_model(
        MODEL,
        img,
        person_results,
        format='xyxy',
        dataset=DATASET,
        dataset_info=DATASET_INFO,
        return_heatmap=True,
        outputs=None)
    return pose_results, returned_outputs