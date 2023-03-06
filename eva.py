import torch
import numpy as np

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.structures.boxes import BoxMode


CONFIG_FILE = 'deps/EVA/det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py'

CHECKPOINT_FILE = 'eva_coco_seg.pth'


def init_model(device='cuda'):
    global model, augment
    cfg = LazyConfig.load(CONFIG_FILE)
    model = instantiate(cfg.model)
    model.to(device)
    DetectionCheckpointer(model).load(CHECKPOINT_FILE)
    model.eval()
    augment = T.ResizeShortestEdge([720, 720], 1280)


def infer(orig_img):
    height, width = orig_img.shape[:2]
    with torch.no_grad():
        img = augment.get_transform(orig_img).apply_image(orig_img)
        inputs = {
            'image': torch.as_tensor(img.astype('float32').transpose(2, 0, 1)),
            'width': width,
            'height': height
        }
        instances = model([inputs])[0]['instances'].to('cpu')

    results = []
    scores = instances.scores
    if len(scores) == 0:
        return results

    classes = instances.pred_classes
    boxes_XYWH = BoxMode.convert(
        instances.pred_boxes.tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).numpy()
    masks = instances.pred_masks

    for i in range(len(scores)):
        score = scores[i].item()
        box = boxes_XYWH[i, :]
        x, y, w, h = box.astype(np.int32).tolist()
        if w * h > 0:
            mask = masks[i][y:y + h, x:x + w].numpy()
            results.append({
                'score': score,
                'xywh': box.tolist(),
                'class': int(classes[i]),
                'mask': mask
            })
    return results
