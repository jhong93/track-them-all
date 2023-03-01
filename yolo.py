from ultralytics import YOLO


def init_model():
    global yolo_model
    yolo_model = YOLO('yolov8x')
    yolo_model.info()


def infer(img):
    dets = []
    result = yolo_model([img])[0]
    for box in result.boxes:
        x, y, x2, y2 = box.xyxy[0].tolist()
        w, h = x2 - x, y2 -y
        conf = box.conf.item()
        dets.append({
            'xywh': [x, y, w, h],
            'score': conf,
            'class': int(box.cls),
            'mask': None    # Use detection YOLO for now
        })
    return dets