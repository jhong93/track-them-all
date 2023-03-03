import random
import cv2
import numpy as np


BOX_COLOR = (0, 0, 255)

LEFT_COLOR = (120, 200, 255)
RIGHT_COLOR = (120, 255, 200)

LEFT_BONES = [(5, 0), (7, 5), (9, 7), (11, 5), (13, 11), (15, 13)]
RIGHT_BONES = [(6, 0), (8, 6), (10, 8), (12, 6), (14, 12), (16, 14)]


def draw(frame, boxes, alpha=0.33, thickness=2, track_colors={}):
    has_mask = False

    for box in boxes:
        if box.track is None:
            box_color = BOX_COLOR
        else:
            if box.track not in track_colors:
                track_colors[box.track] = tuple([
                    random.randint(0, 255) for _ in range(3)])
            box_color = track_colors[box.track]

        cv2.rectangle(
            frame, (int(box.x), int(box.y)),
            (int(box.x + box.w), int(box.y + box.h)),
            box_color, thickness)
        cv2.putText(frame, '{:0.3f}'.format(box.score),
                    (int(box.x), int(box.y) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, box_color, 1)

        for bones, color in (
            (LEFT_BONES, LEFT_COLOR),
            (RIGHT_BONES, RIGHT_COLOR),
        ):
            for b1, b2 in bones:
                b1_int = (int(box.payload.pose[b1, 0]),
                          int(box.payload.pose[b1, 1]))
                b2_int = (int(box.payload.pose[b2, 0]),
                          int(box.payload.pose[b2, 1]))
                cv2.line(frame, b1_int, b2_int, color, thickness)

        if box.payload.mask is not None:
            if not has_mask:
                mask_frame = np.zeros_like(frame)
            has_mask = True

            x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)
            mask_frame[y:y + h, x:x + w, 2] = box.payload.mask * 255

    if has_mask:
        frame = cv2.addWeighted(frame, 1. - alpha, mask_frame, alpha, 0.)
    return frame