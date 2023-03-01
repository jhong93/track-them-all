import cv2
from collections import namedtuple


VideoMetadata = namedtuple('VideoMetadata', [
    'fps', 'num_frames', 'width', 'height'
])


def _get_metadata(vc):
    fps = vc.get(cv2.CAP_PROP_FPS)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoMetadata(fps, num_frames, width, height)


def get_metadata(video_path):
    vc = cv2.VideoCapture(video_path)
    try:
        return _get_metadata(vc)
    finally:
        vc.release()