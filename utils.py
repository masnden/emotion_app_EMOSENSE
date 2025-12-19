import cv2
import numpy as np
from collections import deque

SEQUENCE_LENGTH = 20
FRAME_SIZE = (64, 64)

frame_buffer = deque(maxlen=SEQUENCE_LENGTH)


def preprocess_frame(frame):
    frame = cv2.resize(frame, FRAME_SIZE)
    frame = frame / 255.0
    return frame


def get_video_sequence(frame):
    """
    Mengumpulkan 20 frame dan mengembalikan tensor:
    (1, 20, 64, 64, 3)
    """
    processed = preprocess_frame(frame)
    frame_buffer.append(processed)

    if len(frame_buffer) < SEQUENCE_LENGTH:
        return None

    seq = np.array(frame_buffer)
    seq = np.expand_dims(seq, axis=0)
    return seq
