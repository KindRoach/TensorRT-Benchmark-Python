import itertools
import time
from typing import Tuple

import cv2
import numpy

TEST_VIDEO_PATH = "output/video.mp4"
TEST_IMAGE_PATH = "output/image.jpg"

PYTORCH_TRT_MODEL_PATH_PATTERN = "output/model/%s/%s/model.ts"
ONNX_MODEL_PATH_PATTERN = "output/model/%s/model.onnx"
TRTEXEC_MODEL_PATH_PATTERN = "output/model/%s/%s/model.plan"
MODEL_CFG_PATH_PATTERN = "output/model/%s/model.json"

MODEL_LIST = [
    "resnet18",
    "resnet50",
    "resnet101",
    "resnet152",
]


def read_endless_frames():
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    assert cap.isOpened()

    while True:
        success, frame = cap.read()
        if success:
            yield frame
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


def read_all_frames():
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    assert cap.isOpened()
    while True:
        success, frame = cap.read()
        if success:
            yield frame
        else:
            break

    cap.release()


def preprocess(frames, input_shape: Tuple, input_mean: Tuple, input_std: Tuple) -> numpy.ndarray:
    mean = 255 * numpy.array(input_mean)
    std = 255 * numpy.array(input_std)

    use_batch = len(frames.shape) == 4
    if not use_batch:
        frames = numpy.expand_dims(frames, 0)

    batch_size = frames.shape[0]
    processed_frames = numpy.zeros((batch_size, *input_shape), dtype=numpy.float32)
    for i in range(batch_size):
        frame = frames[i]
        if frame.shape[:2] != input_shape[-2:]:
            frame = cv2.resize(frames[i], input_shape[-2:])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose(2, 0, 1)  # HWC to CHW
        frame = (frame - mean[:, None, None]) / std[:, None, None]
        processed_frames[i] = frame

    return processed_frames


def read_frames_with_time(seconds: int):
    endless_frames = iter(read_endless_frames())

    start_time = time.time()
    while time.time() - start_time < seconds:
        yield next(endless_frames)


def read_input_with_time(seconds: int, input_shape: Tuple, input_mean: Tuple, input_std: Tuple, inference_only: bool):
    shape = (1080, 1920, 3)
    random_frame = numpy.random.randint(0, 256, size=shape, dtype=numpy.uint8)
    random_input = preprocess(random_frame, input_shape, input_mean, input_std)

    endless_inputs = (preprocess(frame, input_shape, input_mean, input_std) for frame in read_endless_frames())
    endless_inputs = itertools.cycle([random_input]) if inference_only else endless_inputs
    endless_inputs = iter(endless_inputs)

    start_time = time.time()
    while time.time() - start_time < seconds:
        yield next(endless_inputs)


def cal_fps_from_tqdm(pbar):
    frames = pbar.format_dict["n"]
    seconds = pbar.format_dict["elapsed"]
    print(f"fps: {frames / seconds:.2f}")
