import itertools
import time
from dataclasses import dataclass
from typing import Tuple, Callable, Dict

import cv2
import numpy
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.models._api import WeightsEnum

TENSORRT_MODEL_PATH_PATTERN = "output/model/%s/model.ts"
TEST_VIDEO_PATH = "output/video.mp4"
TEST_IMAGE_PATH = "output/image.jpg"


@dataclass
class ModelMeta:
    name: str
    input_size: Tuple[int, int, int]
    input_mean: Tuple[float, float, float]
    input_std: Tuple[float, float, float]
    load_func: Callable
    weight: WeightsEnum


MODEL_MAP: Dict[str, ModelMeta] = {
    "resnet_50": ModelMeta(
        "resnet_50",
        (3, 224, 224),
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
        resnet50,
        ResNet50_Weights.DEFAULT
    ),
    "efficientnet_v2_l": ModelMeta(
        "efficientnet_v2_l",
        (3, 480, 480),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        efficientnet_v2_l,
        EfficientNet_V2_L_Weights.DEFAULT
    ),
}


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


def preprocess(frames, model_meta: ModelMeta) -> numpy.ndarray:
    mean = 255 * numpy.array(model_meta.input_mean)
    std = 255 * numpy.array(model_meta.input_std)

    use_batch = len(frames.shape) == 4
    if not use_batch:
        frames = numpy.expand_dims(frames, 0)

    processed_frames = numpy.zeros((frames.shape[0], *model_meta.input_size[-3:]), dtype=numpy.float32)
    for i in range(frames.shape[0]):
        frame = cv2.resize(frames[i], model_meta.input_size[-2:])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose(2, 0, 1)  # HWC to CHW
        frame = (frame - mean[:, None, None]) / std[:, None, None]
        processed_frames[i] = frame

    return processed_frames


def read_frames_with_time(seconds: int, fake_decode: bool):
    shape = (1080, 1920, 3)
    random_frame = numpy.random.randint(0, 256, size=shape, dtype=numpy.uint8)
    endless_frames = itertools.cycle([random_frame]) if fake_decode else read_endless_frames()
    endless_frames = iter(endless_frames)

    start_time = time.time()
    while time.time() - start_time < seconds:
        yield next(endless_frames)


def read_input_with_time(seconds: int, model_meta: ModelMeta, inference_only: bool):
    shape = (1080, 1920, 3)
    random_frame = numpy.random.randint(0, 256, size=shape, dtype=numpy.uint8)
    random_input = preprocess(random_frame, model_meta)

    endless_inputs = (preprocess(frame, model_meta) for frame in read_endless_frames())
    endless_inputs = itertools.cycle([random_input]) if inference_only else endless_inputs
    endless_inputs = iter(endless_inputs)

    start_time = time.time()
    while time.time() - start_time < seconds:
        yield next(endless_inputs)


def cal_fps(pbar):
    frames = pbar.format_dict["n"]
    seconds = pbar.format_dict["elapsed"]
    print(f"fps: {frames / seconds:.2f}")
