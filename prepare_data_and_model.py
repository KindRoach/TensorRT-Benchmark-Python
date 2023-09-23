import logging
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List

import numpy
import torch
import torch_tensorrt
logging.getLogger().setLevel(logging.WARNING)
from dataclasses import dataclass
from simple_parsing import choice, ArgumentParser
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset

from util import MODEL_MAP, ModelMeta, PYTORCH_TRT_MODEL_PATH_PATTERN, TEST_VIDEO_PATH, \
    TEST_IMAGE_PATH, read_all_frames, preprocess, ONNX_MODEL_PATH_PATTERN, TRTEXEC_MODEL_PATH_PATTERN


def download_file(url: str, target_path: str) -> None:
    if not Path(target_path).exists():
        logging.info(f"Downloading to {target_path} ...")
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, target_path)


def download_video_and_image() -> None:
    download_file(
        "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_30MB.mp4",
        TEST_VIDEO_PATH
    )

    download_file(
        "https://storage.openvinotoolkit.org/data/test_data/images/dog.jpg",
        TEST_IMAGE_PATH
    )


def download_model(model: ModelMeta) -> Module:
    logging.info("Downloading Model...")

    weight = model.weight
    load_func = model.load_func

    model = load_func(weights=weight)
    return model.eval().cuda()


def convert_tensorRT_by_torchTRT(model_meta: ModelMeta, model: Module) -> None:
    logging.info("Converting Model to TensorRT...")
    inputs = [
        torch_tensorrt.Input(shape=[1, *model_meta.input_size], dtype=torch.float),
    ]

    # fp32
    trt_ts_module = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions={torch.float, })
    model_ts = PYTORCH_TRT_MODEL_PATH_PATTERN % (model_meta.name, "fp32")
    Path(model_ts).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(trt_ts_module, model_ts)

    # fp16
    trt_ts_module = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions={torch.float, torch.half, })
    model_ts = PYTORCH_TRT_MODEL_PATH_PATTERN % (model_meta.name, "fp16")
    Path(model_ts).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(trt_ts_module, model_ts)

    # int8
    frames = []
    for frame in read_all_frames():
        frame = preprocess(frame, model_meta)[0]
        frames.append(frame)

    frames = torch.tensor(numpy.array(frames))
    dataloader = DataLoader(TensorDataset(frames), batch_size=1)

    calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
        dataloader,
        cache_file="./calibration.cache",
        use_cache=False,
        algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=torch.device("cuda:0"),
    )

    trt_ts_module = torch_tensorrt.compile(
        model, inputs=inputs,
        enabled_precisions={torch.float, torch.half, torch.int8},
        calibrator=calibrator,
        device={
            "device_type": torch_tensorrt.DeviceType.GPU,
            "gpu_id": 0,
            "dla_core": 0,
            "allow_gpu_fallback": False,
            "disable_tf32": False
        })

    model_ts = PYTORCH_TRT_MODEL_PATH_PATTERN % (model_meta.name, "int8")
    Path(model_ts).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(trt_ts_module, model_ts)


def convert_tensorRT_by_trtexec(model_meta: ModelMeta, model: Module) -> None:
    # convert to onnx
    onnx_path = ONNX_MODEL_PATH_PATTERN % model_meta.name
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    torch.onnx.export(model, (dummy_input,), onnx_path)

    # convert to tensorrt engine (FP32)
    plan_path = TRTEXEC_MODEL_PATH_PATTERN % (model_meta.name, "fp32")
    subprocess.run([
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}"
    ])

    # convert to tensorrt engine (FP16)
    plan_path = TRTEXEC_MODEL_PATH_PATTERN % (model_meta.name, "fp16")
    subprocess.run([
        "trtexec",
        "--fp16",
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}"
    ])


@dataclass
class Args:
    model: str = choice(*MODEL_MAP.keys(), "all", alias=["-m"], default="resnet_50")


def main(args: Args) -> None:
    download_video_and_image()
    models = MODEL_MAP.keys() if args.model == "all" else [args.model]

    for model_name in models:
        model_meta = MODEL_MAP[model_name]
        model = download_model(model_meta)
        convert_tensorRT_by_torchTRT(model_meta, model)
        convert_tensorRT_by_trtexec(model_meta, model)


def parse_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="arguments")
    return parser.parse_args(args).arguments


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
