import json
import logging
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List

import numpy
import timm
import torch

import torch_tensorrt
logging.getLogger().setLevel(logging.WARNING)

from dataclasses import dataclass
from simple_parsing import choice, ArgumentParser
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset

from util import read_all_frames, preprocess, TEST_VIDEO_PATH, TEST_IMAGE_PATH, ONNX_MODEL_PATH_PATTERN, \
    TRTEXEC_MODEL_PATH_PATTERN, PYTORCH_TRT_MODEL_PATH_PATTERN, MODEL_LIST, MODEL_CFG_PATH_PATTERN


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


def convert_tensorRT_by_torchTRT(model: Module) -> None:
    logging.info("Converting Model to TensorRT...")

    cfg = model.pretrained_cfg
    model_name = cfg["architecture"]
    input_shape = [1, *cfg["input_size"]]
    inputs = [torch_tensorrt.Input(shape=input_shape, dtype=torch.float)]
    dummy_input = [torch.randn(*input_shape, dtype=torch.float).cuda()]

    model.eval()
    model.cuda()

    # fp32
    trt_ts_module = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions={torch.float, })
    trt_traced_model = torch.jit.trace(trt_ts_module, dummy_input)
    model_ts = PYTORCH_TRT_MODEL_PATH_PATTERN % (model_name, "fp32")
    Path(model_ts).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(trt_traced_model, model_ts)

    # fp16
    trt_ts_module = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions={torch.float, torch.half, })
    trt_traced_model = torch.jit.trace(trt_ts_module, dummy_input)
    model_ts = PYTORCH_TRT_MODEL_PATH_PATTERN % (model_name, "fp16")
    Path(model_ts).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(trt_traced_model, model_ts)

    # int8
    frames = []
    for frame in read_all_frames():
        frame = preprocess(frame, cfg["input_size"], cfg["mean"], cfg["std"])[0]
        frames.append(frame)

    frames = torch.tensor(numpy.array(frames))
    calib_dataloader = DataLoader(TensorDataset(frames), batch_size=1)

    calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
        calib_dataloader,
        use_cache=False,
        algo_type=torch_tensorrt.ptq.CalibrationAlgo.MINMAX_CALIBRATION,
        device=torch.device('cuda:0')
    )

    trt_ts_module = torch_tensorrt.compile(
        model, inputs=inputs,
        enabled_precisions={torch.float, torch.half, torch.int8},
        calibrator=calibrator,
        device="cuda:0")

    trt_traced_model = torch.jit.trace(trt_ts_module, dummy_input)
    model_ts = PYTORCH_TRT_MODEL_PATH_PATTERN % (model_name, "int8")
    Path(model_ts).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(trt_traced_model, model_ts)


def convert_tensorRT_by_trtexec(model: Module) -> None:
    logging.info("Converting Model to TensorRT...")

    cfg = model.pretrained_cfg
    model_name = cfg["architecture"]
    input_shape = [1, *cfg["input_size"]]

    model.eval()
    model.cuda()

    # convert to onnx
    onnx_path = ONNX_MODEL_PATH_PATTERN % model_name
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(*input_shape).cuda()
    torch.onnx.export(model, (dummy_input,), onnx_path)

    # convert to tensorrt engine (FP32)
    plan_path = TRTEXEC_MODEL_PATH_PATTERN % (model_name, "fp32")
    subprocess.run([
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}"
    ])

    # convert to tensorrt engine (FP16)
    plan_path = TRTEXEC_MODEL_PATH_PATTERN % (model_name, "fp16")
    subprocess.run([
        "trtexec",
        "--fp16",
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}"
    ])


@dataclass
class Args:
    model: str = choice(*MODEL_LIST, "all", alias=["-m"], default="resnet50")


def main(args: Args) -> None:
    download_video_and_image()
    models = MODEL_LIST if args.model == "all" else [args.model]

    for model_name in models:
        model = timm.create_model(model_name, pretrained=True)
        with open(MODEL_CFG_PATH_PATTERN % model_name, "w", encoding="utf-8") as f:
            json.dump(model.pretrained_cfg, f)
        convert_tensorRT_by_torchTRT(model)
        convert_tensorRT_by_trtexec(model)


def parse_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="arguments")
    return parser.parse_args(args).arguments


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
