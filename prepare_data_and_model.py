import torch
import logging
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch_tensorrt
from simple_parsing import choice, ArgumentParser
from torch.nn import Module
from util import MODEL_MAP, ModelMeta, TENSORRT_MODEL_PATH_PATTERN, TEST_VIDEO_PATH, \
    TEST_IMAGE_PATH


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
    model.eval()

    return model


def convert_torch_to_tensorRT(model_meta: ModelMeta, model: Module) -> None:
    logging.info("Converting Model to TensorRT...")
    model_ts = TENSORRT_MODEL_PATH_PATTERN % (model_meta.name,)
    inputs = [
        torch_tensorrt.Input(shape=[1, *model_meta.input_size], dtype=torch.float),
    ]
    trt_ts_module = torch_tensorrt.compile(model, inputs=inputs)
    Path(model_ts).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(trt_ts_module, model_ts)


@dataclass
class Args:
    model: str = choice(*MODEL_MAP.keys(), "all", alias=["-m"], default="resnet_50")


def main(args: Args) -> None:
    download_video_and_image()
    models = MODEL_MAP.keys() if args.model == "all" else [args.model]

    for model_name in models:
        model_meta = MODEL_MAP[model_name]
        model = download_model(model_meta)
        convert_torch_to_tensorRT(model_meta, model)


def parse_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="arguments")
    return parser.parse_args(args).arguments


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
