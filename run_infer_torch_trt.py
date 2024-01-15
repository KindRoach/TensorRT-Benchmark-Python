import json
import logging
import os
import sys
from typing import List

import torch

# keep this import for load TensorRT models.
# noinspection PyUnresolvedReferences
import torch_tensorrt
logging.getLogger().setLevel(logging.WARNING)

import tqdm
from dataclasses import dataclass
from simple_parsing import choice, flag, field, ArgumentParser
from torch import nn
from tqdm import tqdm

from util import PYTORCH_TRT_MODEL_PATH_PATTERN, read_input_with_time, cal_fps_from_tqdm, MODEL_LIST, \
    MODEL_CFG_PATH_PATTERN


@dataclass
class Args:
    model: str = choice(*MODEL_LIST, alias=["-m"], default="resnet50")
    model_type: str = choice("fp32", "fp16", "int8", alias=["-mt"], default="fp32")
    device: str = field(alias=["-d"], default="cuda:0")  # The device used for TensorRT: CUDA:0 ...
    inference_only: bool = flag(alias=["-io"], default=False)
    run_mode: str = choice("sync", "async", "multi", "one_decode_multi", alias=["-rm"], default="sync")
    n_stream: int = field(alias=["-n"], default=os.cpu_count())
    duration: int = field(alias=["-t"], default=60)


def sync_infer(args: Args, model: nn.Module, model_cfg: dict) -> list:
    outputs = []
    with tqdm(unit="frame") as pbar:
        frames = read_input_with_time(
            args.duration,
            model_cfg["input_size"],
            model_cfg["mean"],
            model_cfg["std"],
            args.inference_only
        )
        for frame in frames:
            input_tensor = torch.tensor(frame).to(torch.device(args.device))
            output = model(input_tensor)
            outputs.append(output)
            pbar.update(1)

    cal_fps_from_tqdm(pbar)
    return outputs


def load_model(model_name: str, model_type: str, device: str):
    model_ts = PYTORCH_TRT_MODEL_PATH_PATTERN % (model_name, model_type)
    torch_device = torch.device(device)
    model = torch.jit.load(model_ts).to(torch_device)
    return model


def main(args: Args) -> None:
    model = load_model(args.model, args.model_type, args.device)
    with open(MODEL_CFG_PATH_PATTERN % args.model, "r", encoding="utf-8") as f:
        model_cfg = json.load(f)
    globals()[f"{args.run_mode}_infer"](args, model, model_cfg)


def parse_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="arguments")
    return parser.parse_args(args).arguments


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
