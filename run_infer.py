import os
import sys
from dataclasses import dataclass
from typing import List

import torch
import torch_tensorrt  # keep this line for load TensorRT model.
import tqdm
from simple_parsing import choice, flag, field, ArgumentParser
from torch import nn
from tqdm import tqdm

from util import PYTORCH_TRT_MODEL_PATH_PATTERN, ModelMeta, MODEL_MAP, read_input_with_time, cal_fps


@dataclass
class Args:
    model: str = choice(*MODEL_MAP.keys(), alias=["-m"], default="resnet_50")
    model_type: str = choice("fp32", "fp16", "int8", alias=["-mt"], default="int8")
    device: str = field(alias=["-d"], default="cuda:0")  # The device used for OpenVINO: CPU, GPU, MULTI:CPU,GPU ...
    inference_only: bool = flag(alias=["-io"], default=False)
    run_mode: str = choice("sync", "async", "multi", "one_decode_multi", alias=["-rm"], default="sync")
    n_stream: int = field(alias=["-n"], default=os.cpu_count())
    duration: int = field(alias=["-t"], default=60)


def sync_infer(args: Args, model: nn.Module, model_meta: ModelMeta) -> list:
    outputs = []
    with tqdm(unit="frame") as pbar:
        for frame in read_input_with_time(args.duration, model_meta, args.inference_only):
            input_tensor = torch.tensor(frame).to(torch.device(args.device))
            output = model(input_tensor)
            outputs.append(output)
            pbar.update(1)

    cal_fps(pbar)
    return outputs


def load_model(model_meta: ModelMeta, model_type: str, device: str):
    model_ts = PYTORCH_TRT_MODEL_PATH_PATTERN % (model_meta.name, model_type)
    torch_device = torch.device(device)
    model = torch.jit.load(model_ts).to(torch_device)
    return model


def main(args: Args) -> None:
    model_meta = MODEL_MAP[args.model]
    model = load_model(model_meta, args.model_type, args.device)
    globals()[f"{args.run_mode}_infer"](args, model, model_meta)


def parse_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="arguments")
    return parser.parse_args(args).arguments


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
