import os
import sys
from typing import List

import numpy as np
import tensorrt as trt
from cuda import cudart
from dataclasses import dataclass
from simple_parsing import choice, flag, field, ArgumentParser
from tqdm import tqdm

from util import ModelMeta, MODEL_MAP, TRTEXEC_MODEL_PATH_PATTERN, read_input_with_time, cal_fps


@dataclass
class Args:
    model: str = choice(*MODEL_MAP.keys(), alias=["-m"], default="resnet_50")
    model_type: str = choice("fp32", "fp16", alias=["-mt"], default="fp32")
    device: int = field(alias=["-d"], default="0")  # The CUDA device id used for TensorRT ...
    inference_only: bool = flag(alias=["-io"], default=False)
    run_mode: str = choice("sync", "async", "multi", "one_decode_multi", alias=["-rm"], default="sync")
    n_stream: int = field(alias=["-n"], default=os.cpu_count())
    duration: int = field(alias=["-t"], default=60)


def sync_infer(args: Args, engine: trt.ICudaEngine, model_meta: ModelMeta) -> list:
    outputs = []

    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    input_shape = context.get_tensor_shape(input_name)
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    input_nbytes = np.empty(input_shape, dtype=input_dtype).nbytes
    input_buffer_D = cudart.cudaMalloc(input_nbytes)[1]

    output_name = engine.get_tensor_name(1)
    output_shape = context.get_tensor_shape(output_name)
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
    output_buffer_H = np.empty(output_shape, dtype=output_dtype)
    output_buffer_D = cudart.cudaMalloc(output_buffer_H.nbytes)[1]

    with tqdm(unit="frame") as pbar:
        for frame in read_input_with_time(args.duration, model_meta, args.inference_only):
            cudart.cudaMemcpy(
                input_buffer_D,
                frame.ctypes.data,
                frame.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            )

            context.execute_v2([input_buffer_D, output_buffer_D])

            cudart.cudaMemcpy(
                output_buffer_H.ctypes.data,
                output_buffer_D,
                output_buffer_H.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
            )

            outputs.append(output_buffer_H.copy())
            pbar.update(1)

    cudart.cudaFree(input_buffer_D)
    cudart.cudaFree(output_buffer_D)

    cal_fps(pbar)
    return outputs


def load_model(model_meta: ModelMeta, model_type: str):
    model_plan = TRTEXEC_MODEL_PATH_PATTERN % (model_meta.name, model_type)
    with open(model_plan, "rb") as f:
        engine_string = f.read()
    logger = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_string)
    return engine


def main(args: Args) -> None:
    cudart.cudaSetDevice(args.device)
    model_meta = MODEL_MAP[args.model]
    model = load_model(model_meta, args.model_type)
    globals()[f"{args.run_mode}_infer"](args, model, model_meta)


def parse_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="arguments")
    return parser.parse_args(args).arguments


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
