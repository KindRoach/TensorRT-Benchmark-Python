import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

from dataclasses import dataclass
from simple_parsing import choice, field, ArgumentParser
from tqdm import tqdm

from util import read_frames_with_time, cal_fps_from_tqdm


@dataclass
class Args:
    run_mode: str = choice("sync", "multi", alias=["-rm"], default="sync")
    n_stream: int = field(alias=["-n"], default=os.cpu_count())
    duration: int = field(alias=["-t"], default=60)


def sync_decode(args: Args) -> None:
    with tqdm(unit="frame") as pbar:
        for frame in read_frames_with_time(args.duration):
            pbar.update(1)

    cal_fps_from_tqdm(pbar)


def multi_decode(args: Args) -> None:
    with tqdm(unit="frame") as pbar:
        def decode_stream(thread_id: int):
            for frame in read_frames_with_time(args.duration):
                pbar.update(1)

        with ThreadPoolExecutor(args.n_stream) as pool:
            for i in range(args.n_stream):
                pool.submit(decode_stream, i)

    cal_fps_from_tqdm(pbar)


def main(args: Args) -> None:
    globals()[f"{args.run_mode}_decode"](args)


def parse_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="arguments")
    return parser.parse_args(args).arguments


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
