# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
from tqdm import tqdm

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    with open('datasets/douban_book_review/dev.tsv') as f:
        lines = f.readlines()

    prompt = """豆瓣评论：加油，赶紧创作出剧本吧。
情感：正面
###
豆瓣评论：看了个开头，完全没有看下去的欲望。
情感：负面
###
豆瓣评论："""
    acc, counter = 0, 0
    prompts = []
    for l in lines[1:1000]:
        label, text = l.strip().split('\t')
        label_text = "正面" if label == '1' else "负面"
        prompts.append(prompt+text[:128]+'\n'+"情感：")


    for i in tqdm(range(0, len(prompts), 32)):
        if (len(prompts[i:i+32])) == 0:
            break
        results = generator.generate(
            prompts[i*32:(i+1)*32], max_gen_len=512, temperature=temperature, top_p=top_p
        )

        for j, r in enumerate(results):
            answer = r.split('###')[2].strip().split('\n')[1].split('：')[1]
            label = lines[i + j].strip().split('\t')[0]
            label_text = "正面" if label == '1' else "负面"
            if label_text == answer:
                acc += 1
            counter += 1

    print(acc, counter)



if __name__ == "__main__":
    fire.Fire(main)
