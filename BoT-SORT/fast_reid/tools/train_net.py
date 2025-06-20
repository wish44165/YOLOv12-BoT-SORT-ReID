#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import os
import sys

sys.path.append('.')

from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fast_reid.fastreid.utils.checkpoint import Checkpointer


import shutil
from huggingface_hub import hf_hub_download

# Define your target directory
target_dir = "./logs/sbs_S50"
os.makedirs(target_dir, exist_ok=True)  # Ensure directory exists

# List of files to download
files_to_download = ["model_0016.pth", "config.yaml"]

# Download each file only if it doesn't already exist
for filename in files_to_download:
    target_path = os.path.join(target_dir, filename)
    if not os.path.exists(target_path):
        print(f"Downloading {filename}...")
        downloaded_path = hf_hub_download(
            repo_id="wish44165/YOLOv12-BoT-SORT-ReID",
            filename=filename
        )
        shutil.copy(downloaded_path, target_path)
    else:
        print(f"{filename} already exists, skipping download.")

print(f"Checked and ensured all files in: {target_dir}")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
