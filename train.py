import os
import gc
import sys
import random
import argparse

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    GPUStatsMonitor,
    ModelCheckpoint,
)

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data import TrainDataset, ValidDataset, TestDataset, BMSCollator, get_transforms
from models import ImageCaptioner
from tokenizers import Tokenizer, split_form, split_form2
from utils import (
    get_data_paths,
    path_from_image_id,
    set_seed,
    get_logger,
    get_device,
    get_score,
)

tqdm.pandas()
device = get_device()

scale = 1
output_dir = "models/_test"
input_dir = "../input/bms-molecular-translation"
log_wandb = True
debug = False

config = {
    "gpus": 1,
    "max_len": 275,
    "size": 224,
    "num_workers": 8,
    "model_name": "resnet34",
    "scheduler": "CosineAnnealingLR",
    "epochs": 1,
    "encoder_lr": 1e-4 * scale,
    "decoder_lr": 4e-4 * scale,
    "min_lr": 1e-6 * scale,
    "batch_size": 64 * scale,
    "weight_decay": 1e-6,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 5,
    "attention_dim": 256,
    "embed_dim": 256,
    "decoder_dim": 512,
    "dropout": 0.5,
    "seed": 42,
    "valid_size": 0.01,
}


def preprocess(train_csv, train_dir, output_dir):
    print("... loading train data ...")
    train_df = pd.read_csv(train_csv)

    print("... building filepaths ...")
    train_df["file_path"] = train_df["image_id"].progress_apply(
        lambda x: path_from_image_id(x, train_dir)
    )

    print("... building text sequences ...")
    train_df["InChI_1"] = train_df["InChI"].progress_apply(lambda x: x.split("/")[1])
    train_df["InChI_text"] = (
        train_df["InChI_1"].progress_apply(split_form)
        + " "
        + train_df["InChI"]
        .apply(lambda x: "/".join(x.split("/")[2:]))
        .progress_apply(split_form2)
        .values
    )

    print("... fitting tokenizer ...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df["InChI_text"].values)
    torch.save(tokenizer, os.path.join(output_dir, "tokenizer2.pth"))
    print(tokenizer.stoi)
    print("Saved tokenizer to " + os.path.join(output_dir, "tokenizer2.pkl"))

    print("... writing df to pickle ...")
    lengths = []
    tk0 = tqdm(train_df["InChI_text"].values, total=len(train_df))
    for text in tk0:
        seq = tokenizer.text_to_sequence(text)
        length = len(seq) - 2
        lengths.append(length)
    train_df["InChI_length"] = lengths
    train_df.to_pickle(os.path.join(output_dir, "train2.pkl"))

    print("Saved preprocessed to " + os.path.join(output_dir, "train2.pkl"))
    print(train_df)


def train(name, output_dir):
    print("\n... loading data ...")
    df = pd.read_pickle(os.path.join(output_dir, "train2.pkl"))
    print(df)

    print("\n... performing train/validation split ...")
    train_df, valid_df = train_test_split(
        df, shuffle=True, test_size=config["valid_size"]
    )
    print("train_size:", len(train_df))
    print("test_size:", len(valid_df))

    print("\n... loading tokenizer ...")
    tokenizer = torch.load(os.path.join(output_dir, "tokenizer2.pth"))
    print(tokenizer.stoi)

    print("\n... creating datasets ...")
    train_dataset = TrainDataset(
        train_df, tokenizer, transform=get_transforms(data="train", size=config["size"])
    )
    valid_dataset = ValidDataset(
        valid_df, transform=get_transforms(data="valid", size=config["size"])
    )

    bms_collator = BMSCollator(pad_idx=tokenizer.stoi["<pad>"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=bms_collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    print("\n... creating model ...")
    model = ImageCaptioner(
        tokenizer=tokenizer,
        valid_labels=valid_df["InChI"].values,
        **config,
        device=device,
    )

    print("\n... training ...")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_score",
        dirpath=output_dir,
        filename="best_model",
        save_last=True,
        save_top_k=1,
        mode="min",
    )

    logger = True
    if not debug and log_wandb:
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            save_dir=output_dir,
            offline=False,
            project=os.environ["WANDB_PROJECT"],
            log_model=False,
            group=name,
        )

    # NOTE: the gradient_clip_val and accumulate_grad_batches params in trainer
    # have no affect since we are doing manual optimization and need to implement
    # that ourselves (and as such we pass it into the model)
    trainer = pl.Trainer(
        default_root_dir=output_dir,  # set directory to save weights, logs, etc ...
        num_processes=config["num_workers"],  # num processes to use if using cpu
        gpus=config["gpus"],  # num gpus to use if using gpu
        # tpu_cores=None,  # num tpu cores to use if using tpu
        progress_bar_refresh_rate=1,  # change to 20 if using google colab
        fast_dev_run=debug,  # set to True to quickly verify your code works
        max_epochs=config["epochs"],
        min_epochs=1,
        # max_steps=None,  # use if you want to train based on step rather than epoch
        # min_steps=None,  # use if you want to train based on step rather than epoch
        limit_train_batches=1.0 / 100,  # percentage of train data to use
        limit_val_batches=1.0 / 100,  # percentage of validation data to use
        limit_test_batches=1.0,  # percentage of test data to use
        check_val_every_n_epoch=1,  # run validation every n epochs
        val_check_interval=0.20,  # run validation after every n percent of an epoch
        precision=32,  # use 16 for half point precision
        # resume_from_checkpoint=None,  # place path to checkpoint if resuming training
        # auto_lr_find=False,  # set to True to optimize learning rate
        # auto_scale_batch_size=False,  # set to True to find largest batch size that fits in hardware
        log_every_n_steps=int(100 / scale),
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("step"),
            GPUStatsMonitor(temperature=True, fan_speed=True),
        ],
        logger=logger,
    )
    trainer.fit(model, train_loader, valid_loader)

    print("\n... done training :) ...")


if __name__ == "__main__":
    train_csv, test_csv, train_dir, test_dir = get_data_paths(input_dir)

    if os.path.isdir(output_dir):
        print("... skipping proprocessing since output_dir already exists ...\n")
    else:
        print("--------------------------------")
        print("---------- preprocess ----------")
        print("--------------------------------")
        os.makedirs(output_dir)
        preprocess(train_csv, train_dir, output_dir)

    print("---------------------------")
    print("---------- train ----------")
    print("---------------------------")
    name = os.path.basename(output_dir)
    train(name, output_dir)
