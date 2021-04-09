import os
import argparse

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import TestDataset, get_transforms
from utils import get_device, get_data_paths, path_from_image_id
from models import ImageCaptioner

device = get_device()


def predict(
    input_dir,
    output_dir,
    model_ckpt="last.ckpt",
    n_samples=None,
    num_workers=4,
    batch_size=64,
):
    _, test_csv, _, test_dir = get_data_paths(input_dir)

    print("... loading data ...")
    test_df = pd.read_csv(test_csv)
    test_df["file_path"] = test_df["image_id"].apply(
        lambda x: path_from_image_id(x, test_dir)
    )
    tokenizer = torch.load(os.path.join(output_dir, "tokenizer2.pth"))

    if n_samples:
        print(f"\n... reducing predict size to {n_samples}")
        test_df = test_df.head(n_samples)

    print("\n... loading model ...")
    model = ImageCaptioner.load_from_checkpoint(
        os.path.join(output_dir, model_ckpt),
        tokenizer=tokenizer,
        valid_labels=None,
        device=device,
    )

    print("\n... creating datasets ...")
    test_dataset = TestDataset(
        test_df, transform=get_transforms(data="valid", size=model.hparams.size)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    print("\n... predicting ...")
    model.to(device)
    model.eval()

    predictions = []
    for images in tqdm(test_loader, total=len(test_loader)):
        images = images.to(device)
        predictions.extend(model.predict(images))

    print("\nfirst 10 predictions")
    print(predictions[:10])

    print("\nlast 10 predictions")
    print(predictions[-10:])

    print("\n... saving predictions to submission.csv ...")
    test_df[["image_id", "InChI"]].to_csv(
        os.path.join(output_dir, "submission.csv"), index=False
    )
    test_df[["image_id", "InChI"]].head()

    print("... done :) ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create submission.csv predictions.")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_ckpt", type=str, default="last.ckpt")
    parser.add_argument(
        "--input_dir", type=str, default="../input/bms-molecular-translation"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    predict(**args.__dict__)
