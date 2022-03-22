import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from albumentations import Compose, Normalize, Resize, Transpose, VerticalFlip
from albumentations.pytorch import ToTensorV2


def get_transforms(*, data, size):
    if data == "train":
        return Compose(
            [
                Resize(size, size),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return Compose(
            [
                Resize(size, size),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


# CPU Memory Leak Issue: https://github.com/pytorch/pytorch/issues/13246
class TrainDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None):
        super().__init__()
        self.file_paths = np.array(df["file_path"].values)
        self.labels = (
            df["InChI_text"]
            .apply(lambda x: np.array(tokenizer.text_to_sequence(x)))
            .values
        )
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        label = self.labels[idx]
        label_length = len(label)
        return image, label, label_length


class ValidDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.file_paths = np.array(df["file_path"].values)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.file_paths = np.array(df["file_path"].values)
        self.transform = transform
        self.fix_transform = Compose([Transpose(p=1), VerticalFlip(p=1)])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w, _ = image.shape
        if h > w:
            image = self.fix_transform(image=image)["image"]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image


class BMSCollator:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs, labels, label_lengths = list(zip(*batch))

        imgs = torch.stack(imgs)
        labels = pad_sequence(
            [torch.LongTensor(lbl) for lbl in labels],
            batch_first=True,
            padding_value=self.pad_idx,
        )
        label_lengths = torch.stack(
            [torch.LongTensor([lbl_len]) for lbl_len in label_lengths]
        ).reshape(-1, 1)

        return imgs, labels, label_lengths
