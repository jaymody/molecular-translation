import os
import random
import logging

import torch
import Levenshtein
import numpy as np


def get_data_paths(data_dir):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    train_csv = os.path.join(data_dir, "train_labels.csv")
    test_csv = os.path.join(data_dir, "sample_submission.csv")
    return train_csv, test_csv, train_dir, test_dir


def path_from_image_id(image_id, image_dir):
    return os.path.join(
        image_dir, image_id[0], image_id[1], image_id[2], image_id + ".png"
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(log_name, log_file=None):
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_score(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)  # pylint: disable=no-member
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score
