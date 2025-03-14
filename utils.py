import copy
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_metric_learning.datasets as datasets
import pytorch_metric_learning.samplers as samplers
import torch
import torchvision.transforms as tfm
import torchvision.transforms.v2 as v2
from loguru import logger
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def initialize_logger(args):
    start_time = datetime.now()
    logger.remove()
    args.log_dir = Path("logs") / args.save_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(args.log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(args.log_dir / "debug.log", level="DEBUG")
    sys.excepthook = lambda _, value, tb: logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(f"The outputs are being saved in {args.log_dir}")


def setup_datasets(dataset_name, batch_size, sampler_m):
    train_transform = tfm.Compose(
        [
            v2.RGB(),
            tfm.Resize(size=(224, 224), antialias=True),
            tfm.RandAugment(num_ops=3, interpolation=tfm.InterpolationMode.BILINEAR),
            tfm.ToTensor(),
            tfm.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    test_transform = tfm.Compose(
        [
            v2.RGB(),
            tfm.Resize(size=(224, 224), antialias=True),
            tfm.ToTensor(),
            tfm.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    data_root = f"data/{dataset_name}"
    download = not os.path.exists(data_root)  # If dataset does not exist, download it
    train_val_dataset = getattr(datasets, dataset_name)(
        root=data_root, split="train", transform=train_transform, download=download
    )
    test_dataset = getattr(datasets, dataset_name)(
        root=data_root, split="test", transform=test_transform, download=download
    )

    train_dataset, valid_dataset, train_labels_mapper = split_dataset_by_classes(train_val_dataset)
    valid_dataset.dataset.transform = test_transform

    logger.info(
        f"Train size: {len(train_dataset)}, Validation size: {len(valid_dataset)}, Test size: {len(test_dataset)}"
    )

    sampler = samplers.MPerClassSampler(train_dataset.labels, m=sampler_m, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, sampler=sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=32, num_workers=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=8, shuffle=False)
    return train_loader, valid_loader, test_loader, train_labels_mapper


def split_dataset_by_classes(train_val_dataset, split_ratio=0.8):
    # Get unique classes and shuffle them
    unique_classes = np.unique(train_val_dataset.labels)
    # Determine split point
    split_point = int(len(unique_classes) * split_ratio)
    train_classes = set(unique_classes[:split_point])
    val_classes = set(unique_classes[split_point:])
    # Split indices by class
    train_indices = [i for i, label in enumerate(train_val_dataset.labels) if label in train_classes]
    val_indices = [i for i, label in enumerate(train_val_dataset.labels) if label in val_classes]
    # We need to deepcopy the dataset so that the two subsets can use different transforms
    train_dataset = Subset(copy.deepcopy(train_val_dataset), train_indices)
    val_dataset = Subset(copy.deepcopy(train_val_dataset), val_indices)
    # Assign labels to train_dataset. Necessary for the sampler
    train_dataset.orig_labels = [train_val_dataset.labels[i] for i in train_indices]
    # Remap indexes so that they start from 0
    train_labels_mapper = {label: i for i, label in enumerate(sorted(set(train_dataset.orig_labels)))}
    train_dataset.labels = [train_labels_mapper[label] for label in train_dataset.orig_labels]
    assert min(train_dataset.labels) == 0
    assert max(train_dataset.labels) == len(set(train_dataset.orig_labels)) - 1

    return train_dataset, val_dataset, train_labels_mapper


def evaluate(model, eval_loader, name="test set", device="cuda"):
    model = model.eval()
    all_embeddings = []
    all_labels = []
    # Extract embeddings and labels
    with torch.no_grad():
        for images, labels in tqdm(eval_loader, desc=name):
            embeddings = model(images.to(device))
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
            all_labels.append(labels.cpu())
    # Concatenate all embeddings and labels
    all_embeddings = np.concatenate(all_embeddings)
    all_labels = np.concatenate(all_labels)
    # Use AccuracyCalculator to compute metrics
    accuracy_calculator = AccuracyCalculator(
        include=("precision_at_1", "mean_average_precision_at_r"), k="max_bin_count", device=torch.device("cpu")
    )
    accuracy = accuracy_calculator.get_accuracy(all_embeddings, all_labels)
    precision_at_1, mean_average_precision_at_r = accuracy["precision_at_1"], accuracy["mean_average_precision_at_r"]
    logger.info(f"{name}: Precision@1 = {precision_at_1*100:.1f} , MAP@R = {mean_average_precision_at_r*100:.1f}")
    return precision_at_1, mean_average_precision_at_r
