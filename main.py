import argparse
import os
from pathlib import Path

import pytorch_metric_learning.losses as losses
import pytorch_metric_learning.miners as miners
import torch
from tqdm import tqdm

import utils
from retrieval_model import DinoWrapper

torch.multiprocessing.set_sharing_strategy("file_system")  # Due to annoying "RuntimeError: Too many open files."

DATASETS = ["Cars196", "CUB", "INaturalist2018", "StanfordOnlineProducts"]

ALL_LOSSES = [
    "AngularLoss",
    "ArcFaceLoss",
    "BaseMetricLossFunction",
    "CircleLoss",
    "ContrastiveLoss",
    "CosFaceLoss",
    "DynamicSoftMarginLoss",
    "FastAPLoss",
    "GenericPairLoss",
    "HistogramLoss",
    "InstanceLoss",
    "IntraPairVarianceLoss",
    "LargeMarginSoftmaxLoss",
    "GeneralizedLiftedStructureLoss",
    "LiftedStructureLoss",
    "ManifoldLoss",
    "MarginLoss",
    "WeightRegularizerMixin",
    "MultiSimilarityLoss",
    "MultipleLosses",
    "NPairsLoss",
    "NCALoss",
    "NormalizedSoftmaxLoss",
    "NTXentLoss",
    "P2SGradLoss",
    "PNPLoss",
    "ProxyAnchorLoss",
    "ProxyNCALoss",
    "RankedListLoss",
    "SelfSupervisedLoss",
    "SignalToNoiseRatioContrastiveLoss",
    "SoftTripleLoss",
    "SphereFaceLoss",
    "SubCenterArcFaceLoss",
    "SupConLoss",
    "ThresholdConsistentMarginLoss",
    "TripletMarginLoss",
    "TupletMarginLoss",
    "VICRegLoss",
]

CLASSIFICATION_LOSSES = [
    "ArcFaceLoss",
    "CosFaceLoss",
    "LargeMarginSoftmaxLoss",
    "WeightRegularizerMixin",
    "NormalizedSoftmaxLoss",
    "ProxyAnchorLoss",
    "ProxyNCALoss",
    "SoftTripleLoss",
    "SphereFaceLoss",
    "SubCenterArcFaceLoss",
]

ALL_MINERS = [
    "no_miner",
    "AngularMiner",
    "BatchEasyHardMiner",
    "BatchHardMiner",
    "DistanceWeightedMiner",
    "HDCMiner",
    "MultiSimilarityMiner",
    "PairMarginMiner",
    "TripletMarginMiner",
    "UniformHistogramMiner",
]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--lr", type=float, default=1e-6, help="LR")
parser.add_argument("--classifier_lr", type=float, default=1.0, help="classifier LR (only for classification losses)")
parser.add_argument("--sampler_m", type=int, default=4, help="M value for MPerClassSampler")
parser.add_argument("--dataset", type=str, default="Cars196", choices=DATASETS, help="dataset")
parser.add_argument("--dino_size", type=str, default="b", choices=["s", "b", "l", "g"], help="which Dino to use")
parser.add_argument("--loss", type=str, default="MultiSimilarityLoss", choices=ALL_LOSSES, help="loss")
parser.add_argument("--miner", type=str, default="MultiSimilarityMiner", choices=ALL_MINERS, help="miner")
parser.add_argument("--feat_dim", type=int, default=None, help="Output dimensionality. Set to None to use CLS")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="device")
parser.add_argument("--optim", type=str, default="adam", choices=["adam", "rmsprop"], help="optimizer")
parser.add_argument(
    "--save_dir",
    type=Path,
    default=Path("default"),
    help="name of directory in which to save the logs, under logs/save_dir",
)
args = parser.parse_args()

utils.initialize_logger(args)

model = DinoWrapper(dino_size=args.dino_size, feat_dim=args.feat_dim)
model = model.to(args.device)

train_loader, valid_loader, test_loader, train_labels_mapper = utils.setup_datasets(
    args.dataset, args.batch_size, args.sampler_m
)

if args.optim == "adam":
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "rmsprop":
    optim = torch.optim.RMSprop(model.parameters(), lr=args.lr)

if args.loss in CLASSIFICATION_LOSSES:
    # The loss is a classification loss with a learnable matrix, like ArcFaceLoss
    criterion = getattr(losses, args.loss)(len(set(train_loader.dataset.labels)), model.feat_dim)
    is_classification = True
    if args.optim == "adam":
        classifier_optim = torch.optim.Adam(criterion.parameters(), lr=args.classifier_lr)
    elif args.optim == "rmsprop":
        classifier_optim = torch.optim.RMSprop(criterion.parameters(), lr=args.classifier_lr)
else:
    # The loss is a standard contrastive loss with no learnable parameter, like Contrastive or Triplet
    criterion = getattr(losses, args.loss)()
    is_classification = False

if not is_classification:
    if args.miner == "no_miner":
        miner = None
    else:
        miner = getattr(miners, args.miner)()

# Evaluate off-the-shelf model
utils.evaluate(model, valid_loader, "valid")

patience = 3
best_precision = 0
epochs_no_improve = 0

for num_epoch in range(100):
    tqdm_bar = tqdm(train_loader)
    for images, labels in tqdm_bar:
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            # Set map labels to start from 0 for classification losses like ArcFaceLoss
            labels = torch.tensor([train_labels_mapper[int(label)] for label in labels]).to(args.device)
            embeddings = model(images.to(args.device))

            if not is_classification and miner is not None:
                miner_outputs = miner(embeddings, labels)
                loss = criterion(embeddings, labels, miner_outputs)
            else:
                loss = criterion(embeddings, labels)

        loss.backward()
        optim.step()
        optim.zero_grad()
        if is_classification:
            classifier_optim.step()
            classifier_optim.zero_grad()
        tqdm_bar.desc = f"{loss = :.5f}"

    cur_precision, _ = utils.evaluate(model, valid_loader, f"valid - epoch {num_epoch:>2}")

    if cur_precision > best_precision:
        best_precision = cur_precision
        epochs_no_improve = 0
        torch.save(model.state_dict(), args.log_dir / "best_model.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            model.load_state_dict(torch.load(args.log_dir / "best_model.pth", weights_only=True))
            break

utils.evaluate(model, test_loader, "test")
os.remove(args.log_dir / "best_model.pth")
