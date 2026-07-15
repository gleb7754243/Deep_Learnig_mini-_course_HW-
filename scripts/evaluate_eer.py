import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import ASVSpoofDataset
from src.metrics.eer import compute_eer
from src.model import LightCNN


def load_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format")

    clean_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]

        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]

        clean_state_dict[new_key] = value

    model.load_state_dict(clean_state_dict, strict=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="data/raw/LA/LA")
    parser.add_argument("--part", default="dev", choices=["train", "dev", "eval"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_items_per_label", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_kwargs = {
        "data_root": args.data_root,
        "part": args.part,
        "random_crop": False,
    }

    if args.max_items_per_label is not None:
        dataset_kwargs["max_items_per_label"] = args.max_items_per_label

    if args.limit is not None:
        dataset_kwargs["limit"] = args.limit

    dataset = ASVSpoofDataset(**dataset_kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = LightCNN()
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    all_scores = []
    all_labels = []
    all_utterance_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {args.part}"):
            data_object = batch["data_object"].to(device)
            labels = batch["labels"]

            outputs = model(data_object=data_object)
            logits = outputs["logits"]

            probabilities = torch.softmax(logits, dim=-1)
            bonafide_scores = probabilities[:, 1].detach().cpu()

            all_scores.extend(bonafide_scores.tolist())
            all_labels.extend(labels.tolist())
            all_utterance_ids.extend(batch["utterance_id"])

    scores_tensor = torch.tensor(all_scores)
    labels_tensor = torch.tensor(all_labels)

    valid_mask = labels_tensor >= 0

    if valid_mask.sum().item() > 0:
        eer = compute_eer(scores_tensor[valid_mask], labels_tensor[valid_mask])
        print(f"Full-set EER on {args.part}: {eer:.6f}")
    else:
        print("Labels are not available, EER was not computed.")

    print(f"Processed examples: {len(all_scores)}")

    if args.output_csv is not None:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for utterance_id, score in zip(all_utterance_ids, all_scores):
                writer.writerow([utterance_id, score])

        print(f"Scores saved to: {output_path}")


if __name__ == "__main__":
    main()
