import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class ASVSpoofDataset(BaseDataset):
    """
    Dataset for ASVspoof 2019 Logical Access countermeasure task.

    Labels:
        bonafide -> 1
        spoof    -> 0
    """

    PROTOCOL_FILES = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }

    AUDIO_DIRS = {
        "train": "ASVspoof2019_LA_train/flac",
        "dev": "ASVspoof2019_LA_dev/flac",
        "eval": "ASVspoof2019_LA_eval/flac",
    }

    def __init__(
        self,
        data_root,
        part,
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        max_frames=600,
        random_crop=True,
        max_items_per_label=None,
        seed=42,
        num_crops=1,
        *args,
        **kwargs,
    ):
        self.data_root = Path(data_root)

        if not self.data_root.is_absolute():
            self.data_root = ROOT_PATH / self.data_root

        self.part = part
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.max_frames = max_frames
        self.random_crop = random_crop
        self.max_items_per_label = max_items_per_label
        self.seed = seed
        self.num_crops = num_crops

        if self.part not in self.PROTOCOL_FILES:
            raise ValueError(
                f"Unknown part: {self.part}. "
                f"Expected one of {list(self.PROTOCOL_FILES)}"
            )

        if self.num_crops < 1:
            raise ValueError("num_crops must be >= 1")

        index = self._create_index()
        super().__init__(index, *args, **kwargs)

    def _create_index(self):
        protocol_path = (
            self.data_root
            / "ASVspoof2019_LA_cm_protocols"
            / self.PROTOCOL_FILES[self.part]
        )
        audio_dir = self.data_root / self.AUDIO_DIRS[self.part]

        if not protocol_path.exists():
            raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        with open(protocol_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        if self.max_items_per_label is not None:
            rng = random.Random(self.seed)
            rng.shuffle(lines)

        index = []
        label_counts = {0: 0, 1: 0}

        for line in lines:
            fields = line.strip().split()

            if not fields:
                continue

            speaker_id = fields[0]
            utterance_id = fields[1]
            label_text = fields[-1]

            if label_text == "bonafide":
                label = 1
            elif label_text == "spoof":
                label = 0
            else:
                label = -1

            if self.max_items_per_label is not None and label in label_counts:
                if label_counts[label] >= self.max_items_per_label:
                    continue

            audio_path = audio_dir / f"{utterance_id}.flac"

            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            index.append(
                {
                    "path": str(audio_path),
                    "label": label,
                    "speaker_id": speaker_id,
                    "utterance_id": utterance_id,
                }
            )

            if self.max_items_per_label is not None and label in label_counts:
                label_counts[label] += 1

                if all(
                    count >= self.max_items_per_label
                    for count in label_counts.values()
                ):
                    break

        return index

    def __getitem__(self, ind):
        item = self._index[ind]

        data_object = self.load_object(item["path"])
        label = torch.tensor(item["label"], dtype=torch.long)

        instance_data = {
            "data_object": data_object,
            "labels": label,
            "utterance_id": item["utterance_id"],
        }

        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def load_object(self, path):
        waveform, sample_rate = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )

        waveform = waveform.squeeze(0)
        waveform = waveform - waveform.mean()
        waveform = waveform / (waveform.abs().max() + 1e-8)

        window = torch.hann_window(self.win_length)

        spectrogram = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        spectrogram = torch.log1p(spectrogram.abs().pow(2))

        if self.num_crops == 1:
            spectrogram = self._fix_time_length(spectrogram)
            spectrogram = self._normalize_spectrogram(spectrogram)
            return spectrogram.unsqueeze(0).float()

        crops = self._make_time_crops(spectrogram)
        crops = torch.stack(
            [self._normalize_spectrogram(crop) for crop in crops],
            dim=0,
        )

        return crops.unsqueeze(1).float()

    def _normalize_spectrogram(self, spectrogram):
        return (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)

    def _fix_time_length(self, spectrogram):
        current_frames = spectrogram.shape[1]

        if current_frames == self.max_frames:
            return spectrogram

        if current_frames > self.max_frames:
            if self.part == "train" and self.random_crop:
                start = torch.randint(
                    low=0,
                    high=current_frames - self.max_frames + 1,
                    size=(1,),
                ).item()
            else:
                start = (current_frames - self.max_frames) // 2

            return spectrogram[:, start : start + self.max_frames]

        pad_frames = self.max_frames - current_frames
        return F.pad(spectrogram, (0, pad_frames))

    def _make_time_crops(self, spectrogram):
        current_frames = spectrogram.shape[1]

        if current_frames <= self.max_frames:
            fixed = self._fix_time_length(spectrogram)
            return [fixed for _ in range(self.num_crops)]

        max_start = current_frames - self.max_frames

        if self.num_crops == 1:
            starts = [max_start // 2]
        else:
            starts = torch.linspace(
                0,
                max_start,
                steps=self.num_crops,
            ).long().tolist()

        crops = []

        for start in starts:
            crop = spectrogram[:, start : start + self.max_frames]
            crops.append(crop)

        return crops
