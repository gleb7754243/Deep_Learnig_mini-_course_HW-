# ASVspoof 2019 LA Voice Anti-Spoofing with LightCNN

This project is a PyTorch solution for voice anti-spoofing on the **ASVspoof 2019 Logical Access (LA)** dataset. It detects whether an audio recording is real human speech or a synthetic / voice-converted spoof.

The model is based on a custom LightCNN-style architecture with Max-Feature-Map layers. I did not use a ready-made LightCNN implementation; the network used here was written for this project.

## Task

Each recording is classified into one of two classes:

- `bonafide` - genuine human speech;
- `spoof` - synthetic or voice-converted speech.

The main metric is **Equal Error Rate (EER)**. Lower EER means better performance.

## Approach

The pipeline is straightforward:

```text
FLAC audio
-> waveform normalization
-> STFT power spectrogram
-> crop or zero padding in time
-> spectrogram standardization
-> LightCNN
-> bonafide/spoof score
```

The final version uses:

- ASVspoof 2019 LA train, development and evaluation partitions;
- STFT-based acoustic features;
- a custom LightCNN/LCNN-style model;
- Max-Feature-Map convolutional and linear layers;
- dropout before the final Batch Normalization layer;
- weighted Cross-Entropy loss;
- Weights & Biases for experiment tracking;
- full-partition EER calculation;
- three-crop inference for the final predictions.

## Environment

The project was tested with:

```text
Python 3.11
PyTorch 2.8.0
CUDA 12.8
NVIDIA GeForce RTX 5070 Ti
```

Create and activate a virtual environment on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Check CUDA availability:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Dataset

Download the **ASVspoof 2019** dataset and use the Logical Access partition.

The code expects the LA data under `data/raw/LA/LA/`, including the train, development and evaluation FLAC folders, plus the official protocol files:

```text
ASVspoof2019.LA.cm.train.trn.txt
ASVspoof2019.LA.cm.dev.trl.txt
ASVspoof2019.LA.cm.eval.trl.txt
```

Dataset sizes used by the protocol files:

```text
Train: 25,380 recordings
Dev:   24,844 recordings
Eval:  71,237 recordings
```

The dataset itself is not included in this repository.

## Feature Extraction

For each recording, the loader:

1. reads the FLAC file;
2. converts audio to mono;
3. resamples it to 16 kHz if needed;
4. centers and peak-normalizes the waveform;
5. builds an STFT power spectrogram;
6. applies `log1p`;
7. crops or pads the spectrogram to a fixed length;
8. standardizes it by mean and standard deviation.

Main STFT parameters:

```text
sample_rate = 16000
n_fft       = 512
win_length  = 400
hop_length  = 160
max_frames  = 600
```

One model input has shape:

```text
[1, 257, 600]
```

## Model

The model is a LightCNN/LCNN-style binary classifier. It uses Max-Feature-Map blocks, Batch Normalization, Max Pooling, adaptive average pooling, a Max-Feature-Map linear embedding, dropout and a final two-class output layer.

The output logits are ordered as:

```text
class 0: spoof
class 1: bonafide
```

## Loss

The final run uses weighted Cross-Entropy:

```text
spoof weight:    1.0
bonafide weight: 8.0
```

This helps compensate for the class imbalance in the full training partition.

## One-Batch Overfit Check

Before full training, I checked the whole pipeline on a tiny balanced subset:

```text
8 spoof + 8 bonafide recordings
```

The model reached:

```text
Train accuracy: 1.0
Validation accuracy: 1.0
Test accuracy: 1.0
```

This was a sanity check that the dataset reading, labels, model, loss and training loop worked together correctly.

## Training

The final tracked experiment can be launched with:

```powershell
python train.py datasets=asvspoof_full_train model=lcnn metrics=asvspoof_eer trainer.n_epochs=8 trainer.epoch_len=397 dataloader.batch_size=64 dataloader.num_workers=0 writer.mode=online writer.project_name=asvspoof_lcnn_hw writer.run_name=final_lcnn_weighted_v2 trainer.override=true trainer.monitor='min test_loss' trainer.save_period=1 optimizer.lr=0.0003 loss_function._target_=src.loss.WeightedCrossEntropyLoss +loss_function.spoof_weight=1.0 +loss_function.bonafide_weight=8.0 transforms.instance_transforms.train=null transforms.instance_transforms.inference=null transforms.batch_transforms.train=null transforms.batch_transforms.inference=null
```

One epoch is approximately one full pass over the 25,380 training recordings.

Checkpoints are saved locally under:

```text
saved/final_lcnn_weighted_v2/
```

## Weights & Biases

Experiment tracking includes training and validation losses, accuracy, gradient norm, learning rate and batch-level EER as an auxiliary metric.

W&B project with training charts: [Weights & Biases dashboard](https://wandb.ai/gopenai7754243-hse-university/asvspoof_lcnn_hw?nw=nwusergopenai7754243)

Batch-level EER during training is only a rough indicator. The final metric is calculated from all scores on the full partition.

## Full-Set EER Evaluation

Calculate EER on the full development partition:

```powershell
python scripts/evaluate_eer.py --checkpoint saved/final_lcnn_weighted_v2/model_best.pth --part dev --batch_size 32 --num_crops 3 --output_csv outputs/dev_final_v2_multicrop3.csv
```

Result:

```text
Development EER: 0.031795
Development EER: 3.1795%
Processed recordings: 24,844
```

## Evaluation Predictions

Generate predictions for the evaluation partition:

```powershell
python scripts/evaluate_eer.py --checkpoint saved/final_lcnn_weighted_v2/model_best.pth --part eval --batch_size 32 --num_crops 3 --output_csv outputs/eval_final_v2_multicrop3.csv
```

Three-crop inference takes three evenly spaced time crops from a sufficiently long spectrogram. The model predicts a bonafide probability for each crop, and the final recording score is the average of these probabilities.

The output CSV has no header:

```text
utterance_id,score
```

Example:

```text
LA_E_1000147,0.8234
LA_E_1000273,0.1021
```

## Grading Result

The final CSV was checked with the official `grading.py` script.

```text
Evaluation EER: approximately 7.68%
Performance grade: 6.6
```

The script successfully processed all 71,237 evaluation recordings.

## Current Results

| Evaluation set | Inference | EER |
|---|---:|---:|
| Full development | 1 crop | 3.2193% |
| Full development | 3 crops | 3.1795% |
| Full development | 5 crops | 3.1795% |
| Full evaluation | 3 crops | approximately 7.68% |

Three crops were selected because they gave the same development EER as five crops while requiring less computation.

## Metric Note

Accuracy uses one fixed decision threshold. EER looks at the balance between false accepts and false rejects across thresholds.

Because of that, accuracy and EER can move differently. The official result is based on full-set EER, not on batch accuracy or averaged batch EER.

## Reproducibility

The main random seed is:

```text
seed = 1
```

Balanced dataset subsets use:

```text
seed = 42
```

To reproduce the final result:

1. install the required environment;
2. place the ASVspoof 2019 LA dataset under the expected local path;
3. run the final training command;
4. use `model_best.pth`;
5. generate predictions with three-crop inference;
6. validate the CSV with the official grading script.
