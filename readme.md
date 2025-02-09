# CLIP for Stress Detection

This repository implements a multimodal stress detection system using CLIP (Contrastive Language-Image Pre-training) architecture adapted for physiological and behavioral data. The system can detect stress levels by analyzing Fitbit data, tabular features, and textual descriptions.

## Overview

The project uses a CLIP-inspired architecture to learn joint representations of physiological signals (from Fitbit) and textual descriptions for stress detection. It supports multiple datasets including LifeSnaps and PMData.

## Features

- Multimodal stress detection using:
  - Fitbit physiological signals
  - Tabular behavioral data
  - Textual descriptions
- Multiple encoder architectures:
  - LSTM
  - CNN
  - Transformer
  - ResNet for sequential data
  - BERT for text encoding
- Training approaches:
  - CLIP-style contrastive learning
  - Supervised baseline
  - Fine-tuning options
- Evaluation metrics and analysis tools

## Model Architecture

The system consists of several key components:
1. Signal Encoders (LSTM, CNN, Transformer, ResNet)
2. Tabular Data Encoder
3. Text Encoder (BERT)
4. Projection heads for alignment
5. Contrastive learning framework

## Requirements

- Python 3.6+
- PyTorch
- transformers
- scikit-learn
- pandas
- numpy
- captum

## Usage

### Training

```bash
python main.py --mode clip \
    --dataset lifesnaps \
    --seq_encoder resnet \
    --hidden_size 128 \
    --n_layers 4 \
    --n_epochs 100 \
    --lr 1e-4
```

### Fine-tuning

```bash
python main.py --mode finetune \
    --dataset lifesnaps \
    --seq_encoder resnet \
    --hidden_size 128 \
    --n_layers 4 \
    --n_epochs 50 \
    --lr 1e-5
```

## Ablation Studies

Run ablation experiments using:
```bash
python ablations.py --model_type clip \
    --component [encoder|projection|loss]
```

## Data

The system supports two datasets:
- LifeSnaps: A dataset containing Fitbit data and stress annotations
- PMData: Physiological monitoring dataset with stress labels
