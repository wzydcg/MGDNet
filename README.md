# MGDNet: Multi-Granularity Difference-aware Network for Pulmonary Lesion Segmentation

[![stars - MGDNet](https://img.shields.io/github/stars/wzydcg/MGDNet?style=social)](https://github.com/wzydcg/MGDNet)
[![forks - MGDNet](https://img.shields.io/github/forks/wzydcg/MGDNet?style=social)](https://github.com/wzydcg/MGDNet)
![language](https://img.shields.io/github/languages/top/wzydcg/MGDNet?color=lightgrey)
![license](https://img.shields.io/github/license/wzydcg/MGDNet)
---

## Introduction
![running.jpg](picture/running.jpg)
## Approach

![model.jpg](picture/model.jpg)

## Frequency-Aware Attention
![FAA.jpg](picture/FAA.jpg)

## Conditional-Enhanced Attention
![CEA1.jpg](picture/CEA1.jpg)

## Experimental results
![vision1.jpg](picture/vision1.jpg)

## Training

### Default Scripts
All default hyperparameters among these models are tuned for Lung lesion datasets.

Wandb is needed if visualization of training parameters is wanted

### Customized Execution

run script like this:
```bash
python main.py \
--model MGDNet \
--dataset RAOS \
--batch_size 4 \
--num_epochs 200 \
--learning_rate 1e-4 \
--dropout 0.1 \
--do_train \
--do_evaluate
```

## Dependencies
- python==3.12
- opencv-python==4.7.0.68
- einops
- nilearn==0.10.4
- scikit-learn==1.3.2
- scipy
- torch==2.3.0
- pydicom==2.4.4
- pandas==1.5.3
- nibabel==5.2.1
- wandb

