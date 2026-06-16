# FineSTAN-EEG-BCI
A Fine-grained Spatio-Temporal Attention Network for EEG-based Handwriting Imagery Decoding

## Overview
This repository provides the official PyTorch implementation of **FineSTAN**, a lightweight fine-grained spatio-temporal attention network for high-accuracy handwriting imagery EEG decoding. The model consists of four collaborative modules:
- Adaptive Channel Selection Module (ACSM)
- Multi-Scale Temporal Convolution Module (MSTCM)
- Segmented Spatio-Temporal Attention Module (SegSTA)
- Dual Statistical Gating Module (DSGM)

This repository also includes complete implementations of 11 baseline models for fair benchmark comparison.

## Environment Requirements
- Python 3.9+
- PyTorch 2.1.0
- CUDA 12.8
- MNE-Python, NumPy, SciPy, scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Repository Structure
FineSTAN-EEG-BCI/
├── FineSTAN/ # Core implementation of the proposed FineSTAN model
├── HI_BaseLine_Model/ # Complete baseline model implementations
└── jupyter_Code/ # Experimental analysis and visualization notebooks


## Datasets
This work is evaluated on three self-built handwriting imagery EEG datasets:
- CCS-HI (Chinese character strokes)
- SV-HI (Chinese pinyin)
- ELL-HI (English lowercase letters)

The ELL-HI dataset is publicly available at [Figshare](https://doi.org/10.6084/m9.figshare.32676945.v1).

## Usage
1. Download the dataset and place it in the corresponding data directory.
2. Modify the data path and hyperparameters in the configuration file.
3. Run the training script in the `FineSTAN/` directory to start model training.

## Citation
If you find this work useful, please cite our paper.
```bibtex
@article{finestan2026,
  title={FineSTAN: A Fine-grained Spatio-Temporal Attention Network for EEG-based Handwriting Imagery Decoding},
  author={Fan Wang},
  journal={},
  year={2026}
}


