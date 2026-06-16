## Included Baseline Models
We extensively evaluate the performance of the proposed FineSTAN against **11 state-of-the-art EEG decoding baseline models**:
1. EEGNet
2. ShallowConvNet
3. DeepConvNet
4. FBCNet
5. CTNet
6. SATransNet
7. ATCNet
8. EEGConformer
9. MCTD
10. EEG-EA
11. EEG-TransNet

Lightweight Jupyter notebook demos for four classic CNN-based baselines (EEGNet, FBCNet, ShallowConvNet, DeepConvNet) are provided in the `jupyter_Code` folder, validated on the CCS-HI and SV-HI datasets. All 11 baseline algorithms are fully reproduced in our experimental scripts for fair cross-lingual comparison.

### Code Structure for Baseline Reproduction
1. `model/` folder: Contains full network definition files for all 11 baseline models, including layer construction, forward propagation and model initialization logic.
2. Baseline running scripts:
   Each baseline model is equipped with independent batch training & testing scripts (prefixed with `HI_`) to accommodate model-specific hyperparameters, input processing and training strategies.
   - `*_Batch training.py`: Batch train the target model on Session 1 across all subjects
   - `*_Batch testing.py`: Evaluate trained model on fully unseen Session 2 for cross-session generalization analysis
3. `utils.py`: Shared utility functions for BIDS dataset loading, signal preprocessing, classification metric calculation and result saving.
4. `Model_complexity.py`: Calculate model parameter count and inference latency for complexity comparison in our manuscript.
5. `jupyter_Code/`: Lightweight Jupyter notebook demos for quick validation of classic CNN baselines (EEGNet, FBCNet).


## Requirements
* Python ≥ 3.9
* PyTorch ≥ 2.1.0
* mne == 1.11.0
* numpy
* einops
* pyyaml
* matplotlib
* jupyter

## Datasets
All baseline models and the proposed FineSTAN are validated on our unified cross-lingual handwriting imagery EEG benchmark suite:
* CCS-HI & SV-HI (Chinese strokes & Pinyin vowels): https://doi.org/10.6084/m9.figshare.29987758.v4  
  Formal dataset paper: Wang F, et al. An EEG dataset for handwriting imagery decoding of Chinese character strokes and Pinyin single vowels[J]. Scientific Data, 2026.
* ELL-HI (English lowercase letter handwriting imagery): https://doi.org/10.6084/m9.figshare.32676945.v1

## Results
Comprehensive quantitative comparisons, ablation studies, parameter analysis, and feature visualization results are reported in our submitted manuscript.

## Citation
If you use this code, baseline implementations, or datasets in your research, please cite our work.

### Main Manuscript (FineSTAN Model)

