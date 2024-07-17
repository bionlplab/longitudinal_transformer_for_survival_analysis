# Harnessing the power of longitudinal medical imaging for eye disease prognosis using Transformer-based sequence modeling

## Overview

Deep learning has enabled breakthroughs in automated diagnosis from medical imaging, with many successful applications in ophthalmology. However, standard medical image classification approaches only assess disease presence at the time of acquisition, neglecting the common clinical setting of longitudinal imaging. For slow, progressive eye diseases like age-related macular degeneration (AMD) and primary open-angle glaucoma (POAG), patients undergo repeated imaging over time to track disease progression and forecasting the future risk of developing a disease is critical to properly plan treatment. Our proposed Longitudinal Transformer for Survival Analysis (LTSA) enables dynamic disease prognosis from longitudinal medical imaging, modeling the time to disease from sequences of fundus photography images captured over long, irregular time periods. Using longitudinal imaging data from the Age-Related Eye Disease Study (AREDS) and Ocular Hypertension Treatment Study (OHTS), LTSA significantly outperformed a single-image baseline in 19/20 head-to-head comparisons on late AMD prognosis and 18/20 comparisons on POAG prognosis. A temporal attention analysis also suggested that, while the most recent image is typically the most influential, prior imaging still provides additional prognostic value.

## Usage

To reproduce results:
```
# Create + activate environment
conda env create -f ltsa.yml
conda activate ltsa

# Train LTSA on AREDS
python train.py --results_dir results --dataset AREDS --model LTSA --dropout 0.25 --augment --reduce_lr --batch_size 32

# Train LTSA on OHTS
python train.py --results_dir results --dataset OHTS --model LTSA --dropout 0.25 --augment --reduce_lr --batch_size 32

# Train baseline on AREDS
python train.py --results_dir results --dataset AREDS --model image --dropout 0.25 --augment --reduce_lr --batch_size 448

# Train baseline on OHTS
python train.py --results_dir results --dataset OHTS --model image --dropout 0.25 --augment --reduce_lr --batch_size 448
```

## Citation

If you find our work helpful, please cite:

  Holste G, Lin M, Zhou R, Wang F, Liu L, Yan Q, Van Tassel SH, Kovacs K, Chew EY, Lu Z, Wang Z. Harnessing the power of longitudinal medical imaging for eye disease prognosis using Transformer-based sequence modeling. arXiv preprint arXiv:2405.08780. 2024 May 14.

## Acknowledgement

This work is supported by the National Eye Institue under Award No. R21EY035296 and National Science Foundation under Award No. 2145640 and 2306556.
