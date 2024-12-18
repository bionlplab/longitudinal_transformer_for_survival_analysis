# Harnessing the power of longitudinal medical imaging for eye disease prognosis using Transformer-based sequence modeling

[**Gregory Holste**](https://gholste.me), Mingquan Lin, Ruiwen Zhou, Fei Wang, Lei Liu, Qi Yan, Sarah H Van Tassel, Kyle Kovacs, Emily Y Chew, Zhiyong Lu, Zhangyang Wang, [**Yifan Peng**](https://penglab.weill.cornell.edu/team/yifan-peng)

### [npj Digital Medicine](https://www.nature.com/articles/s41746-024-01207-4) | 16 August 2024

## Overview

<p align=center>
    <img src=figs/Fig2.png height=500>
</p>

**Overview of proposed longitudinal survival analysis approach.** In longitudinal medical imaging, patients undergo repeated imaging over long periods of time at irregular intervals **(a)**. Rather than predict the presence of disease at the time of imaging, our method leverages a patient’s longitudinal imaging history to forecast the future risk of developing disease through a survival analysis framework **(b)**. Our approach represents the collection of fundus images for an eye over time as a sequence fit for modeling with Transformers. To accommodate large, irregular intervals between consecutive visits, a temporal positional encoder fuses this information with the image embeddings from each visit. A Transformer encoder then employs causal temporal attention over the sequence, only attending to prior visits. The entire model is optimized end-to-end to predict the time-varying hazard function for each unique sequence of consecutive visits. From the hazard function, we compute eye-specific survival curves, allowing for dynamic eye disease risk prognosis evaluated through the framework of longitudinal survival analysis **(c)**.

## Abstract

Deep learning has enabled breakthroughs in automated diagnosis from medical imaging, with many successful applications in ophthalmology. However, standard medical image classification approaches only assess disease presence at the time of acquisition, neglecting the common clinical setting of longitudinal imaging. For slow, progressive eye diseases like age-related macular degeneration (AMD) and primary open-angle glaucoma (POAG), patients undergo repeated imaging over time to track disease progression and forecasting the future risk of developing a disease is critical to properly plan treatment. Our proposed **Longitudinal Transformer for Survival Analysis (LTSA)** enables dynamic disease prognosis from longitudinal medical imaging, modeling the time to disease from sequences of fundus photography images captured over long, irregular time periods. Using longitudinal imaging data from the Age-Related Eye Disease Study (AREDS) and Ocular Hypertension Treatment Study (OHTS), LTSA significantly outperformed a single-image baseline in 19/20 head-to-head comparisons on late AMD prognosis and 18/20 comparisons on POAG prognosis. A temporal attention analysis also suggested that, while the most recent image is typically the most influential, prior imaging still provides additional prognostic value.

## Usage

To reproduce the main results in our paper:
```
# Create + activate environment
conda env create -f ltsa.yml
conda activate ltsa

# Navigate to `src/` directory
cd src

# Train LTSA on AREDS
python train.py --results_dir results --dataset AREDS --model LTSA --dropout 0.25 --augment --reduce_lr --batch_size 32 --tpe --step_ahead

# Train LTSA on OHTS
python train.py --results_dir results --dataset OHTS --model LTSA --dropout 0.25 --augment --reduce_lr --batch_size 32 --tpe --step_ahead

# Train baseline on AREDS
python train.py --results_dir results --dataset AREDS --model image --dropout 0.25 --augment --reduce_lr --batch_size 448

# Train baseline on OHTS
python train.py --results_dir results --dataset OHTS --model image --dropout 0.25 --augment --reduce_lr --batch_size 448
```

## Citation

If you find our work helpful, please cite the following.

MLA:
```
Holste, Gregory, et al. "Harnessing the power of longitudinal medical imaging for eye disease prognosis using Transformer-based sequence modeling." npj Digital Medicine 7.1 (2024): 216.
```

Bibtex:
```
@article{holste2024harnessing,
  title={Harnessing the power of longitudinal medical imaging for eye disease prognosis using Transformer-based sequence modeling},
  author={Holste, Gregory and Lin, Mingquan and Zhou, Ruiwen and Wang, Fei and Liu, Lei and Yan, Qi and Van Tassel, Sarah H and Kovacs, Kyle and Chew, Emily Y and Lu, Zhiyong and others},
  journal={npj Digital Medicine},
  volume={7},
  number={1},
  pages={216},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## Acknowledgment

This work is supported by the National Eye Institute under Award No. R21EY035296 and the National Science Foundation under Award Nos. 2145640 and 2306556. It is also supported by the NIH Intramural Research Program, the National Library of Medicine, and the National Eye Institute.


