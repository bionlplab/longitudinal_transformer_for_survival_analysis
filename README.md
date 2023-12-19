# longitudinal_transformer_for_survival_analysis

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
python train.py --results_dir results --dataset AREDS --model LTSA --dropout 0.25 --augment --reduce_lr --batch_size 448

# Train baseline on OHTS
python train.py --results_dir results --dataset OHTS --model LTSA --dropout 0.25 --augment --reduce_lr --batch_size 448
```