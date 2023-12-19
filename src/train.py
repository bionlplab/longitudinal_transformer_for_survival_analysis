import os
import shutil

import argparse
import numpy as np
import pandas as pd
import torch

from datasets import AREDS_Survival_Dataset, OHTS_Survival_Dataset, AREDS_Longitudinal_Survival_Dataset, OHTS_Longitudinal_Survival_Dataset
from utils import set_seed, worker_init_fn, val_worker_init_fn, train, validate, evaluate, train_LTSA, validate_LTSA, evaluate_LTSA
from losses import CrossEntropySurvLoss, NLLSurvLoss, CoxSurvLoss
from models import create_model

def AREDS_collate_fn(data):
    x = torch.stack([d['x'] for d in data], dim=0).reshape(-1, 3, 224, 224)  # batch*seq_length x 3 x 224 x 224
    y = torch.stack([d['y'] for d in data], dim=0)  # batch x 1
    censorship = torch.stack([d['censorship'] for d in data], dim=0)  # batch x 1

    rel_time = torch.stack([d['rel_time'] for d in data], dim=0)  # batch x seq_length
    prior_AMD_sev = torch.stack([d['prior_AMD_sev'] for d in data], dim=0)  # batch x seq_length
    obs_time = torch.stack([d['obs_time'] for d in data], dim=0)  # batch x seq_length

    fname = np.concatenate([d['fname'] for d in data])  # sum_batch_seq_len
    patient_id = np.concatenate([d['patient_id'] for d in data])  # sum_batch_seq_len
    laterality = np.concatenate([d['laterality'] for d in data])  # sum_batch_seq_len
    event_time = np.concatenate([d['event_time'] for d in data])  # sum_batch_seq_len

    seq_length = [d['seq_length'] for d in data]  # batch

    return {'x': x, 'y': y, 'censorship': censorship, 'event_time': event_time, 'obs_time': obs_time, 'fname': fname, 'seq_length': seq_length, 'rel_time': rel_time, 'prior_AMD_sev': prior_AMD_sev, 'patient_id': patient_id, 'laterality': laterality}

def OHTS_collate_fn(data):
    x = torch.stack([d['x'] for d in data], dim=0).reshape(-1, 3, 224, 224)  # batch*seq_length x 3 x 224 x 224
    y = torch.stack([d['y'] for d in data], dim=0)  # batch x 1
    censorship = torch.stack([d['censorship'] for d in data], dim=0)  # batch x 1

    rel_time = torch.stack([d['rel_time'] for d in data], dim=0)  # batch x seq_length
    prior_AMD_sev = None
    obs_time = torch.stack([d['obs_time'] for d in data], dim=0)  # batch x seq_length

    fname = np.concatenate([d['fname'] for d in data])  # sum_batch_seq_len
    patient_id = np.concatenate([d['patient_id'] for d in data])  # sum_batch_seq_len
    laterality = np.concatenate([d['laterality'] for d in data])  # sum_batch_seq_len
    event_time = np.concatenate([d['event_time'] for d in data])  # sum_batch_seq_len

    seq_length = [d['seq_length'] for d in data]  # batch

    return {'x': x, 'y': y, 'censorship': censorship, 'event_time': event_time, 'obs_time': obs_time, 'fname': fname, 'seq_length': seq_length, 'rel_time': rel_time, 'prior_AMD_sev': prior_AMD_sev, 'patient_id': patient_id, 'laterality': laterality}

def main(args):
    # Set detailed model name
    MODEL_NAME = 'surv'
    MODEL_NAME += f'_{args.dataset}'
    MODEL_NAME += f'_{args.arch}'
    MODEL_NAME += f'_{args.model}'
    MODEL_NAME += f'_{args.loss}'
    MODEL_NAME += f'_lr-reduce-x{args.reduce_lr_factor}-p{args.reduce_lr_patience}' if args.reduce_lr else ''
    MODEL_NAME += f'_val-{args.val}'
    MODEL_NAME += f'_alpha-{args.alpha}' if args.loss != 'cox' else ''
    MODEL_NAME += f'_step-ahead' if args.step_ahead else ''
    MODEL_NAME += 'learned-pe' if args.learned_pe else ''
    MODEL_NAME += f'_tpe-{args.tpe_mode}' if args.tpe else ''
    MODEL_NAME += '_AMD-sev-enc' if args.amd_sev_enc else ''
    MODEL_NAME += f'_{args.n_layers}-layers' if args.n_layers > 1 else ''
    MODEL_NAME += f'_{args.n_heads}-heads' if args.model == 'transformer' else ''
    MODEL_NAME += f'_bs-{args.batch_size}'
    MODEL_NAME += '_amp' if args.amp else ''
    MODEL_NAME += f'_lr-{args.lr}'
    MODEL_NAME += f'_{args.max_epochs}-ep'
    MODEL_NAME += f'_patience-{args.patience}'
    MODEL_NAME += '_aug' if args.augment else ''
    MODEL_NAME += f'_drp-{args.dropout}' if args.dropout > 0 else ''
    MODEL_NAME += f'_seed-{args.seed}' if args.seed != 0 else ''

    # Create directory for this model (or delete contents if already exists)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    model_dir = os.path.join(args.results_dir, MODEL_NAME)

    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # Create csv documenting training history
    column_list = ['epoch', 'phase', 'loss'] + \
        [f'c_index_t-{t}_del-t-{del_t}' for t in args.t_list for del_t in args.del_t_list] + \
        [f'brier_t-{t}_del-t-{del_t}' for t in args.t_list for del_t in args.del_t_list] + \
        ['mean_c_index', 'mean_brier']
    history = pd.DataFrame(columns=column_list)
    history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)

    # Fix random seed
    set_seed(args.seed)

    device = torch.device('cuda:0')

    if args.dataset == 'AREDS':
        args.data_dir = '/prj0129/grh4006/AMD_224'
        args.label_dir = '/prj0129/grh4006/AREDS/labels'
        args.n_classes = 27  # 6-month intervals for 13 years

        if args.model == 'LTSA':
            dataset = AREDS_Longitudinal_Survival_Dataset
            collate_fn = AREDS_collate_fn
        else:
            dataset = AREDS_Survival_Dataset
            collate_fn = None
    elif args.dataset == 'OHTS':
        args.data_dir = '/prj0129/grh4006/image_crop2'
        args.label_dir = '/prj0129/grh4006/AREDS/glaucoma_labels'
        args.n_classes = 15  # 1-year intervals for 14 years

        if args.model == 'LTSA':
            dataset = OHTS_Longitudinal_Survival_Dataset
            collate_fn = OHTS_collate_fn
        else:
            dataset = OHTS_Survival_Dataset
            collate_fn = None

    # Create train, val, and test datasets
    train_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='train', augment=args.augment, tpe_mode=args.tpe_mode, learned_pe=args.learned_pe)
    val_dataset   = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='val', augment=False, tpe_mode=args.tpe_mode, learned_pe=args.learned_pe)
    test_dataset  = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='test', augment=False, tpe_mode=args.tpe_mode, learned_pe=args.learned_pe)

    # Create train, val, and test data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True, worker_init_fn=val_worker_init_fn, collate_fn=collate_fn)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True, worker_init_fn=val_worker_init_fn, collate_fn=collate_fn)

    # Create model
    model = create_model(args).to(device)
    print(model)

    # Get loss function
    if args.loss == 'ce':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha)
    elif args.loss == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.alpha)
    elif args.loss == 'cox':
        loss_fn = CoxSurvLoss()

    # Get optimizer and learning rate scheduler (if enabled)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = None
    if args.reduce_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if args.val == 'c-index' else 'min', factor=args.reduce_lr_factor, patience=args.reduce_lr_patience, verbose=True)

    # Train with early stopping
    epoch = 1
    if args.val == 'c-index':
        early_stopping_dict = {'best_metric': 0., 'epochs_no_improve': 0, 'metric': args.val}
    elif args.val == 'loss':
        early_stopping_dict = {'best_metric': 1e8, 'epochs_no_improve': 0, 'metric': args.val}
    elif args.val == 'brier':
        early_stopping_dict = {'best_metric': 1e8, 'epochs_no_improve': 0, 'metric': args.val}
    best_model_wts = None
    while epoch <= args.max_epochs and early_stopping_dict['epochs_no_improve'] < args.patience:
        history = train(model=model, device=device, loss_fn=loss_fn, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, amp=args.amp, t_list=args.t_list, del_t_list=args.del_t_list, step_ahead=args.step_ahead, dataset=args.dataset)
        history, early_stopping_dict, best_model_wts = validate(model=model, device=device, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, data_loader=val_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts=best_model_wts, amp=args.amp, t_list=args.t_list, del_t_list=args.del_t_list, step_ahead=args.step_ahead, dataset=args.dataset)

        epoch += 1

    # Evaluate trained model on test set
    evaluate(model=model, device=device, loss_fn=loss_fn, data_loader=test_loader, history=history, model_dir=model_dir, weights=best_model_wts, amp=args.amp, t_list=args.t_list, del_t_list=args.del_t_list, step_ahead=args.step_ahead, dataset=args.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/prj0129/grh4006/AMD_224')
    parser.add_argument('--label_dir', type=str, default='/prj0129/grh4006/AREDS/labels')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32, help='Note that effective batch size is max_seq_len*batch_size when model=LTSA')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'swin_v2_t', 'convnext_t', 'caformer_s36'], help='Image encoder architecture')
    parser.add_argument('--model', type=str, default='image', choices=['image', 'LTSA'], help='Single-image baseline vs. Longitudinal Transformer for Survival Analysis (LTSA)')
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--tpe', action='store_true', default=False, help='Use temporal positional encoding (TPE) to embed knowledge of visit time in longitudinal image sequences')
    parser.add_argument('--tpe_mode', type=str, default='months', choices=['bins', 'months'], help='Embed visit time measured in months or discrete 6-month time bins')
    parser.add_argument('--amd_sev_enc', action='store_true', default=False, help='Embed AMD severity score from prior visit')
    parser.add_argument('--learned_pe', action='store_true', default=False, help='Use learned positional encoding rather than fixed sinusoidal encoding')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'nll', 'cox'], help='Survival loss function (cross-entropy, negative log likelihood, or Cox)')
    parser.add_argument('--alpha', type=float, default=0.15, help='Weight applied to term that upweights uncensored cases')
    parser.add_argument('--val', type=str, default='c-index', choices=['loss', 'c-index', 'brier'], help='Validation metric')
    parser.add_argument('--max_seq_len', type=int, default=14, help='Maximum sequence length (all sequences are padded to this length)')
    parser.add_argument('--t_list', type=int, nargs='+', default=[1, 2, 3, 5, 8])
    parser.add_argument('--del_t_list', type=int, nargs='+', default=[1, 2, 5, 8])
    parser.add_argument('--step_ahead', action='store_true', default=False, help='Whether to enable step-ahead feature prediction')
    parser.add_argument('--dataset', type=str, default='AREDS', choices=['AREDS', 'OHTS'])
    parser.add_argument('--reduce_lr', action='store_true', default=False, help='Whether to apply a "reduce on plataeu" learning rate scheduler')
    parser.add_argument('--reduce_lr_factor', type=float, default=0.5, help='Factor by which to reduce learning rate on plateau')
    parser.add_argument('--reduce_lr_patience', type=int, default=2, help='Patience for learning rate scheduler')

    args = parser.parse_args()

    print(args)
    main(args)