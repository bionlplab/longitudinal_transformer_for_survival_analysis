import math

import timm
import torch
import torchvision

def create_model(args):
    if args.model == 'LTSA':
        return LTSA(args)
    else:
        return ImageSurvivalModel(args)

class ImageEncoder(torch.nn.Module):
    def __init__(self, arch):
        super(ImageEncoder, self).__init__()

        if arch == 'resnet18':
            self.model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            self.n_features = self.model.fc.in_features
            self.model.fc = torch.nn.Identity()
        elif arch == 'swin_v2_t':
            self.model = torchvision.models.swin_v2_t(weights='IMAGENET1K_V1')
            self.n_features = self.model.head.in_features
            self.model.head = torch.nn.Identity()
        elif arch == 'convnext_t':
            self.model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
            self.n_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = torch.nn.Identity()
        elif arch == 'caformer_s36':
            self.model = timm.create_model('caformer_s36', pretrained_cfg='sail_in22k_ft_in1k', pretrained=True, num_classes=0)
            self.n_features = self.model.fc.in_features
        elif arch == 'tf_efficientnet_b3':
            self.model = timm.create_model('tf_efficientnet_b3', pretrained_cfg='ns_jft_in1k', pretrained=True, num_classes=0)
            self.n_features = 1536

    def forward(self, x):
        return self.model(x)

class LearnedPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.pe = torch.nn.Sequential(
            torch.nn.Linear(1, int(math.sqrt(d_model))),
            torch.nn.Tanh(),
            torch.nn.Linear(int(math.sqrt(d_model)), d_model)
        )

    def forward(self, x, times):
        # x: batch x max_seq_len x 512
        # times: batch x max_seq_len
        times = times.unsqueeze(-1).float().to(x.device)

        # time_embeddings: batch x max_seq_len x 512
        time_embeddings = self.pe(times)

        return x + time_embeddings

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # Switch dimensions to BATCH FIRST (seq x batch x feat -> batch x seq x feat)
        self.pe = torch.permute(self.pe, (1, 0, 2))

    def forward(self, x, rel_times):
        if rel_times is None:
            x = x + self.pe[0, :x.size(1), :]
        else:
            for i, t in enumerate(rel_times):
                x[i, :, :] += self.pe[0, t, :]

        return self.dropout(x)

class LTSA(torch.nn.Module):
    def __init__(self, args):
        super(LTSA, self).__init__()
        self.args = args

        self.encoder = ImageEncoder(arch=args.arch)

        if args.attn_map:
            # Use custom TransformerEncoder + TransformerEncoderLayer to allow attention map extraction
            from transformers import TransformerEncoderLayer, TransformerEncoder

            transformer_encoder = TransformerEncoderLayer(d_model=self.encoder.n_features, nhead=args.n_heads, dim_feedforward=self.encoder.n_features, dropout=args.dropout, activation='relu', batch_first=True)
            self.transformer = TransformerEncoder(transformer_encoder, num_layers=args.n_layers)
        else:
            transformer_encoder = torch.nn.TransformerEncoderLayer(d_model=self.encoder.n_features, nhead=args.n_heads, dim_feedforward=self.encoder.n_features, dropout=args.dropout, activation='relu', batch_first=True)
            self.transformer = torch.nn.TransformerEncoder(transformer_encoder, num_layers=args.n_layers)

        if args.learned_pe:
            self.pos_encoder = LearnedPositionalEncoding(d_model=self.encoder.n_features)
        else:
            if args.tpe: 
                self.pos_encoder = PositionalEncoding(d_model=self.encoder.n_features, dropout=0, max_len=args.max_seq_len*12)  # 12 since measured in months
            else:
                self.pos_encoder = PositionalEncoding(d_model=self.encoder.n_features, dropout=0, max_len=args.max_seq_len)

        if args.amd_sev_enc:
            self.amd_sev_encoder = PositionalEncoding(d_model=self.encoder.n_features, dropout=0, max_len=args.max_seq_len)
        else:
            self.amd_sev_encoder = None

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=args.dropout),
            torch.nn.Linear(self.encoder.n_features, args.n_classes),
            torch.nn.Sigmoid()
        )

        if args.step_ahead:
            self.rel_pos_encoder = PositionalEncoding(d_model=self.encoder.n_features, dropout=0, max_len=args.max_seq_len*12)

            self.step_ahead_predictor = torch.nn.Sequential(
                torch.nn.Dropout(p=args.dropout),
                torch.nn.Linear(self.encoder.n_features, self.encoder.n_features)
            )

        self.causal_mask = torch.triu(torch.full((args.max_seq_len, args.max_seq_len), float('-inf'), device='cuda:0'), diagonal=1)

    def forward(self, x, seq_lengths, rel_times, prior_AMD_sevs):
        # embeddings: batch_size*max_seq_len x n_features
        embeddings = self.encoder(x)

        # Reshape embeddings to batch_size x seq_length x n_features
        embeddings = embeddings.reshape(len(seq_lengths), self.args.max_seq_len, self.encoder.n_features)

        if self.args.tpe:
            x = self.pos_encoder(embeddings, rel_times)
        else:
            x = self.pos_encoder(embeddings, None)

        if self.args.amd_sev_enc:
            x = self.amd_sev_encoder(x, prior_AMD_sevs)

        # Create mask to ignore padding tokens. For each sequence of visits, mask all tokens beyond last visit
        # Here, 1 = pad (ignore), 0 = valid (keep)
        src_key_padding_mask = torch.ones((x.shape[0], x.shape[1])).float().to('cuda:0')  # batch x seq_length
        for i, seq_length in enumerate(seq_lengths):
            src_key_padding_mask[i, :seq_length] = 0

        # Transformer modeling with "decoder-style" causal attention (only attend to current + PRIOR elements of each sequence)
        feats, attn_map = self.transformer(x, mask=self.causal_mask, src_key_padding_mask=src_key_padding_mask, is_causal=True, need_weights=args.attn_map)

        # Using src_key_padding_mask undoes padding... so re-pad each sequence in the batch with zeroes
        if feats.shape[1] < self.args.max_seq_len:  # b x seq x feat
            feats = torch.nn.functional.pad(feats, (0, 0, 0, self.args.max_seq_len - feats.shape[1], 0, 0), mode='constant')

        # Predict discrete-time hazard distribution
        hazards = self.classifier(feats)
        
        # Generative discrete-time survival probabilities
        surv = torch.cumprod(1-hazards.view(-1, hazards.shape[-1]), dim=1).view(hazards.shape[0], hazards.shape[1], hazards.shape[2])

        # Padding mask used to compute loss later
        padding_mask = torch.bitwise_not(src_key_padding_mask.bool()).unsqueeze(-1)

        if self.args.step_ahead:
            # Get time elapsed (delta) between consecutive visits
            delta_times = torch.diff(rel_times)  # batch x max_seq_len-1
            delta_times[delta_times < 0] = 0
            delta_times = torch.nn.functional.pad(delta_times, (0, 1), 'constant', 0)  # batch x max_seq_len

            # Use relative temporal timestep encoding to inform the model of "# months of into the future for which to predict imaging features"
            delta_encoded_feats = self.rel_pos_encoder(feats, delta_times) 

            # Predict imaging features of *next* visit for each subsequence
            feat_preds = self.step_ahead_predictor(delta_encoded_feats)

            # Get actual imaging features of next visit
            feat_targets = torch.nn.functional.pad(feats[:, 1:, :], (0, 0, 0, 1), 'constant', 0)

            if self.args.attn_map:
                return hazards, surv, feat_preds, feat_targets, padding_mask, attn_map
            else:
                return hazards, surv, feat_preds, feat_targets, padding_mask

        if self.args.attn_map:
            return hazards, surv, padding_mask, attn_map
        else:
            return hazards, surv, padding_mask

class ImageSurvivalModel(torch.nn.Module):
    def __init__(self, args):
        super(ImageSurvivalModel, self).__init__()

        self.encoder = ImageEncoder(arch=args.arch)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=args.dropout),
            torch.nn.Linear(self.encoder.n_features, args.n_classes),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        
        hazards = self.classifier(x)
        surv = torch.cumprod(1-hazards, dim=1)
        
        return hazards, surv