import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

from .resnet3d import resnet3d_mt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network layer."""

    def __init__(self, in_features, out_features, grid_size=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        self.grid = nn.Parameter(torch.linspace(-1, 1, grid_size))
        self.coef = nn.Parameter(torch.randn(in_features, out_features, grid_size))

        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bases = self._compute_bases(x)
        y = torch.einsum('bik,iok->bo', bases, self.coef)
        y = y * self.scale + self.shift
        return y

    def _compute_bases(self, x):
        x_expand = x.unsqueeze(-1)
        grid_expand = self.grid.unsqueeze(0).unsqueeze(0)
        dist = torch.abs(x_expand - grid_expand)
        bases = torch.maximum(1 - dist, torch.zeros_like(dist))
        return bases


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the transformer."""

    def __init__(self, d_model, max_len=10):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ModalityTokenizer(nn.Module):
    """Modality-specific tokenization with learnable modality embeddings."""

    def __init__(self, feature_dim, num_modalities=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities

        self.modality_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_modalities)
        ])

        self.modality_embeddings = nn.Parameter(torch.randn(num_modalities, feature_dim))

    def forward(self, features_list):
        tokenized_features = []

        for i, features in enumerate(features_list):
            projected = self.modality_projectors[i](features)
            modal_embedded = projected + self.modality_embeddings[i].unsqueeze(0)
            tokenized_features.append(modal_embedded)

        tokenized_sequence = torch.stack(tokenized_features, dim=0)
        return tokenized_sequence


class TransformerFusionModule(nn.Module):
    """Transformer-based multi-modal fusion with cross-modal attention."""

    def __init__(self, feature_dim, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.pos_encoding = PositionalEncoding(feature_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )

        self.fusion_strategy = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        logger.info(f"Initialized TransformerFusionModule with {num_layers} layers, {num_heads} heads")

    def forward(self, tokenized_sequence):
        pos_encoded = self.pos_encoding(tokenized_sequence)

        transformer_output = self.transformer_encoder(pos_encoded)

        cross_attended, attention_weights = self.cross_modal_attention(
            transformer_output, transformer_output, transformer_output
        )

        fused_sequence = transformer_output + cross_attended

        # Multi-strategy pooling
        mean_pooled = fused_sequence.mean(dim=0)
        max_pooled, _ = fused_sequence.max(dim=0)

        attention_scores = F.softmax(
            torch.sum(fused_sequence * fused_sequence.mean(dim=0, keepdim=True), dim=-1),
            dim=0
        )

        weighted_pooled = torch.sum(
            fused_sequence * attention_scores.unsqueeze(-1),
            dim=0
        )

        combined_features = mean_pooled + 0.5 * max_pooled + 0.5 * weighted_pooled

        fused_features = self.fusion_strategy(combined_features)

        return fused_features, attention_weights, {
            'mean_pooled': mean_pooled,
            'max_pooled': max_pooled,
            'weighted_pooled': weighted_pooled,
            'transformer_output': transformer_output,
            'cross_attended': cross_attended
        }


class MultiSequenceEncoder(nn.Module):
    """Multi-sequence 3D ResNet encoder with shared architecture across modalities."""

    def __init__(self, backbone='resnet34', pretrained_path=None):
        super().__init__()

        self.encoders = nn.ModuleList()

        for _ in range(4):
            encoder = resnet3d_mt(arch=backbone, num_classes=2, pretrained=None)
            if pretrained_path is not None:
                encoder = self._load_pretrained_weights(encoder, pretrained_path)
            self.encoders.append(encoder)

        self.feature_dim = 512 if backbone == 'resnet34' else 2048

    def _load_pretrained_weights(self, encoder, weight_path):
        try:
            pretrained_dict = torch.load(weight_path, map_location='cpu')
            encoder.load_state_dict(pretrained_dict, strict=False)
            logger.info("Pretrained weights loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
        return encoder

    def forward(self, sequences):
        features = []
        feature_maps = []

        for i, seq in enumerate(sequences):
            feat_map = self.encoders[i](seq)
            feature_maps.append(feat_map)

            feat = F.adaptive_avg_pool3d(feat_map, 1)
            feat = feat.view(feat.size(0), -1)
            features.append(feat)

        return features, feature_maps


class MSFNet(nn.Module):
    """MSFNet: Multi-Sequence Fusion Network for MRI-based classification.

    A hybrid CNN-Transformer architecture that extracts features from multiple
    MRI sequences using 3D ResNet encoders and fuses them via a Transformer
    with cross-modal attention and KAN-based classification head.
    """

    def __init__(self,
                 num_classes=2,
                 backbone='resnet34',
                 pretrained_path=None,
                 num_transformer_layers=3,
                 num_heads=8,
                 dropout=0.1,
                 fusion_strategy='hybrid'):
        super().__init__()

        self.encoder = MultiSequenceEncoder(backbone=backbone, pretrained_path=pretrained_path)
        feature_dim = self.encoder.feature_dim

        self.modality_tokenizer = ModalityTokenizer(feature_dim, num_modalities=4)

        self.transformer_fusion = TransformerFusionModule(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout
        )

        self.traditional_attention = nn.MultiheadAttention(feature_dim, num_heads=8)

        self.fusion_strategy = fusion_strategy

        if fusion_strategy == 'transformer_only':
            self.fusion_dim = feature_dim
        elif fusion_strategy == 'traditional_only':
            self.fusion_dim = feature_dim
        elif fusion_strategy == 'concat':
            self.fusion_dim = feature_dim * 4
        elif fusion_strategy == 'hybrid':
            self.fusion_dim = feature_dim * 2
            self.hybrid_fusion = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        classifier_input_dim = feature_dim if fusion_strategy == 'hybrid' else self.fusion_dim

        self.kan_classifier = nn.Sequential(
            KANLayer(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            KANLayer(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            KANLayer(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        logger.info(f"Initialized MSFNet with {backbone} backbone and {fusion_strategy} fusion")

    def forward(self, data_dict):
        sequences = [
            data_dict['T1W'],
            data_dict['T2'],
            data_dict['T2_SPAIR'],
            data_dict['DWI']
        ]

        features, feature_maps = self.encoder(sequences)

        tokenized_sequence = self.modality_tokenizer(features)

        if self.fusion_strategy == 'transformer_only':
            transformer_fused, attention_weights, fusion_details = self.transformer_fusion(tokenized_sequence)
            fused_features = transformer_fused

        elif self.fusion_strategy == 'traditional_only':
            stacked_features = torch.stack(features, dim=1).transpose(0, 1)
            traditional_fused, _ = self.traditional_attention(
                stacked_features, stacked_features, stacked_features
            )
            fused_features = traditional_fused.transpose(0, 1).mean(dim=1)
            attention_weights = None
            fusion_details = {}

        elif self.fusion_strategy == 'concat':
            fused_features = torch.cat(features, dim=1)
            attention_weights = None
            fusion_details = {}

        elif self.fusion_strategy == 'hybrid':
            transformer_fused, attention_weights, fusion_details = self.transformer_fusion(tokenized_sequence)

            stacked_features = torch.stack(features, dim=1).transpose(0, 1)
            traditional_fused, _ = self.traditional_attention(
                stacked_features, stacked_features, stacked_features
            )
            traditional_fused = traditional_fused.transpose(0, 1).mean(dim=1)

            combined = torch.cat([transformer_fused, traditional_fused], dim=1)
            fused_features = self.hybrid_fusion(combined)

        output = self.kan_classifier(fused_features)

        return output, {
            'fused_features': fused_features,
            'modal_features': features,
            'feature_maps': feature_maps,
            'attention_weights': attention_weights,
            'fusion_details': fusion_details,
            'tokenized_sequence': tokenized_sequence
        }
