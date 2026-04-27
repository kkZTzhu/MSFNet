# MSFNet

Official PyTorch implementation of **MSFNet**, a hybrid CNN-Transformer architecture for multi-sequence MRI classification. The model employs 3D ResNet encoders to extract features from four MRI sequences (T1W, T2, T2-SPAIR, DWI), a Transformer-based fusion module with cross-modal attention to integrate multi-modal information, and a Kolmogorov-Arnold Network (KAN) classifier head.

## Architecture Overview

MSFNet consists of three core components:

1. **Multi-Sequence Encoder**: Four parallel 3D ResNet backbones (ResNet-34/50/101) extract modality-specific features from each MRI sequence independently.
2. **Transformer Fusion Module**: A modality tokenizer projects features into a shared space, followed by a Transformer encoder with positional encoding and cross-modal multi-head attention to model inter-sequence dependencies. The module supports multiple fusion strategies: transformer-only, traditional attention, concatenation, and hybrid.
3. **KAN Classifier**: A Kolmogorov-Arnold Network replaces the standard MLP head, using learnable B-spline basis functions for improved non-linear classification.

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- MONAI >= 1.0
- See `requirements.txt` for full dependencies.

### Training



## License

This project is released under the MIT License.

