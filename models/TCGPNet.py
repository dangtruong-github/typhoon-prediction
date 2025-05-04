import torch

import torch.nn as nn

from transformers.models.swin.modeling_swin import SwinModel, SwinConfig
from typing import List


class TCGPNet(nn.Module):
    """
        Important Note: Image_size > 2 ^ (len(depths) - 1) * window_size * patch_size
    """
    def __init__(
        self,
        img_size = (128, 128),
        predictor_configs: List = None,  # (in_channels, is_3d, resolution)
        patch_size: int = 4,
        embed_dim: int = 96,
        window_size: int = 7,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        head_dim: int = 1024,
        num_classes: int = 2
    ):
        super().__init__()
        self.img_size = img_size
        self.predictor_configs = predictor_configs
        self.num_classes = num_classes
        self.head_dim = head_dim
        
        # Compute channel offsets for splitting input tensor
        self.channel_offsets = []
        current_channel = 0
        for in_channels in self.predictor_configs:
            self.channel_offsets.append((current_channel, current_channel + in_channels))
            current_channel += in_channels
        
        # Swin Transformer Encoders
        self.encoders = nn.ModuleList([])
        for num_channels in self.predictor_configs:
            swin_config = SwinConfig(
                num_channels=num_channels,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                patch_size=patch_size,  # Handled by PatchEmbedding
                image_size=img_size,
                mlp_ratio=4.0,
                attention_dropout=0.1,
                drop_rate=0.1,
            )
            swin_model = SwinModel(swin_config)
            self.encoders.append(swin_model)
        
        # Feature Fusion Module
        self.final_dim = embed_dim * (2 ** (len(depths) - 1))  # After all patch mergings
        self.norm = nn.LayerNorm(self.final_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(self.final_dim * len(self.predictor_configs), head_dim)
        self.fc2 = nn.Linear(head_dim, num_classes)
    
    
    def check_input_shape(self, x: torch.Tensor) -> None:
        B, C_total, H, W = x.shape
        assert C_total == sum(config for config in self.predictor_configs), "Input channel mismatch"
        assert H == self.img_size[0] and W == self.img_size[1], "Input size mismatch"
    
    def feature_extraction(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Check input shape
        self.check_input_shape(x)

        # Extract predictor tensors
        predictor_tensors = []
        for offset in self.channel_offsets:
            predictor_tensor = x[:, offset[0]:offset[1], :, :]  # (B, T, in_channels, H, W)
            predictor_tensors.append(predictor_tensor)
        
        # Feature Extraction
        features = []
        for x, encoder in zip(predictor_tensors, self.encoders):
            x = encoder(x).last_hidden_state  # (B, num_patches', final_dim)
            features.append(x)
        
        return features
    
    def feature_fusion(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Feature Fusion
        features = [self.norm(f) for f in features]
        features = torch.cat([self.avg_pool(f.transpose(1, 2)).squeeze(-1) for f in features], dim=1)
        
        x = self.gelu(self.fc1(features))
        x = self.fc2(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature Extraction
        features = self.feature_extraction(x)
        # Feature Fusion
        x = self.feature_fusion(features)
        return x


if __name__ == "__main__":
    # Adjust predictor_configs to match the dummy input's channel dimension (228)
    # Let's divide 228 channels evenly across predictors
    num_predictors = 4  # Keeping the same number of predictors
    channels_per_predictor = 228 // num_predictors
    predictor_configs = [channels_per_predictor] * num_predictors
    # Adjust the last one in case 228 isn't divisible by 4
    predictor_configs[-1] = 228 - sum(predictor_configs[:-1])
    
    model = TCGPNet(
        img_size=(33, 33),
        predictor_configs=predictor_configs,
        patch_size=4,
        window_size=4,
        depths=[2, 2],
        num_heads=[3, 6],
    )

    # Keep the original dummy input shape
    dummy_input = torch.rand((32, 228, 33, 33))

    output = model(dummy_input)
    print(output.shape)  # Should be (32, 2)