import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoImageProcessor, ViTModel, ViTPreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from torchvision.models import convnext_base
from torchvision.ops import Permute

from typing import Optional

class GPTGuessrConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=512,
        patch_size=16,
        num_channels=3,
        num_countries=177,
        qkv_bias=True,
        encoder_stride=16,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_countries = num_countries
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride

class GPTGuessrViT(ViTPreTrainedModel):
    def __init__(self, config: GPTGuessrConfig) -> None:
        super().__init__(config)

        self.num_countries = config.num_countries
        self.vit = ViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.country_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.Linear(512, 256),
            nn.Linear(256, config.num_countries),
        )
        self.coordinates_classifier = nn.Sequential(
            nn.Linear(config.hidden_size + config.num_countries, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 2),
        )

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
    ) -> tuple:
        
        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        sequence_output = outputs[0]

        country = self.country_classifier(sequence_output[:, 0, :])
        coordinates = self.coordinates_classifier(torch.cat((country, sequence_output[:, 0, :]), dim=1))

        return country, coordinates

class Flatten(torch.nn.Module):
    def __init__(self, start_dim=0, end_dim=- 1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return torch.flatten(x, self.start_dim, self.end_dim)

class Condenser(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.condenser = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm((layer_size,), eps=1e-06),
            Permute([0, 3, 1, 2]),
            Flatten(start_dim=1, end_dim=-1),
        )
    
    def forward(self, x):
        return self.condenser(x)

class GPTGuessrConvNeXt(nn.Module):
    def __init__(self, num_countries=177, num_channels=9) -> None:
        super().__init__()
        
        self.num_countries = num_countries
        self.convnext = convnext_base(num_classes=1024)
        self.convnext.features[0][0] = nn.Conv2d(num_channels, 128, kernel_size=(4, 4), stride=(4, 4))
        self.convnext.avgpool = nn.Identity()
        self.convnext.classifier = nn.Identity()
        
        self.country_classifier = nn.Sequential(
            Permute([0, 2, 3, 1]),
            nn.Linear(1024, 512),
            nn.LayerNorm((512,), eps=1e-06),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm((256,), eps=1e-06),
            nn.GELU(),
            Permute([0, 3, 1, 2]),
            Condenser(256),
            nn.Linear(256, self.num_countries),
            nn.Softmax(dim=1),
        )
        self.coordinates_classifier = nn.Sequential(
            Permute([0, 2, 3, 1]),
            nn.Linear(1024 + self.num_countries, 512),
            nn.LayerNorm((512,), eps=1e-06),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm((256,), eps=1e-06),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm((128,), eps=1e-06),
            nn.GELU(),
            Permute([0, 3, 1, 2]),
            Condenser(128),
            nn.Linear(128, 2),
        )
    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> tuple:
        
        outputs = self.convnext(pixel_values)
        
        country = self.country_classifier(outputs)
        coordinates = self.coordinates_classifier(torch.cat((country.view(country.shape[0], country.shape[1], 1, 1).repeat(1, 1, outputs.shape[2], outputs.shape[3]), outputs), dim=1))

        return country, coordinates
