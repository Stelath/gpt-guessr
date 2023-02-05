import torch
import torch.nn as nn

from transformers import AutoImageProcessor, ViTModel, ViTPreTrainedModel
from transformers.configuration_utils import PretrainedConfig

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
        image_size=224,
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

class GPTGuessr(ViTPreTrainedModel):
    def __init__(self, config: GPTGuessrConfig) -> None:
        super().__init__(config)

        self.num_countries = config.num_countries
        self.vit = ViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.country_classifier = nn.Linear(config.hidden_size, config.num_countries) if config.num_labels > 0 else nn.Identity()
        self.coordinates_classifier = nn.Linear(config.hidden_size + config.num_countries, 2) if config.num_labels > 0 else nn.Identity()

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
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        sequence_output = outputs[0]

        country = self.country_classifier(sequence_output[:, 0, :])
        coordinates = self.coordinates_classifier(torch.cat(country, sequence_output[:, 0, :]))

        return country, coordinates