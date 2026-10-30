import torch
from torch import nn


class UNetWithPresenceHead(nn.Module):
    def __init__(self, unet: nn.Module, hidden_features: int = 128):
        """
        Wraps an nnU-Net encoder/decoder network (PlainConvUNet, ResidualEncoderUNet, ...) and adds an image-level
        head that predicts whether the input patch contains foreground. The head sees mean and max pooled bottleneck
        features. The segmentation path is unchanged.

        train mode: forward returns (segmentation output, presence logits (b, ))
        eval mode:  forward returns the segmentation output only, so that nnU-Net's sliding window inference works
                    as usual. The maximum presence probability over all forward passes since
                    reset_case_level_outputs() (= over all sliding windows and mirrorings of a case) is available
                    through get_case_level_outputs(), which nnUNetPredictor adds to the exported case properties.
        """
        super().__init__()
        self.unet = unet
        bottleneck_features = unet.encoder.output_channels[-1]
        self.presence_head = nn.Sequential(
            nn.Linear(2 * bottleneck_features, hidden_features),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_features, 1)
        )
        self._max_presence_probability = None

    @property
    def encoder(self):
        return self.unet.encoder

    @property
    def decoder(self):
        # nnUNetTrainer.set_deep_supervision_enabled sets network.decoder.deep_supervision
        return self.unet.decoder

    def forward(self, x):
        skips = self.unet.encoder(x)
        seg_output = self.unet.decoder(skips)
        bottleneck = skips[-1].flatten(2)
        presence_logits = self.presence_head(torch.cat((bottleneck.mean(2), bottleneck.amax(2)), 1))[:, 0]
        if self.training:
            return seg_output, presence_logits

        probability = torch.sigmoid(presence_logits.float()).max().detach()
        self._max_presence_probability = probability if self._max_presence_probability is None else \
            torch.maximum(self._max_presence_probability, probability)
        return seg_output

    def reset_case_level_outputs(self):
        self._max_presence_probability = None

    def get_case_level_outputs(self) -> dict:
        if self._max_presence_probability is None:
            return {}
        return {'presence_probability': self._max_presence_probability.item()}
