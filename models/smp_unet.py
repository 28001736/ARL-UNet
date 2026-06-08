import torch.nn as nn
import segmentation_models_pytorch as smp


class SMPUNet(nn.Module):
    """
    Thin wrapper around segmentation_models_pytorch UNet.

    Returns (None, out) so engine.py can detect the absence of deep-supervision
    auxiliary outputs and route to the simple loss path.
    """

    deep_supervision = False

    def __init__(self, encoder_name='resnet34', in_channels=3, classes=1, logger=None):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid',
        )
        if logger is not None:
            logger.info(f'SMPUNet | encoder={encoder_name} | in_channels={in_channels}')

    def forward(self, x):
        out = self.model(x)
        return None, out
