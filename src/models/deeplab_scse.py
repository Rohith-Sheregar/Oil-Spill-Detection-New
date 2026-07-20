"""
DeepLabV3+ (MobileNetV2 backbone) with a bolt-on scSE attention module
applied to the decoder output, per Zhang et al. 2024. segmentation_models_
pytorch ships DeepLabV3+/MobileNetV2 natively but has no scSE option --
this wraps it.

IMPORTANT correctness note: scSE's channel count depends on smp's internal
decoder output, which varies slightly by smp version, so it has to be sized
dynamically. The bug to avoid: if you size/build it lazily on the first
*training* forward pass, and your optimizer was already constructed from
`model.parameters()` before that forward pass (the normal order of
operations), scSE's weights are silently never registered with the
optimizer and never receive gradients -- it trains, looks fine, and is
quietly doing nothing. This module forces a dummy forward pass inside
__init__ instead, so scSE exists before you ever call .parameters().
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SCSEModule(nn.Module):
    """Spatial + Channel Squeeze-Excitation (Roy et al., 2018)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )
        self.sse = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)


class DeepLabV3PlusSCSE(nn.Module):
    def __init__(self, in_channels=4, classes=1, input_size=256):
        """in_channels=4 for the v0 build (VV, VH, H, alpha); switch to 5
        once the wind-corrected ratio band is ready (week 3-4)."""
        super().__init__()
        self.base = smp.DeepLabV3Plus(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",  # smp adapts the first conv for in_channels != 3
            in_channels=in_channels,
            classes=classes,
        )
        # Verify the encoder name is still valid for your installed smp
        # version: smp.encoders.get_encoder_names()
        #
        # Two things verified against a real install (smp 0.5.0) that are
        # easy to get wrong by analogy with older smp versions/tutorials:
        #   1. decoder.forward() takes the encoder's feature list as ONE
        #      argument (`decoder(features)`), not unpacked (`decoder(*features)`).
        #   2. the dummy probe batch must be >1 -- ASPP's global-pooling
        #      branch collapses spatial dims to 1x1, and BatchNorm raises
        #      ("Expected more than 1 value per channel when training") on
        #      a batch of 1 in training mode. torch.no_grad() does NOT put
        #      the module in eval mode, so this bites even though no
        #      gradients are computed.
        with torch.no_grad():
            dummy = torch.zeros(2, in_channels, input_size, input_size)
            dec = self.base.decoder(self.base.encoder(dummy))
        self.scse = SCSEModule(dec.shape[1])

    def forward(self, x):
        feats = self.base.encoder(x)
        dec = self.base.decoder(feats)
        return self.base.segmentation_head(self.scse(dec))


if __name__ == "__main__":
    model = DeepLabV3PlusSCSE(in_channels=4, classes=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} parameters")  # sanity check: should be well under
                                         # the ~6M Zhang et al. report for
                                         # the full MobileNetV2+scSE model
    out = model(torch.zeros(2, 4, 256, 256))
    print(out.shape)  # expect (2, 1, 256, 256)
