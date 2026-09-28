"""U-Net architecture for binary image-to-image refinement."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3×3 convolutions each followed by BatchNorm and ReLU."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant spatial regions in skip connections."""

    def __init__(self, gate_c, skip_c, inter_c):
        super().__init__()
        self.W_gate = nn.Conv2d(gate_c, inter_c, 1, bias=False)
        self.W_skip = nn.Conv2d(skip_c, inter_c, 1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_c, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attention = self.psi(g + s)
        return skip * attention


class UNet(nn.Module):
    """Attention U-Net for DXF image refinement.

    Encoder–decoder with skip connections and attention gates at every
    decoder level for the best spatial accuracy.

    Args:
        in_channels:  Number of input channels (1 for grayscale).
        out_channels: Number of output channels (1 for binary mask).
        features:     Channel sizes for each encoder level.
    """

    def __init__(self, in_channels=1, out_channels=1,
                 features=(64, 128, 256, 512)):
        super().__init__()
        features = list(features)

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoders.append(DoubleConv(prev, f))
            self.pools.append(nn.MaxPool2d(2))
            prev = f

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # ---- Decoder ----
        self.upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(prev, f, kernel_size=2, stride=2))
            self.attention_gates.append(
                AttentionGate(gate_c=f, skip_c=f, inter_c=f // 2))
            self.decoders.append(DoubleConv(f * 2, f))
            prev = f

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encode
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decode
        skips = skips[::-1]
        for i, (up, ag, dec) in enumerate(
                zip(self.upconvs, self.attention_gates, self.decoders)):
            x = up(x)
            skip = skips[i]
            # Handle odd sizes
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            skip = ag(gate=x, skip=skip)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return torch.sigmoid(self.final(x))
