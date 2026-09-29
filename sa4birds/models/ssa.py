import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sa4birds.models.block import MultiHeadSABlock


class SSA(nn.Module):
    """
    Single-branch Spectrogram Attention (SSA) Model.

    Applies class-wise spatial attention to the full-resolution
    time-frequency features of a CNN backbone, for weakly supervised
    multi-label classification. sa4birds/models/dsa.py is the dual-branch
    variant, which adds a pooled global branch and fuses the two.

    Architecture Overview
    ----------------------
    1. Backbone (timm model):
      Extracts time-frequency feature maps.

    2. Projection:
      LayerNorm, 1x1 projection and ReLU, added back as a residual.

    3. Multi-Head Spatial Attention:
      Produces class-wise attention-weighted predictions.

    Parameters
    ----------
    cfg : object
       Configuration object containing at least:
           - cfg.frontend.in_chans
           - cfg.network.model_name
           - cfg.network.droppath_rate
           - cfg.network.temperature
           - cfg.network.dropout_rate
           - cfg.train.num_classes

    Input
    -----
    x : torch.Tensor
       Shape (B, C, F, T)
       B = batch size
       C = input channels
       F = frequency bins
       T = time frames

    Returns
    -------
    torch.Tensor
       Weak (clip-level) class predictions of shape
       (B, num_classes).

    Notes
    -----
    - Designed for spectrogram-like inputs.
    - Uses timm backbone for feature extraction.
    - Supports optional center cropping for 5-second inference.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Input channels
        self.in_chans = self.cfg.frontend.in_chans

        # Expected pixel length for 5-second segment
        # (depends on FFT hop size etc.)
        self.px_per_5s = 501

        # Backbone downsampling factor
        # (depends on architecture)
        self.downsample_factor = 64

        # --------------------------------------------------
        # Backbone (timm model)
        # --------------------------------------------------
        timm_params = {"model_name": cfg.network.model_name,
                       "pretrained": True,
                       "in_chans": self.in_chans,
                       "drop_path_rate": self.cfg.network.droppath_rate}

        self.backbone = timm.create_model(**timm_params)
        backbone_out = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0, '')

        # --------------------------------------------------
        # Attention block
        # --------------------------------------------------
        self.attention = MultiHeadSABlock(backbone_out,
                                           self.cfg.train.num_classes,
                                            heads=1,
                                            activation='sigmoid',
                                            temperature=self.cfg.network.temperature)

        # --------------------------------------------------
        # projection layers
        # --------------------------------------------------
        self.l_proj = nn.Conv2d(backbone_out, backbone_out, kernel_size=(1,1), bias=True)

        self.dropout = cfg.network.dropout_rate

        # Feature normalization
        self.norm = nn.LayerNorm(normalized_shape=backbone_out)

    def forward(self, x, center_5s=False, return_attention=False):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, F, T).
        center_5s : bool, optional
            If True, only the center 5-second region of the
            feature map is used for prediction.
        return_attention : bool, optional
            If True, also return the per-class spatial attention maps even in eval
            mode (normally only available via the training-mode return path). Does
            not affect the predictions themselves -- dropout/batchnorm still follow
            ``self.training``, not this flag.

        Returns
        -------
        torch.Tensor or tuple
            Weak (clip-level) predictions of shape (B, num_classes). In
            training mode, or with ``return_attention=True``, a tuple of
            those predictions and the attention map.
        """
        def normalize_and_project(feat, proj_layer):
            """
                Apply LayerNorm and 1x1 projection with ReLU.
            """
            feat = self.norm(feat.transpose(1, -1)).transpose(1, -1)
            feat = F.relu(proj_layer(feat))
            return feat

        t = x.shape[-1]
        # Backbone feature extraction
        x = self.backbone(x)  # -> (b, feats, f, t)

        # --------------------------------------------------
        # Feature projection
        # --------------------------------------------------
        l_x = F.dropout(x, p=self.dropout, training=self.training)
        residual = l_x
        l_x = normalize_and_project(l_x, self.l_proj)
        l_x = residual + l_x
        l_x = F.dropout(l_x, p=self.dropout, training=self.training)


        # --------------------------------------------------
        # Optional center cropping
        # --------------------------------------------------
        if center_5s:
            crop = int(round((t - self.px_per_5s) / self.downsample_factor))
        else:
            crop = 0

        weak, l_att, _ = self.attention(l_x, crop)

        if self.training or return_attention:
            return weak, l_att
        return weak
