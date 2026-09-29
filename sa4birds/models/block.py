import torch
from torch import nn
from torch.nn import functional as F



class MultiHeadSABlock(nn.Module):
    """
    Multi-Head Spectrogram Attention Block.

    This module applies class-wise multi-head attention over
    2D feature maps (e.g., time-frequency representations).
    It learns separate attention maps and classification
    features per head and aggregates them into final
    class-level predictions.

    Parameters
    ----------
    in_features : int
       Number of input feature channels.
    num_classes : int
       Number of output classes.
    heads : int, optional
       Number of attention heads. Default: 1.
    activation : str, optional
       Activation function applied to classification features.
       Supported: {"linear", "sigmoid", "tanh"}.
       Default: "linear".
    temperature : float, optional
       Temperature scaling factor applied to attention logits
       before softmax. Higher values produce softer attention.
       Default: 2.0.
    att_dropout : float, optional
       Dropout rate on the attention logits during training. Default: 0.1.

    Input
    -----
    x : torch.Tensor
       Shape (B, C, F, T)
       B = batch size
       C = input channels
       F = frequency dimension
       T = time dimension

    Returns
    -------
    tuple:
       final_features : torch.Tensor
           Shape (B, num_classes)
           Aggregated class predictions.
       norm_att : torch.Tensor
           Normalized attention maps.
           Shape (B, num_classes, heads, F, T)
       cla_feat : torch.Tensor
           Activated class-wise features before aggregation.
           Shape (B, num_classes, heads, F, T)
   """
    def __init__(self, in_features, num_classes, heads=1, activation="linear", temperature=2.0,
                 att_dropout=0.1):
        super().__init__()
        self.heads = heads
        self.activation = activation
        self.temperature = torch.tensor(float(temperature))
        # Dropout on the attention logits, before the softmax (training only).
        self.att_dropout = float(att_dropout)

        # Attention map generator (1x1 conv)
        self.att = nn.Conv2d(in_features, num_classes * heads, kernel_size=(1, 1), bias=True)

        # Class feature generator (1x1 conv)
        self.cla = nn.Conv2d(in_features, num_classes * heads, kernel_size=1, bias=True)

        # Learnable head importance weights
        self.head_weights = nn.Parameter(torch.ones(self.heads))

    def nonlinearity(self, x):
        """
            Apply selected activation function.
        """
        activations = {
            "linear": lambda t: t,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
        }
        if self.activation not in activations:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return activations[self.activation](x)

    def forward(self, x, crop):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, F, T).
        crop : int
            Number of frames to crop from both temporal sides.

        Returns
        -------
        tuple:
            (final_features, norm_att, cla_feat)
        """

        # Optional temporal cropping
        if crop > 0:
            x = x[:, :, :, crop: -crop]

        B, C, Fq, T = x.size()

        # --------------------------------------------------
        # Attention branch
        # --------------------------------------------------
        att_map = self.att(x) / self.temperature
        att_map = att_map.view(B, -1, self.heads, Fq, T)

        # Dropout on the logits, not a mask: a dropped logit becomes 0 rather than
        # -inf, and the survivors are scaled by 1/(1-p), which sharpens the softmax.
        if self.att_dropout:
            att_map = F.dropout(att_map, p=self.att_dropout, training=self.training)

        # Softmax over spatial dimensions (F*T)
        norm_att = torch.softmax(att_map.view(B, -1, self.heads, Fq * T), dim=-1)
        norm_att = norm_att.view(B, -1, self.heads, Fq, T)

        # --------------------------------------------------
        # Classification branch
        # --------------------------------------------------
        cla_feat = self.nonlinearity(self.cla(x))
        cla_feat = cla_feat.view(B, -1, self.heads, Fq, T)

        # Apply attention weighting
        weighted = norm_att * cla_feat

        # Aggregate spatial dimensions
        weighted = weighted.sum(dim=-1).sum(dim=-1)

        # Weighted aggregation across heads
        final_features = (weighted * self.head_weights.view(1, 1, -1)).sum(dim=-1) / self.head_weights.sum()

        return final_features, norm_att, cla_feat
