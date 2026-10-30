import torch
from torch import nn


class PresenceLoss(nn.Module):
    def __init__(self, topk: int = 64, pos_weight: float = 1.0, neg_weight: float = 1.0, ignore_label: int = None):
        """
        Image-level BCE computed from the segmentation output. For each sample the presence score is the mean
        foreground probability of its top-k most confident voxels. Samples without foreground are pushed towards 0
        (gradient only reaches the most confident false positive voxels), samples with foreground towards 1.

        For positive samples k is capped at the number of foreground voxels so that a lesion that is only partially
        inside the patch does not force the network to grow it.

        Soft Dice provides (almost) no gradient on samples without foreground and CE averages a small false positive
        region away over the whole patch. This term targets exactly those false positives.
        """
        super().__init__()
        self.topk = topk
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.ignore_label = ignore_label

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        net_output must be logits (b, c, x, y(, z)), target must be a label map (b, 1, x, y(, z))
        """
        p_fg = 1 - torch.softmax(net_output.float(), 1)[:, 0]
        target = target[:, 0]
        if self.ignore_label is not None:
            valid = target != self.ignore_label
            p_fg = p_fg * valid
            gt_fg = (target > 0) & valid
        else:
            gt_fg = target > 0

        p_fg = p_fg.flatten(1)
        num_fg = gt_fg.flatten(1).sum(1)
        is_positive = num_fg > 0

        k = min(self.topk, p_fg.shape[1])
        top_values = torch.topk(p_fg, k, dim=1, sorted=True).values
        k_per_sample = torch.where(is_positive, num_fg.clamp(max=k), torch.full_like(num_fg, k))
        keep = torch.arange(k, device=p_fg.device)[None] < k_per_sample[:, None]
        score = (top_values * keep).sum(1) / k_per_sample
        score = score.clamp(1e-6, 1 - 1e-6)

        # BCE written out because nn.functional.binary_cross_entropy refuses to run under autocast. score is float32
        # and clamped, so this is numerically safe
        bce = torch.where(is_positive, -torch.log(score), -torch.log(1 - score))
        weights = torch.where(is_positive, self.pos_weight, self.neg_weight)
        return (weights * bce).mean()


def patch_has_foreground(target: torch.Tensor, ignore_label: int = None) -> torch.Tensor:
    """
    target (b, 1, x, y(, z)) label map -> (b, ) bool, True if the sample contains any foreground voxel
    """
    fg = target > 0
    if ignore_label is not None:
        fg &= target != ignore_label
    return fg.flatten(1).any(1)


class SegmentationAndPresenceHeadLoss(nn.Module):
    def __init__(self, segmentation_loss: nn.Module, weight_presence: float, deep_supervision: bool,
                 ignore_label: int = None):
        """
        For networks with an image-level presence head (UNetWithPresenceHead). In training mode they return
        (segmentation output, presence logits). The segmentation output gets the regular nnU-Net loss, the presence
        logits a BCE against "this patch contains foreground". In eval mode (validation) the network only returns the
        segmentation output and only the segmentation loss is computed.
        """
        super().__init__()
        self.segmentation_loss = segmentation_loss
        self.weight_presence = weight_presence
        self.deep_supervision = deep_supervision
        self.ignore_label = ignore_label

    def forward(self, net_output, target):
        if not isinstance(net_output, tuple):
            return self.segmentation_loss(net_output, target)
        seg_output, presence_logits = net_output
        has_fg = patch_has_foreground(target[0] if self.deep_supervision else target, self.ignore_label)
        presence_loss = nn.functional.binary_cross_entropy_with_logits(presence_logits.float(), has_fg.float())
        return self.segmentation_loss(seg_output, target) + self.weight_presence * presence_loss


class SegmentationAndPresenceLoss(nn.Module):
    def __init__(self, segmentation_loss: nn.Module, presence_loss: PresenceLoss, weight_presence: float,
                 deep_supervision: bool):
        """
        segmentation_loss is the regular nnU-Net loss (already wrapped for deep supervision if enabled). The presence
        loss is only applied to the highest resolution output because small structures can disappear in the
        downsampled deep supervision targets.
        """
        super().__init__()
        self.segmentation_loss = segmentation_loss
        self.presence_loss = presence_loss
        self.weight_presence = weight_presence
        self.deep_supervision = deep_supervision

    def forward(self, net_output, target):
        seg_loss = self.segmentation_loss(net_output, target)
        if self.deep_supervision:
            net_output, target = net_output[0], target[0]
        return seg_loss + self.weight_presence * self.presence_loss(net_output, target)
