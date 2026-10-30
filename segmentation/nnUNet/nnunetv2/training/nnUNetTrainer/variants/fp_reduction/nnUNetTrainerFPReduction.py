import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.presence_loss import PresenceLoss, SegmentationAndPresenceHeadLoss, \
    SegmentationAndPresenceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.fp_reduction.presence_head_network import UNetWithPresenceHead
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerFPReduction(nnUNetTrainer):
    """
    For datasets that mix images with and without foreground (empty masks), e.g. bilateral breast MRI split into
    left/right halves. Network architecture, augmentation, optimizer and Dice+CE loss are unchanged. Added on top:

    1. Case-balanced sampling: positive_case_fraction of the sampled training cases contain foreground, independent
       of how many empty cases are in the dataset.
    2. Presence loss (see nnunetv2/training/loss/presence_loss.py) that penalizes confident false positives in patches
       without foreground.

    Ablation variants: nnUNetTrainerFPR_BalancedOnly (1. only), nnUNetTrainerFPR_PresenceOnly (2. only).
    nnUNetTrainerPresenceHead replaces 2. with a classification head (architecture change) instead of a penalty.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # fraction of sampled training cases that contain foreground. None = uniform sampling (nnU-Net default)
        self.positive_case_fraction = 0.5
        # weight of the presence loss relative to Dice+CE. 0 disables it
        self.presence_loss_weight = 1.0
        self.presence_topk = 64
        self.presence_pos_weight = 1.0
        self.presence_neg_weight = 1.0

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = super()._build_loss()
        if self.presence_loss_weight > 0:
            presence = PresenceLoss(self.presence_topk, self.presence_pos_weight, self.presence_neg_weight,
                                    self.label_manager.ignore_label)
            loss = SegmentationAndPresenceLoss(loss, presence, self.presence_loss_weight, self.enable_deep_supervision)
        return loss

    def get_training_sampling_probabilities(self, dataset_tr):
        if self.positive_case_fraction is None:
            return None
        fg_locations = dataset_tr.foreground_locations
        # tuple keys are regions or the annotated_classes_key (ignore label), not foreground classes
        has_fg = np.array([any(not isinstance(c, tuple) and c > 0 for c in fg_locations.eligible_classes(i))
                           for i in dataset_tr.identifiers])
        num_pos, num_neg = int(has_fg.sum()), int((~has_fg).sum())
        self.print_to_log_file(f"Training cases with foreground: {num_pos}, without foreground: {num_neg}")
        if num_pos == 0 or num_neg == 0:
            self.print_to_log_file("Only one kind of case present, using uniform case sampling")
            return None
        self.print_to_log_file(f"Case-balanced sampling: {self.positive_case_fraction:.2f} of sampled cases contain "
                               f"foreground (uniform sampling would give {num_pos / len(has_fg):.2f})")
        return np.where(has_fg, self.positive_case_fraction / num_pos, (1 - self.positive_case_fraction) / num_neg)


class nnUNetTrainerPresenceHead(nnUNetTrainerFPReduction):
    """
    Case-balanced sampling + an image-level presence classification head on the U-Net bottleneck
    (UNetWithPresenceHead). The segmentation output is trained with the unchanged nnU-Net loss: there is no false
    positive penalty on it, so it is not pushed towards predicting less. Rejecting images without foreground is left
    to the head: at inference its case-level probability (max over sliding windows, mean over folds) is exported with
    the case properties, and nnUNetv2_fp_postprocessing gates the segmentation with a threshold that is tuned on the
    cross-validation results.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.presence_loss_weight = 0
        # weight of the BCE of the presence head relative to Dice+CE
        self.presence_head_loss_weight = 0.5

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        unet = nnUNetTrainer.build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                                        num_output_channels, enable_deep_supervision)
        return UNetWithPresenceHead(unet)

    def _build_loss(self):
        loss = super()._build_loss()
        return SegmentationAndPresenceHeadLoss(loss, self.presence_head_loss_weight, self.enable_deep_supervision,
                                               self.label_manager.ignore_label)


class nnUNetTrainerFPR_BalancedOnly(nnUNetTrainerFPReduction):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.presence_loss_weight = 0


class nnUNetTrainerFPR_PresenceOnly(nnUNetTrainerFPReduction):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.positive_case_fraction = None


class nnUNetTrainerPresenceHead_350epochs(nnUNetTrainerPresenceHead):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 350


class nnUNetTrainerFPReduction_350epochs(nnUNetTrainerFPReduction):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 350

class nnUNetTrainer_350epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 350


class nnUNetTrainerFPR_BalancedOnly_350epochs(nnUNetTrainerFPR_BalancedOnly):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 350


class nnUNetTrainerFPR_PresenceOnly_350epochs(nnUNetTrainerFPR_PresenceOnly):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 350