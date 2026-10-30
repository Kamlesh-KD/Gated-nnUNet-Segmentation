"""
Case-level evaluation and false positive post-processing for datasets with empty reference masks.

nnU-Net's summary.json reports Dice per case, which is NaN (and therefore ignored) when reference and prediction are
both empty. It does not tell how many images without foreground received a false positive. This tool reports that
and tunes a simple post-processing on the cross-validation predictions:
    if the case-level presence probability < presence_threshold: empty prediction (networks with a presence head,
        nnUNetTrainerPresenceHead, only; the probability is read from the .pkl next to the .npz)
    otherwise foreground probability >= threshold, then remove connected components smaller than min_volume_mm3
Optionally only settings that keep the sensitivity (fraction of cases with foreground that are detected) above
--min_sensitivity are considered.

Commands
    evaluate  score predicted masks against the reference
              nnUNetv2_fp_postprocessing evaluate -p PRED_DIR -g LABELS_DIR -o metrics.json
    tune      grid search threshold / min_volume_mm3 on validation probabilities (nnUNetv2_train ... --npz)
              nnUNetv2_fp_postprocessing tune -p fold_0/validation fold_1/validation ... -g LABELS_DIR -o pp.json
    apply     apply tuned parameters to predictions made with nnUNetv2_predict ... --save_probabilities
              nnUNetv2_fp_postprocessing apply -p PRED_DIR --params pp.json -o OUT_DIR

Tuning objective ("case Dice"): mean over all cases of Dice for cases with foreground, and for cases without
foreground 1 if the prediction is empty, else 0. It rewards segmentation quality and penalizes false positive images
equally. Foreground = all labels > 0 (binary).
"""
import argparse
import os

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join, load_json, load_pickle, save_json, subfiles
from scipy import ndimage

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# 0 = no gating
PRESENCE_THRESHOLDS = (0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_VOLUMES_MM3 = (0, 10, 25, 50, 100, 250, 500, 1000)
STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def voxel_volume(image: sitk.Image) -> float:
    return float(np.prod(image.GetSpacing()))


def foreground_probability(npz_file: str) -> np.ndarray:
    probabilities = np.load(npz_file)['probabilities']
    return 1 - probabilities[0].astype(np.float32)


def postprocess(fg_prob: np.ndarray, threshold: float, min_volume_mm3: float, voxel_mm3: float) -> np.ndarray:
    mask = fg_prob >= threshold
    if min_volume_mm3 > 0 and mask.any():
        components, n = ndimage.label(mask, STRUCTURE)
        sizes = np.bincount(components.ravel(), minlength=n + 1) * voxel_mm3
        keep = sizes >= min_volume_mm3
        keep[0] = False
        mask = keep[components]
    return mask


def case_metrics(pred: np.ndarray, gt: np.ndarray, voxel_mm3: float) -> dict:
    gt_volume, pred_volume = int(gt.sum()), int(pred.sum())
    intersection = int((pred & gt).sum())
    metrics = {'reference_positive': gt_volume > 0, 'predicted_positive': pred_volume > 0,
               'reference_volume_mm3': gt_volume * voxel_mm3, 'predicted_volume_mm3': pred_volume * voxel_mm3}
    if gt_volume > 0:
        metrics['dice'] = 2 * intersection / (gt_volume + pred_volume)
        metrics['detected'] = intersection > 0
        metrics['case_dice'] = metrics['dice']
    else:
        metrics['false_positive_volume_mm3'] = pred_volume * voxel_mm3
        metrics['case_dice'] = float(pred_volume == 0)
    return metrics


def summarize(per_case: dict) -> dict:
    pos = [m for m in per_case.values() if m['reference_positive']]
    neg = [m for m in per_case.values() if not m['reference_positive']]
    summary = {'num_cases': len(per_case), 'num_reference_positive': len(pos), 'num_reference_negative': len(neg),
               'case_dice': float(np.mean([m['case_dice'] for m in per_case.values()]))}
    if pos:
        summary['dice_on_positive_cases'] = float(np.mean([m['dice'] for m in pos]))
        summary['sensitivity'] = float(np.mean([m['detected'] for m in pos]))
    if neg:
        summary['specificity'] = float(np.mean([not m['predicted_positive'] for m in neg]))
        summary['false_positive_images'] = int(sum(m['predicted_positive'] for m in neg))
        summary['mean_false_positive_volume_mm3'] = float(np.mean([m['false_positive_volume_mm3'] for m in neg]))
    return summary


def read_mask(file: str):
    image = sitk.ReadImage(file)
    return sitk.GetArrayFromImage(image) > 0, image


def evaluate(args):
    per_case = {}
    for f in subfiles(args.p, suffix=args.file_ending, join=False):
        pred, _ = read_mask(join(args.p, f))
        gt, gt_image = read_mask(join(args.g, f))
        per_case[f] = case_metrics(pred, gt, voxel_volume(gt_image))
    result = {'summary': summarize(per_case), 'per_case': per_case}
    save_json(result, args.o, sort_keys=False)
    print(result['summary'])


def presence_probability(pred_dir: str, case: str):
    """
    Case-level presence probability that nnUNetPredictor exports for networks with a presence head
    (nnUNetTrainerPresenceHead). None for regular networks.
    """
    return load_pickle(join(pred_dir, case + '.pkl')).get('presence_probability')


def tune(args):
    case_dice, detected, is_positive, presence = [], [], [], []
    for pred_dir in args.p:
        for f in subfiles(pred_dir, suffix='.npz', join=False):
            case = f[:-4]
            gt, gt_image = read_mask(join(args.g, case + args.file_ending))
            voxel_mm3 = voxel_volume(gt_image)
            fg_prob = foreground_probability(join(pred_dir, f))
            assert fg_prob.shape == gt.shape, f'{case}: probabilities {fg_prob.shape} vs reference {gt.shape}'
            gt_volume = gt.sum() * voxel_mm3
            # [threshold index, min volume index]
            dice_here = np.zeros((len(THRESHOLDS), len(MIN_VOLUMES_MM3)))
            detected_here = np.zeros_like(dice_here, dtype=bool)
            for i, threshold in enumerate(THRESHOLDS):
                components, n = ndimage.label(fg_prob >= threshold, STRUCTURE)
                sizes = np.bincount(components.ravel(), minlength=n + 1) * voxel_mm3
                overlaps = np.bincount(components[gt], minlength=n + 1) * voxel_mm3
                for j, min_volume in enumerate(MIN_VOLUMES_MM3):
                    keep = sizes >= min_volume
                    keep[0] = False
                    pred_volume, intersection = sizes[keep].sum(), overlaps[keep].sum()
                    if gt_volume > 0:
                        dice_here[i, j] = 2 * intersection / (gt_volume + pred_volume)
                        detected_here[i, j] = intersection > 0
                    else:
                        dice_here[i, j] = float(pred_volume == 0)
            case_dice.append(dice_here)
            detected.append(detected_here)
            is_positive.append(gt_volume > 0)
            presence.append(presence_probability(pred_dir, case))
            print(f'{case}: default (0.5, 0 mm3) case Dice {dice_here[THRESHOLDS.index(0.5), 0]:.3f}, '
                  f'presence probability {presence[-1]}')

    case_dice, detected, is_positive = np.array(case_dice), np.array(detected), np.array(is_positive)
    use_gating = all(p is not None for p in presence)
    gates = PRESENCE_THRESHOLDS if use_gating else (0,)
    # [gate index, threshold index, min volume index]
    mean_case_dice = np.zeros((len(gates), len(THRESHOLDS), len(MIN_VOLUMES_MM3)))
    sensitivity = np.ones_like(mean_case_dice)
    for g, gate in enumerate(gates):
        rejected = np.array([use_gating and p < gate for p in presence])
        # a rejected case gets an empty prediction: case Dice 1 without foreground, 0 with foreground
        gated_dice = np.where(rejected[:, None, None], (~is_positive).astype(float)[:, None, None], case_dice)
        mean_case_dice[g] = gated_dice.mean(0)
        if is_positive.any():
            sensitivity[g] = (detected & ~rejected[:, None, None])[is_positive].mean(0)

    allowed = sensitivity >= args.min_sensitivity
    if not allowed.any():
        print(f'WARNING: no setting reaches sensitivity {args.min_sensitivity}, using the most sensitive ones')
        allowed = sensitivity == sensitivity.max()
    g, i, j = np.unravel_index(np.argmax(np.where(allowed, mean_case_dice, -1)), mean_case_dice.shape)
    default = (0, THRESHOLDS.index(0.5), 0)
    params = {'presence_threshold': gates[g], 'threshold': THRESHOLDS[i], 'min_volume_mm3': MIN_VOLUMES_MM3[j],
              'min_sensitivity': args.min_sensitivity,
              'case_dice_tuned': float(mean_case_dice[g, i, j]), 'sensitivity_tuned': float(sensitivity[g, i, j]),
              'case_dice_default': float(mean_case_dice[default]), 'sensitivity_default': float(sensitivity[default]),
              'num_cases': len(case_dice), 'presence_thresholds': gates, 'thresholds': THRESHOLDS,
              'min_volumes_mm3': MIN_VOLUMES_MM3,
              'case_dice_grid': mean_case_dice.tolist(), 'sensitivity_grid': sensitivity.tolist()}
    save_json(params, args.o, sort_keys=False)
    print(f"best: presence_threshold {params['presence_threshold']}, threshold {params['threshold']}, "
          f"min_volume_mm3 {params['min_volume_mm3']}: case Dice {params['case_dice_tuned']:.4f}, "
          f"sensitivity {params['sensitivity_tuned']:.4f} (default: case Dice {params['case_dice_default']:.4f}, "
          f"sensitivity {params['sensitivity_default']:.4f})")


def apply(args):
    params = load_json(args.params)
    os.makedirs(args.o, exist_ok=True)
    for f in subfiles(args.p, suffix='.npz', join=False):
        case = f[:-4]
        # the mask written by nnU-Net next to the .npz carries the geometry
        reference_image = sitk.ReadImage(join(args.p, case + args.file_ending))
        fg_prob = foreground_probability(join(args.p, f))
        mask = postprocess(fg_prob, params['threshold'], params['min_volume_mm3'], voxel_volume(reference_image))
        if params['presence_threshold'] > 0 and presence_probability(args.p, case) < params['presence_threshold']:
            mask[:] = False
        out = sitk.GetImageFromArray(mask.astype(np.uint8))
        out.CopyInformation(reference_image)
        sitk.WriteImage(out, join(args.o, case + args.file_ending), True)
        print(f'{case}: {"foreground" if mask.any() else "empty"}')


def entry_point():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('evaluate')
    p.add_argument('-p', required=True, help='folder with predicted masks')
    p.add_argument('-g', required=True, help='folder with reference masks (e.g. labelsTr)')
    p.add_argument('-o', required=True, help='output json')
    p.set_defaults(func=evaluate)

    p = sub.add_parser('tune')
    p.add_argument('-p', required=True, nargs='+', help='folder(s) with validation .npz probabilities')
    p.add_argument('-g', required=True, help='folder with reference masks (e.g. labelsTr)')
    p.add_argument('-o', required=True, help='output json with the selected parameters')
    p.add_argument('--min_sensitivity', type=float, default=0.,
                   help='only consider settings that detect at least this fraction of the cases with foreground '
                        '(e.g. 0.95). Default: no constraint')
    p.set_defaults(func=tune)

    p = sub.add_parser('apply')
    p.add_argument('-p', required=True, help='folder with .npz probabilities and predicted masks')
    p.add_argument('--params', required=True, help='json written by tune')
    p.add_argument('-o', required=True, help='output folder')
    p.set_defaults(func=apply)

    for p in sub.choices.values():
        p.add_argument('--file_ending', default='.nii.gz')
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    entry_point()
