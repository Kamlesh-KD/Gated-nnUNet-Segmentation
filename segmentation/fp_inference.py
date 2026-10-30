"""
Inference with the nnU-Net models trained for Dataset002 (nnUNetTrainerPresenceHead, nnUNetTrainerFPReduction,
nnUNetTrainer, ...) in any of the configurations 3d_lowres, 3d_fullres and 3d_cascade_fullres.

Steps per image:
  1. nnU-Net prediction (sliding window, mirroring, ensemble of the selected folds). For 3d_cascade_fullres the
     3d_lowres model of the same trainer is run first and its prediction is used as additional input.
  2. Post-processing tuned on cross-validation (nnUNetv2_fp_postprocessing tune), read from fp_postprocessing.json
     in the model folder: presence gating (presence head models), probability threshold, minimum lesion volume.
     Without that file the raw nnU-Net prediction is returned.

Command line (nnUNet_results etc. must be set, e.g. `set -a; . /workspace/.env; set +a`):
    python fp_inference.py -i IMAGE.nii.gz_or_FOLDER -o OUT -c 3d_fullres -tr nnUNetTrainerPresenceHead
Python / notebook:
    from fp_inference import predict_single_image
    result = predict_single_image('/path/image.nii.gz', '/path/out', '3d_cascade_fullres')
"""
import argparse
import os
import shutil
from typing import Sequence, Union

import numpy as np
import SimpleITK as sitk
import torch
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, save_json, subfiles

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results
from nnunetv2.postprocessing.fp_postprocessing import foreground_probability, postprocess, presence_probability, \
    voxel_volume

DATASET = 'Dataset002_dartmouth_unilateral_mixed'
PLANS = 'nnUNetPlans'
CONFIGURATIONS = ('3d_lowres', '3d_fullres', '3d_cascade_fullres')
FILE_ENDING = '.nii.gz'


def model_folder(trainer: str, configuration: str) -> str:
    return join(nnUNet_results, DATASET, f'{trainer}__{PLANS}__{configuration}')


def available_folds(trainer: str, configuration: str, checkpoint: str = 'checkpoint_final.safetensors') -> list:
    """folds of a model that have the requested checkpoint ([] if the model has not been trained)"""
    folder = model_folder(trainer, configuration)
    if not os.path.isdir(folder):
        return []
    return sorted(int(d[5:]) for d in os.listdir(folder)
                  if d.startswith('fold_') and d[5:].isdigit() and isfile(join(folder, d, checkpoint)))


def default_postprocessing_params(trainer: str, configuration: str) -> Union[dict, None]:
    """fp_postprocessing.json written by nnUNetv2_fp_postprocessing tune (see train.docx), if present"""
    file = join(model_folder(trainer, configuration), 'fp_postprocessing.json')
    return load_json(file) if isfile(file) else None


def predict_folder(input_folder: str, output_folder: str, configuration: str,
                   trainer: str = 'nnUNetTrainerPresenceHead', folds: Sequence[Union[int, str]] = (0, 1, 2, 3, 4),
                   checkpoint: str = 'checkpoint_final.safetensors', device: str = 'cuda',
                   save_probabilities: bool = True, prev_stage_folder: str = None):
    """
    Raw nnU-Net prediction of all images in input_folder (nnU-Net naming: CASE_0000.nii.gz). Writes CASE.nii.gz and,
    with save_probabilities, CASE.npz (probabilities) and CASE.pkl (properties incl. presence_probability).
    """
    assert configuration in CONFIGURATIONS, f'configuration must be one of {CONFIGURATIONS}'
    if configuration == '3d_cascade_fullres' and prev_stage_folder is None:
        prev_stage_folder = output_folder.rstrip('/') + '_lowres_stage'
        print(f'3d_cascade_fullres: predicting the 3d_lowres stage first -> {prev_stage_folder}')
        predict_folder(input_folder, prev_stage_folder, '3d_lowres', trainer, folds, checkpoint, device,
                       save_probabilities=False)

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                device=torch.device(device), allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(model_folder(trainer, configuration), tuple(folds), checkpoint)
    predictor.predict_from_files(input_folder, output_folder, save_probabilities=save_probabilities, overwrite=True,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=prev_stage_folder)


def postprocess_folder(pred_folder: str, output_folder: str, params: Union[dict, None]) -> dict:
    """
    Applies the tuned post-processing to every case in pred_folder (needs .npz/.pkl, i.e. save_probabilities).
    params None = keep the raw nnU-Net mask. Returns per case: presence probability, lesion present, lesion volume.
    """
    os.makedirs(output_folder, exist_ok=True)
    results = {}
    for f in subfiles(pred_folder, suffix='.npz', join=False):
        case = f[:-4]
        raw_mask_file = join(pred_folder, case + FILE_ENDING)
        raw_image = sitk.ReadImage(raw_mask_file)
        presence = presence_probability(pred_folder, case)
        if params is None:
            mask = sitk.GetArrayFromImage(raw_image) > 0
        else:
            mask = postprocess(foreground_probability(join(pred_folder, f)), params['threshold'],
                               params['min_volume_mm3'], voxel_volume(raw_image))
            if params['presence_threshold'] > 0 and presence is not None and presence < params['presence_threshold']:
                mask[:] = False
        out = sitk.GetImageFromArray(mask.astype(np.uint8))
        out.CopyInformation(raw_image)
        sitk.WriteImage(out, join(output_folder, case + FILE_ENDING), True)
        results[case] = {'mask': join(output_folder, case + FILE_ENDING), 'raw_mask': raw_mask_file,
                         'presence_probability': presence, 'lesion_present': bool(mask.any()),
                         'lesion_volume_mm3': float(mask.sum() * voxel_volume(raw_image))}
    save_json(results, join(output_folder, 'results.json'), sort_keys=False)
    return results


def predict(input_path: str, output_folder: str, configuration: str, trainer: str = 'nnUNetTrainerPresenceHead',
            folds: Sequence[Union[int, str]] = (0, 1, 2, 3, 4), checkpoint: str = 'checkpoint_final.safetensors',
            device: str = 'cuda', postprocessing_params: Union[str, dict, None] = 'auto') -> dict:
    """
    input_path: one image (any file name) or a folder in nnU-Net naming (CASE_0000.nii.gz).
    postprocessing_params: 'auto' = fp_postprocessing.json of the model if present, None = raw prediction,
                           a dict or a path to a json written by nnUNetv2_fp_postprocessing tune.
    Output: output_folder/raw (nnU-Net output) and output_folder/final (post-processed masks + results.json)
    """
    if os.path.isfile(input_path):
        name = os.path.basename(input_path)[:-len(FILE_ENDING)]
        case = name[:-5] if name.endswith('_0000') else name
        input_folder = join(output_folder, 'input')
        os.makedirs(input_folder, exist_ok=True)
        shutil.copy(input_path, join(input_folder, case + '_0000' + FILE_ENDING))
    else:
        input_folder = input_path

    if postprocessing_params == 'auto':
        postprocessing_params = default_postprocessing_params(trainer, configuration)
        print('post-processing:', postprocessing_params if postprocessing_params is None else
              {k: postprocessing_params[k] for k in ('presence_threshold', 'threshold', 'min_volume_mm3')})
    elif isinstance(postprocessing_params, str):
        postprocessing_params = load_json(postprocessing_params)

    predict_folder(input_folder, join(output_folder, 'raw'), configuration, trainer, folds, checkpoint, device)
    return postprocess_folder(join(output_folder, 'raw'), join(output_folder, 'final'), postprocessing_params)


def predict_single_image(image_path: str, output_folder: str, configuration: str = '3d_fullres',
                         trainer: str = 'nnUNetTrainerPresenceHead', folds: Sequence[Union[int, str]] = (0, 1, 2, 3, 4),
                         checkpoint: str = 'checkpoint_final.safetensors', device: str = 'cuda',
                         postprocessing_params: Union[str, dict, None] = 'auto') -> dict:
    """Returns the result dict of the single case (mask path, presence probability, lesion present, volume)"""
    results = predict(image_path, output_folder, configuration, trainer, folds, checkpoint, device,
                      postprocessing_params)
    assert len(results) == 1
    return next(iter(results.values()))


def entry_point():
    parser = argparse.ArgumentParser(description='Inference with the Dataset002 nnU-Net models incl. tuned '
                                                 'false positive post-processing')
    parser.add_argument('-i', required=True, help='image file or folder with CASE_0000.nii.gz images')
    parser.add_argument('-o', required=True, help='output folder (raw/ and final/ are created inside)')
    parser.add_argument('-c', required=True, choices=CONFIGURATIONS)
    parser.add_argument('-tr', default='nnUNetTrainerPresenceHead')
    parser.add_argument('-f', nargs='+', default=['0', '1', '2', '3', '4'], help='folds to ensemble (or "all")')
    parser.add_argument('-chk', default='checkpoint_final.safetensors')
    parser.add_argument('-device', default='cuda')
    parser.add_argument('--pp_params', default='auto',
                        help='"auto" (fp_postprocessing.json in the model folder), "none" or a json file')
    args = parser.parse_args()
    folds = [f if f == 'all' else int(f) for f in args.f]
    pp = None if args.pp_params.lower() == 'none' else args.pp_params
    results = predict(args.i, args.o, args.c, args.tr, folds, args.chk, args.device, pp)
    for case, r in results.items():
        print(f"{case}: lesion present {r['lesion_present']}, volume {r['lesion_volume_mm3']:.0f} mm3, "
              f"presence probability {r['presence_probability']}, mask {r['mask']}")


if __name__ == '__main__':
    entry_point()
