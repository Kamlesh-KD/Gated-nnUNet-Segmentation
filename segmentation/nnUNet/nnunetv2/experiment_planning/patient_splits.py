"""
Create nnU-Net's splits_final.json so that
  - all images of one patient (e.g. left/right halves of a bilateral scan, repeated scans) end up in the same fold,
  - every fold has a similar ratio of images with and without foreground.

nnU-Net's default split is random over image identifiers. If several images come from the same patient, they would
otherwise land in train and validation at the same time, which inflates validation scores.

Run after nnUNetv2_plan_and_preprocess and before nnUNetv2_train:
    nnUNetv2_create_patient_splits -d 2
"""
import argparse
import os
import re

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, save_json, subfiles
from sklearn.model_selection import StratifiedGroupKFold

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def create_patient_splits(dataset_name: str, patient_regex: str = r'case_\d+', n_splits: int = 5,
                          seed: int = 12345):
    labels_folder = join(nnUNet_raw, dataset_name, 'labelsTr')
    file_ending = load_json(join(nnUNet_raw, dataset_name, 'dataset.json'))['file_ending']

    identifiers, patients, has_fg = [], [], []
    for f in subfiles(labels_folder, suffix=file_ending, join=False):
        identifier = f[:-len(file_ending)]
        match = re.search(patient_regex, identifier)
        if match is None:
            raise RuntimeError(f'patient_regex {patient_regex} does not match {identifier}')
        identifiers.append(identifier)
        patients.append(match.group())
        has_fg.append(bool(np.any(sitk.GetArrayFromImage(sitk.ReadImage(join(labels_folder, f))) > 0)))
    identifiers, patients, has_fg = np.array(identifiers), np.array(patients), np.array(has_fg)
    print(f'{len(identifiers)} images ({has_fg.sum()} with foreground) from {len(np.unique(patients))} patients')

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fold, (tr_idx, val_idx) in enumerate(splitter.split(identifiers, has_fg, groups=patients)):
        splits.append({'train': sorted(identifiers[tr_idx].tolist()), 'val': sorted(identifiers[val_idx].tolist())})
        print(f'fold {fold}: train {len(tr_idx)} ({has_fg[tr_idx].sum()} with fg), '
              f'val {len(val_idx)} ({has_fg[val_idx].sum()} with fg), '
              f'val patients {sorted(set(patients[val_idx].tolist()))}')
    return splits


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', required=True, help='dataset name or id, e.g. 2')
    parser.add_argument('--patient_regex', default=r'case_\d+',
                        help='regex that extracts the patient id from an image identifier (default: case_\\d+)')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--overwrite', action='store_true',
                        help='replace an existing splits_final.json (models trained on the old split become invalid)')
    args = parser.parse_args()

    dataset_name = maybe_convert_to_dataset_name(args.d)
    splits_file = join(nnUNet_preprocessed, dataset_name, 'splits_final.json')
    if isfile(splits_file) and not args.overwrite:
        raise RuntimeError(f'{splits_file} exists. Use --overwrite to replace it')

    splits = create_patient_splits(dataset_name, args.patient_regex, args.n_splits, args.seed)
    os.makedirs(os.path.dirname(splits_file), exist_ok=True)
    save_json(splits, splits_file, sort_keys=False)
    print(f'saved {splits_file}')


if __name__ == '__main__':
    entry_point()
