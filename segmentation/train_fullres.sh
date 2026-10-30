#!/bin/bash
for TR in nnUNetTrainerPresenceHead_350epochs nnUNetTrainer_350epochs nnUNetTrainerFPReduction_350epochs; do
  for FOLD in 0 1 2 3 4; do
    nnUNetv2_train 2 3d_fullres $FOLD -tr $TR --npz > $nnUNet_results/logs/${TR}_3d_fullres_fold$FOLD.log 2>&1
  done
done
