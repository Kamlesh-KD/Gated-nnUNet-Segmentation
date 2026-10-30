import os
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
import SimpleITK as sitk
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json, write_pickle

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn

import nnunetv2.experiment_planning.patient_splits as patient_splits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.postprocessing import fp_postprocessing
from nnunetv2.training.loss.presence_loss import PresenceLoss, SegmentationAndPresenceHeadLoss, \
    SegmentationAndPresenceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.fp_reduction.nnUNetTrainerFPReduction import nnUNetTrainerFPReduction
from nnunetv2.training.nnUNetTrainer.variants.fp_reduction.presence_head_network import UNetWithPresenceHead


def _logits_and_target():
    # confident background everywhere. Sample 0 is negative, sample 1 has 8 fg voxels, sample 2 a single fg voxel
    logits = torch.zeros(3, 2, 16, 16, 16)
    logits[:, 0] = 3
    target = torch.zeros(3, 1, 16, 16, 16, dtype=torch.int16)
    target[1, 0, 2:4, 2:4, 2:4] = 1
    target[2, 0, 5, 5, 5] = 1
    return logits, target


class TestPresenceLoss(unittest.TestCase):
    def test_false_positive_on_negative_sample_is_penalized(self):
        loss = PresenceLoss(topk=8)
        logits, target = _logits_and_target()
        with_fp = logits.clone()
        with_fp[0, 1, 10:12, 10:12, 10:12] = 6
        self.assertGreater(loss(with_fp, target).item(), loss(logits, target).item() + 0.5)

    def test_gradient_only_reaches_topk_voxels(self):
        logits, target = _logits_and_target()
        logits[0, 1, 10:12, 10:12, 10:12] = 6
        logits.requires_grad_()
        PresenceLoss(topk=8)(logits, target).backward()
        grad = logits.grad[0, 1]
        self.assertEqual(int((grad != 0).sum()), 8)
        self.assertTrue(bool((grad[10:12, 10:12, 10:12] > 0).all()))

    def test_correct_prediction_has_low_loss(self):
        logits, target = _logits_and_target()
        logits[1, 1, 2:4, 2:4, 2:4] = 6
        logits[2, 1, 5, 5, 5] = 6
        # k is capped at the number of fg voxels, so the single voxel lesion is not asked to grow
        self.assertLess(PresenceLoss(topk=64)(logits, target).item(), 0.1)

    def test_ignore_label_is_excluded(self):
        logits, target = _logits_and_target()
        reference = PresenceLoss(topk=8, ignore_label=2)(logits, target).item()
        logits[0, 1, 10:12, 10:12, 10:12] = 6
        target[0, 0, 10:12, 10:12, 10:12] = 2
        self.assertAlmostEqual(PresenceLoss(topk=8, ignore_label=2)(logits, target).item(), reference, places=5)

    @unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
    def test_runs_under_autocast(self):
        logits, target = _logits_and_target()
        expected = PresenceLoss(topk=8)(logits, target).item()
        with torch.autocast('cuda'):
            result = PresenceLoss(topk=8)(logits.cuda().half(), target.cuda())
        self.assertEqual(result.dtype, torch.float32)
        self.assertAlmostEqual(result.item(), expected, places=3)

    def test_deep_supervision_uses_highest_resolution_only(self):
        logits, target = _logits_and_target()
        seg_loss = mock.Mock(return_value=torch.tensor(0.))
        presence = mock.Mock(return_value=torch.tensor(1.))
        loss = SegmentationAndPresenceLoss(seg_loss, presence, 0.5, deep_supervision=True)
        self.assertEqual(loss([logits, logits[:, :, ::2]], [target, target[:, :, ::2]]).item(), 0.5)
        self.assertIs(presence.call_args[0][0], logits)
        self.assertIs(presence.call_args[0][1], target)


def _tiny_presence_head_unet(deep_supervision=True):
    unet = PlainConvUNet(1, 3, [4, 8, 16], nn.Conv3d, 3, [1, 2, 2], 1, 2, 1, deep_supervision=deep_supervision)
    return UNetWithPresenceHead(unet, hidden_features=8)


class TestPresenceHead(unittest.TestCase):
    def test_train_mode_returns_presence_logits(self):
        network = _tiny_presence_head_unet().train()
        seg, presence = network(torch.rand(2, 1, 16, 16, 16))
        self.assertIsInstance(seg, list)
        self.assertEqual(tuple(seg[0].shape), (2, 2, 16, 16, 16))
        self.assertEqual(tuple(presence.shape), (2, ))

    def test_eval_mode_tracks_maximum_probability(self):
        network = _tiny_presence_head_unet(deep_supervision=False).eval()
        self.assertEqual(network.get_case_level_outputs(), {})
        with torch.no_grad():
            x = torch.rand(3, 1, 16, 16, 16)
            seg = network(x)
            expected = torch.sigmoid(network.presence_head(torch.cat(
                (network.encoder(x)[-1].flatten(2).mean(2), network.encoder(x)[-1].flatten(2).amax(2)), 1))).max()
            network(torch.zeros(1, 1, 16, 16, 16))
        self.assertIsInstance(seg, torch.Tensor)
        self.assertGreaterEqual(network.get_case_level_outputs()['presence_probability'], expected.item() - 1e-6)
        network.reset_case_level_outputs()
        self.assertEqual(network.get_case_level_outputs(), {})

    def test_deep_supervision_switch_reaches_decoder(self):
        network = _tiny_presence_head_unet()
        network.decoder.deep_supervision = False
        self.assertFalse(network.unet.decoder.deep_supervision)

    def test_head_loss(self):
        logits, target = _logits_and_target()
        seg_loss = mock.Mock(return_value=torch.tensor(0.))
        loss = SegmentationAndPresenceHeadLoss(seg_loss, 1., deep_supervision=True)
        presence_logits = torch.tensor([-10., 10., 10.])  # matches: sample 0 negative, 1 and 2 positive
        self.assertLess(loss(([logits], presence_logits), [target]).item(), 1e-3)
        self.assertGreater(loss(([logits], -presence_logits), [target]).item(), 5)
        # eval mode: only the segmentation loss
        self.assertEqual(loss([logits], [target]).item(), 0)

    def test_predictor_exports_presence_probability_averaged_over_folds(self):
        network = _tiny_presence_head_unet(deep_supervision=False)
        params = [{k: v.clone() for k, v in network.state_dict().items()} for _ in range(2)]
        params[1]['presence_head.2.bias'] += 3  # second "fold" is more confident
        predictor = nnUNetPredictor(device=torch.device('cpu'), use_mirroring=False, allow_tqdm=False)
        predictor.network = network
        predictor.list_of_parameters = params
        predictor.configuration_manager = SimpleNamespace(patch_size=[16, 16, 16])
        predictor.label_manager = SimpleNamespace(num_segmentation_heads=2)
        data = torch.rand(1, 24, 16, 16)
        per_fold = []
        for p in params:
            network.load_state_dict(p)
            predictor.predict_sliding_window_return_logits(data)
            per_fold.append(predictor.case_level_outputs['presence_probability'])
        self.assertGreater(per_fold[1], per_fold[0])
        predictor.predict_logits_from_preprocessed_data(data)
        self.assertAlmostEqual(predictor.case_level_outputs['presence_probability'], np.mean(per_fold), places=5)

    def test_regular_network_has_no_case_level_outputs(self):
        predictor = nnUNetPredictor(device=torch.device('cpu'), use_mirroring=False, allow_tqdm=False)
        predictor.network = _tiny_presence_head_unet(deep_supervision=False).unet
        predictor.list_of_parameters = [predictor.network.state_dict()]
        predictor.configuration_manager = SimpleNamespace(patch_size=[16, 16, 16])
        predictor.label_manager = SimpleNamespace(num_segmentation_heads=2)
        predictor.predict_logits_from_preprocessed_data(torch.rand(1, 16, 16, 16))
        self.assertEqual(predictor.case_level_outputs, {})


class _FakeForegroundLocations:
    def __init__(self, positives):
        self.positives = positives

    def eligible_classes(self, identifier):
        return [1] if identifier in self.positives else []


def _trainer_stub(cls, positive_case_fraction=0.5):
    trainer = cls.__new__(cls)
    trainer.positive_case_fraction = positive_case_fraction
    trainer.print_to_log_file = lambda *args, **kwargs: None
    return trainer


class TestCaseBalancedSampling(unittest.TestCase):
    def setUp(self):
        identifiers = [f'pos_{i}' for i in range(10)] + [f'neg_{i}' for i in range(30)]
        self.dataset = SimpleNamespace(identifiers=identifiers, foreground_locations=_FakeForegroundLocations(
            set(identifiers[:10])))

    def test_default_trainer_samples_uniformly(self):
        self.assertIsNone(_trainer_stub(nnUNetTrainer).get_training_sampling_probabilities(self.dataset))

    def test_positive_fraction(self):
        p = _trainer_stub(nnUNetTrainerFPReduction, 0.5).get_training_sampling_probabilities(self.dataset)
        self.assertAlmostEqual(p.sum(), 1)
        self.assertAlmostEqual(p[:10].sum(), 0.5)
        p = _trainer_stub(nnUNetTrainerFPReduction, 0.7).get_training_sampling_probabilities(self.dataset)
        self.assertAlmostEqual(p[:10].sum(), 0.7)

    def test_disabled_or_single_kind(self):
        self.assertIsNone(_trainer_stub(nnUNetTrainerFPReduction, None).get_training_sampling_probabilities(
            self.dataset))
        self.dataset.foreground_locations = _FakeForegroundLocations(set(self.dataset.identifiers))
        self.assertIsNone(_trainer_stub(nnUNetTrainerFPReduction).get_training_sampling_probabilities(self.dataset))


def _write(array, file, spacing=(1., 1., 1.)):
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.SetSpacing(spacing)
    sitk.WriteImage(image, file)


def _cube(start, size, shape=(30, 30, 30)):
    a = np.zeros(shape)
    a[start:start + size, start:start + size, start:start + size] = 1
    return a


class TestFPPostprocessing(unittest.TestCase):
    """
    Cases are written the way nnU-Net exports validation/prediction results with probabilities: mask (.nii.gz),
    probabilities (.npz) and properties (.pkl, contains presence_probability for presence head networks)
    """
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.pred, self.gt, self.out = join(self.tmp.name, 'pred'), join(self.tmp.name, 'gt'), join(self.tmp.name, 'out')
        os.makedirs(self.pred)
        os.makedirs(self.gt)

    def tearDown(self):
        self.tmp.cleanup()

    def _case(self, name, reference, foreground_probability, presence=None):
        _write(reference, join(self.gt, name + '.nii.gz'))
        _write(foreground_probability >= 0.5, join(self.pred, name + '.nii.gz'))
        np.savez(join(self.pred, name + '.npz'),
                 probabilities=np.stack([1 - foreground_probability, foreground_probability]))
        write_pickle({} if presence is None else {'presence_probability': presence}, join(self.pred, name + '.pkl'))

    def _run(self, min_sensitivity=0.):
        tmp = self.tmp.name
        with mock.patch('builtins.print'):
            fp_postprocessing.evaluate(SimpleNamespace(p=self.pred, g=self.gt, o=join(tmp, 'raw.json'),
                                                       file_ending='.nii.gz'))
            fp_postprocessing.tune(SimpleNamespace(p=[self.pred], g=self.gt, o=join(tmp, 'pp.json'),
                                                   file_ending='.nii.gz', min_sensitivity=min_sensitivity))
            fp_postprocessing.apply(SimpleNamespace(p=self.pred, params=join(tmp, 'pp.json'), o=self.out,
                                                    file_ending='.nii.gz'))
            fp_postprocessing.evaluate(SimpleNamespace(p=self.out, g=self.gt, o=join(tmp, 'pp_metrics.json'),
                                                       file_ending='.nii.gz'))
        return (load_json(join(tmp, 'raw.json'))['summary'], load_json(join(tmp, 'pp.json')),
                load_json(join(tmp, 'pp_metrics.json'))['summary'])

    def test_small_false_positive_is_removed_by_min_volume(self):
        # 216 mm3 lesion predicted with p=0.9, 27 mm3 false positive with p=0.6 on a lesion free image
        lesion = _cube(5, 6)
        self._case('pos', lesion, 0.9 * lesion)
        self._case('neg', np.zeros_like(lesion), 0.6 * _cube(20, 3))
        raw, params, post = self._run()
        self.assertEqual(raw['false_positive_images'], 1)
        self.assertAlmostEqual(raw['case_dice'], 0.5)
        self.assertEqual(params['presence_threshold'], 0)
        self.assertTrue(27 < params['min_volume_mm3'] <= 216)
        self.assertEqual(post['false_positive_images'], 0)
        self.assertAlmostEqual(post['dice_on_positive_cases'], 1)

    def test_presence_gating_removes_large_false_positive(self):
        # the false positive is as large and confident as the lesion: only the presence head can tell them apart
        lesion = _cube(5, 8)
        self._case('pos', lesion, 0.95 * lesion, presence=0.9)
        self._case('neg', np.zeros_like(lesion), 0.95 * _cube(15, 8), presence=0.05)
        raw, params, post = self._run()
        self.assertAlmostEqual(raw['case_dice'], 0.5)
        self.assertTrue(0.05 < params['presence_threshold'] <= 0.9)
        self.assertAlmostEqual(params['case_dice_tuned'], 1)
        self.assertEqual(post['false_positive_images'], 0)
        self.assertAlmostEqual(post['sensitivity'], 1)

    def test_min_sensitivity_constraint(self):
        # removing the 64 mm3 false positives also removes the 27 mm3 lesion. Unconstrained tuning trades the
        # lesion for two clean images, min_sensitivity=1 must keep the lesion
        lesion = _cube(5, 3)
        self._case('pos', lesion, 0.9 * lesion)
        for i in range(2):
            self._case(f'neg{i}', np.zeros_like(lesion), 0.95 * _cube(15, 4))
        _, params, _ = self._run()
        self.assertEqual(params['sensitivity_tuned'], 0)
        _, params, post = self._run(min_sensitivity=1.)
        self.assertEqual(params['sensitivity_tuned'], 1)
        self.assertAlmostEqual(post['sensitivity'], 1)


class TestPatientSplits(unittest.TestCase):
    def test_patients_do_not_cross_folds(self):
        with TemporaryDirectory() as tmp:
            dataset = 'Dataset999_Test'
            labels = join(tmp, dataset, 'labelsTr')
            os.makedirs(labels)
            save_json({'file_ending': '.nii.gz'}, join(tmp, dataset, 'dataset.json'))
            # 10 patients x 2 scans x left/right, the lesion is in one side per patient
            patient_of, has_fg = {}, {}
            for p in range(10):
                for scan in ('00', '01'):
                    for side in ('left', 'right'):
                        identifier = f'{side}_case_{p:03d}_{scan}'
                        patient_of[identifier] = p
                        has_fg[identifier] = side == ('left' if p % 2 else 'right')
                        seg = np.zeros((4, 4, 4))
                        seg[0, 0, 0] = has_fg[identifier]
                        _write(seg, join(labels, identifier + '.nii.gz'))
            with mock.patch.object(patient_splits, 'nnUNet_raw', tmp), mock.patch('builtins.print'):
                splits = patient_splits.create_patient_splits(dataset)
        self.assertEqual(len(splits), 5)
        all_val = sum([s['val'] for s in splits], [])
        self.assertEqual(len(all_val), 40)
        self.assertEqual(len(set(all_val)), 40)
        for s in splits:
            self.assertFalse({patient_of[i] for i in s['val']} & {patient_of[i] for i in s['train']})
            self.assertEqual(sum(has_fg[i] for i in s['val']), len(s['val']) // 2)


if __name__ == '__main__':
    unittest.main()
