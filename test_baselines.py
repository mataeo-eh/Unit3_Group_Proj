"""Regression checks for leakage, native pixels, schedules, and saved heads."""
import tempfile
import unittest
from pathlib import Path

import keras
import numpy as np
from PIL import Image as PILImage
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image, Value
from baselines import (split_dataset, preprocess_image, cache_pooled_features,
                       build_head, WarmStableDecay, train_head, evaluate_probabilities,
                       prepare_pooled_backbone, pooled_inference_function)


class BaselineTests(unittest.TestCase):
    def make_data(self, n=120):
        """Use original row IDs and balanced class labels to detect split leakage."""
        return Dataset.from_dict({'label': [i % 3 for i in range(n)], 'original_id': list(range(n))},
            features=Features({'label': ClassLabel(names=['a', 'b', 'c']), 'original_id': Value('int32')}))

    def test_stratification_and_author_splits(self):
        data = self.make_data()
        splits = split_dataset(DatasetDict(train=data))
        self.assertEqual([len(splits[k]) for k in ('train', 'validation', 'test')], [84, 18, 18])
        for part in splits.values():
            self.assertEqual(len(set(np.bincount(part['label']))), 1)
        repeated = split_dataset(DatasetDict(train=data))
        self.assertEqual(splits['validation']['original_id'], repeated['validation']['original_id'])
        author = DatasetDict(train=data.select(range(90)), test=data.select(range(90,120)))
        honored = split_dataset(author)
        self.assertEqual(honored['test']['original_id'], author['test']['original_id'])
        author['validation'] = author['train'].select(range(15))
        author['train'] = author['train'].select(range(15,90))
        honored = split_dataset(author)
        for name in author:
            self.assertEqual(honored[name]['original_id'], author[name]['original_id'])

    def test_native_cache_excludes_test(self):
        image = PILImage.fromarray(np.full((32, 48, 3), 255, dtype=np.uint8))
        pixels = preprocess_image(image)
        self.assertEqual(tuple(pixels.shape), (1,32,48,3))
        np.testing.assert_allclose(pixels.numpy()[0,0,0], (np.ones(3)-[.485,.456,.406])/[.229,.224,.225], rtol=1e-6)
        data = Dataset.from_dict({'image': [image]*3, 'label': [0,1,2]},
            features=Features({'image': Image(), 'label': ClassLabel(names=['a','b','c'])}))
        class SpyBackbone:
            trainable = False
            trainable_weights = []
            def __init__(self):
                import tensorflow as tf
                self.calls = tf.Variable(0)
            def __call__(self, x, training):
                import tensorflow as tf
                self.calls.assign_add(1)
                assert training is False
                tf.debugging.assert_equal(tf.shape(x), [1,32,48,3])
                return np_to_tensor(np.ones((1,768)))
        spy = SpyBackbone()
        cache, report = cache_pooled_features(spy, {'train': data, 'validation': data, 'test': None})
        self.assertEqual(int(spy.calls.numpy()), 6)
        self.assertEqual(report['test_inference_calls'], 0)
        self.assertEqual(set(cache), {'train','validation'})

    def test_pooled_graph_precision_and_weights(self):
        # A dynamic toy backbone exercises graph pruning and checkpoint copying.
        inputs = keras.Input((None, None, 3))
        spatial = keras.layers.Conv2D(8, 3, padding='same')(inputs)
        spatial = keras.layers.LayerNormalization()(spatial)
        pooled = keras.layers.GlobalAveragePooling2D()(spatial)
        original = keras.Model(inputs, {'pooled': pooled, 'feature_map': spatial})
        weights = original.get_weights()
        for mixed in (False, True):
            prepared = prepare_pooled_backbone(original, mixed_precision=mixed)
            infer = pooled_inference_function(prepared)
            self.assertFalse(prepared.trainable_weights)
            for before, after in zip(weights, prepared.get_weights()):
                np.testing.assert_array_equal(before, after)
            for height, width in ((32,48), (48,64)):
                pixels = np.random.default_rng(42).normal(size=(1,height,width,3)).astype('float32')
                expected = original(pixels, training=False)['pooled'].numpy()
                actual = infer(pixels).numpy()
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_allclose(actual, expected, rtol=0.01, atol=0.001)
            self.assertEqual(infer.experimental_get_tracing_count(), 1)

    def test_schedule_heads_and_checkpoint(self):
        schedule = WarmStableDecay(100)
        np.testing.assert_allclose([float(schedule(i)) for i in (0,9,10,79,80,99)], [.0001,.001,.001,.001,.001,.00001], rtol=1e-5)
        restored = keras.optimizers.schedules.deserialize(keras.optimizers.schedules.serialize(schedule))
        self.assertAlmostEqual(float(restored(99)), float(schedule(99)))
        self.assertEqual(build_head(768,6).count_params(), 4614)
        self.assertEqual(build_head(768,6,(8,)).count_params(), 6206)
        rng = np.random.default_rng(42)
        x = rng.normal(size=(60, 8)).astype('float32')
        y = np.arange(60, dtype='int32') % 3
        cache = {'train': (x[:45],y[:45]), 'validation': (x[45:],y[45:])}
        with tempfile.TemporaryDirectory() as directory:
            for hidden, wsd in [((),False),((8,),True)]:
                head, metadata = train_head({'name': f'head_{wsd}', 'hidden_units': hidden, 'learning_rate': .001, 'wsd': wsd},
                    cache, 3, directory, epochs=3, batch_size=15)
                self.assertTrue((Path(directory)/f'head_{wsd}_best.keras').exists())
                scores = evaluate_probabilities(y[45:], head(x[45:],training=False).numpy(), ['a','b','c'])
                self.assertAlmostEqual(scores['loss'], min(metadata['history']['val_loss']), places=5)
                self.assertEqual(metadata['epochs_run'], len(metadata['history']['loss']))

    def test_five_patience_epochs_preserve_history(self):
        # Zero LR produces a true plateau, exercising actual callback behavior.
        x = np.zeros((18, 4), dtype='float32')
        y = np.arange(18, dtype='int32') % 3
        cache = {'train': (x[:12], y[:12]), 'validation': (x[12:], y[12:])}
        with tempfile.TemporaryDirectory() as directory:
            _, metadata = train_head({'name': 'plateau', 'hidden_units': (),
                                      'learning_rate': 0., 'wsd': False},
                cache, 3, directory, epochs=50, batch_size=6, patience=5)
        self.assertEqual(metadata['epochs_run'], 6)
        self.assertEqual(metadata['best_epoch'], 1)
        self.assertTrue(metadata['stopped_early'])
        self.assertEqual(len(metadata['history']['val_loss']), 6)

    def test_majority_metrics(self):
        labels = np.array([0,0,0,1,2])
        p = np.zeros((5,3), dtype='float32'); p[:,0] = 1
        result = evaluate_probabilities(labels,p,['a','b','c'])
        self.assertAlmostEqual(result['accuracy'], .6)
        self.assertAlmostEqual(result['recall_macro'], 1/3)
        self.assertAlmostEqual(result['precision_macro'], .2)
        self.assertAlmostEqual(result['f1_macro'], .25)
        self.assertAlmostEqual(result['roc_auc_macro'], .5)
        self.assertTrue(np.isfinite(result['loss']))


def np_to_tensor(x):
    """Create float32 tensors for the backbone contract spy."""
    import tensorflow as tf
    return tf.convert_to_tensor(x, dtype=tf.float32)


if __name__ == '__main__':
    unittest.main()
