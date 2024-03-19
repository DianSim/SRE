import os
import tensorflow as tf
from config import config
import numpy as np
from scipy.stats import pearsonr
import tensorflow_addons as tfa
from sklearn.metrics import f1_score


class Test:
    def __init__(self, dataset):
        self.dataset = dataset

    def test(self):
        model_name = config['model_name']
        models_dir = './models'
        checkpoint_dir = os.path.join(models_dir, model_name, 'checkpoints')

        files = os.listdir(checkpoint_dir)
        if len(files):
            # latest_checkpoint= max(files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
            latest_checkpoint = 'ckpt_6.keras'
            model = tf.keras.models.load_model(os.path.join(checkpoint_dir, latest_checkpoint))
            print(f"Loaded latest checkpoint: {latest_checkpoint}")
            model.evaluate(self.dataset)

            y_true = []
            y_pred = []
            for x, y in self.dataset:
                pred = np.argmax(model(x), axis=-1)
                y_true += list(y.numpy())
                y_pred += list(pred)
            f1_scores = f1_score(y_true, y_pred, average=None)

            for i, score in enumerate(f1_scores):
                print(f'f1_score class_{i}: {score}')
        else:
            print(f"There's no saved checkpoint found in the {checkpoint_dir} dir.")