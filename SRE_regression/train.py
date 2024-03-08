import tensorflow as tf
from callbacks import WeightsSaver , TensorboardCallback
from config import config
import os
from scipy.stats import pearsonr
import tensorflow_addons as tfa


config_train_params = config['train_params']

class Train:
    def __init__(self, model_object, train_dataset, dev_dataset):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.model = model_object
        models_dir = './models'
        self.model_name = config['model_name']
        self.checkpoint_dir = os.path.join(models_dir, self.model_name, 'checkpoints')
        self.summary_dir = os.path.join(models_dir, self.model_name, 'summaries')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        self.summary_train = os.path.join(self.summary_dir, 'train')
        self.summary_test = os.path.join(self.summary_dir, 'test')
        os.makedirs(self.summary_train, exist_ok=True)
        os.makedirs(self.summary_test, exist_ok=True)

    def train(self):
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tfa.metrics.PearsonsCorrelation()] 
            )
        
        checkpoint_cb = WeightsSaver(self.model, self.checkpoint_dir)
        tensorboard_cb = TensorboardCallback(self.summary_dir)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                             patience=3,
                                                             restore_best_weights=True)

        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            checkpoint = tf.train.Checkpoint(model=self.model)
            checkpoint.restore(latest_checkpoint)
            print(f"Loaded latest checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoint found. Training from scratch.")
        
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.dev_dataset,
            epochs=config_train_params['epochs'],
            callbacks=[checkpoint_cb, tensorboard_cb, early_stopping_cb]
            )
