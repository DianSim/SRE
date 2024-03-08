import tensorflow as tf
import os
from config import config


class TensorboardCallback(tf.keras.callbacks.Callback):
    """
    this class is for tensorboard summaries
    implement __init__() and on_train_batch_end() functions
    you should be able to save summaries with config['train_params']['summary_step'] frequency
    tensorboard should show loss and accuracy for train and validation separately
    those in their respective folders defined in train
    """
    def __init__(self, log_dir):
        super(TensorboardCallback, self).__init__()
        self.log_dir = log_dir
        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
        self.val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'validation'))

    def on_epoch_end(self, epoch, logs=None):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', logs['loss'], step=epoch)
            tf.summary.scalar('pearsons_correlation', logs['pearsons_correlation'], step=epoch) #reg

        with self.val_summary_writer.as_default():
            tf.summary.scalar('val_loss', logs['val_loss'], step=epoch)
            tf.summary.scalar('val_pearsons_correlation', logs['val_pearsons_correlation'], step=epoch) #reg


class WeightsSaver(tf.keras.callbacks.Callback):
    def __init__(self, model, checkpoint_dir):
        super(WeightsSaver, self).__init__()
        self.model = model
        # self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.save_frequency = config["train_params"]['latest_checkpoint_step']
        self.max_to_keep = config["train_params"]['max_checkpoints_to_keep']
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.checkpoint_dir, 
            max_to_keep=self.max_to_keep,
            step_counter=tf.Variable(self.save_frequency , trainable=False))

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.save_frequency == 0:
            self.manager.save()