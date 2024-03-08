import os
from random import shuffle
import tensorflow as tf
import glob
from config import config
import statistics as stat


class Preprocessing:
    def __init__(self):
        print('preprocessing instance creation started')

        self.dir_val_clean = '/Users/dianasimonyan/Desktop/ASDS/Thesis/Implementation/datasets/LibriSpeechChuncked/dev-clean'
        # self.dir_val_other = '/Users/dianasimonyan/Desktop/ASDS/Thesis/Implementation/datasets/LibriSpeechChuncked/dev-other'
        self.dir_test_clean = '/Users/dianasimonyan/Desktop/ASDS/Thesis/Implementation/datasets/LibriSpeechChuncked/test-clean'
        # self.dir_test_other = '/Users/dianasimonyan/Desktop/ASDS/Thesis/Implementation/datasets/LibriSpeechChuncked/test-other'
        self.dir_train_clean = '/Users/dianasimonyan/Desktop/ASDS/Thesis/Implementation/datasets/LibriSpeechChuncked/train-clean-100'
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def create_iterators(self):
        val_clean_files = sorted(glob.glob(os.path.join(self.dir_val_clean, '**/*.wav'), recursive=True))
        test_clean_files = sorted(glob.glob(os.path.join(self.dir_test_clean, '**/*.wav'), recursive=True))
        train_clean_files = sorted(glob.glob(os.path.join(self.dir_train_clean, '**/*.wav'), recursive=True))[:100000]
        shuffle(train_clean_files)
        print('len(train_data)', len(train_clean_files))
        print('len(test_data)', len(test_clean_files))
        print('len(val_data)', len(val_clean_files))

        # make tf dataset object
        self.train_dataset = self.make_tf_dataset_from_list(train_clean_files)
        self.val_dataset = self.make_tf_dataset_from_list(val_clean_files, is_validation = True)
        self.test_dataset = self.make_tf_dataset_from_list(test_clean_files)
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_label(self, file_path):
        filename_wav= tf.strings.split(file_path, os.path.sep)[-1]
        filename = tf.strings.split(filename_wav, '.')[0]
        label = tf.strings.split(filename, '_')[0]
        label = tf.strings.to_number(label, tf.float32)
        return label

    def make_tf_dataset_from_list(self, filenames_list, is_validation = False):
        """
        ARGS:
            filenames_list is a list of file_paths
            is_validation is a boolean which should be true when makeing val_dataset

        Using the list create tf.data.Dataset object
        do necessary mappings (methods starting with 'map'),
        use prefetch, shuffle, batch methods
        bonus points` mix with background noise 
        """   
        dataset = tf.data.Dataset.from_tensor_slices(filenames_list)
        dataset = dataset.map(self.map_get_waveform_and_label)
        dataset = dataset.cache()
        if not is_validation:
            dataset = dataset.shuffle(buffer_size=len(filenames_list))
        dataset = dataset.padded_batch(batch_size=config['train_params']['batch_size'], 
                                       padded_shapes=([None], [])).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.map(lambda batch_wav, batch_label: (self.chunking(batch_wav), batch_label))
        return dataset

    def map_get_waveform_and_label(self, file_path):
        """
        Map function
        for every filepath return its waveform (use only tensorflow) and label 
        """
        label = self.get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform, sample_rate = tf.audio.decode_wav(audio_binary)
        waveform = tf.squeeze(waveform, axis=-1)
        waveform = waveform - tf.math.reduce_mean(waveform)
        waveform = waveform/tf.math.reduce_std(waveform)
        return waveform, label 

    def chunking(self, wav):
        return tf.signal.frame(wav, frame_length=int(config['frame_length']*config['sample_rate']/1000), frame_step=int(config['window_shift']*config['sample_rate']/1000), pad_end=True)


if __name__ == '__main__':
    p = Preprocessing()
    p.create_iterators()
