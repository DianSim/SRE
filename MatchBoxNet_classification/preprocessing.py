import os
from random import shuffle
import tensorflow as tf
import glob
from config import config
import soundfile as sf



class Preprocessing:
    def __init__(self):
        print('preprocessing instance creation started')

        self.dir_val_clean = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeechChuncked_v2/dev-clean'
        self.dir_test_clean = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeechChuncked_v2/test-clean'
        self.dir_train_clean = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeechChuncked_v2/train-clean-100'
        self.input_len = config['input_len']
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def create_iterators(self):
        val_clean_files = glob.glob(os.path.join(self.dir_val_clean, '**/*.wav'), recursive=True)
        test_clean_files = glob.glob(os.path.join(self.dir_test_clean, '**/*.wav'), recursive=True)
        train_clean_files = glob.glob(os.path.join(self.dir_train_clean, '**/*.wav'), recursive=True)
        # shuffle(train_clean_files)
        print('len(train_data)', len(train_clean_files))
        print('len(test_data)', len(test_clean_files))
        print('len(val_data)', len(val_clean_files))

        # make tf dataset object
        self.train_dataset = self.make_tf_dataset_from_list(train_clean_files)
        self.val_dataset = self.make_tf_dataset_from_list(val_clean_files, is_validation = True)
        self.test_dataset = self.make_tf_dataset_from_list(test_clean_files, is_validation = True)
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_label(self, file_path):
        filename_wav= tf.strings.split(file_path, os.path.sep)[-1]
        filename = tf.strings.split(filename_wav, '.')[0]
        label = tf.strings.split(filename, '_')[0]
        label = tf.strings.to_number(label, tf.int32)
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
        dataset = dataset.map(self.map_get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.map_add_padding, num_parallel_calls=tf.data.AUTOTUNE)
        if not is_validation:
            dataset = dataset.shuffle(buffer_size=20000)
        dataset = dataset.batch(config['train_params']['batch_size'])
        if not is_validation:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
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
    
    def map_add_padding(self, audio, label):
        return (self.add_paddings(audio), label)
         
    def add_paddings(self, wav):
        padded_wf = tf.concat([wav, tf.zeros(self.input_len - tf.shape(wav))], 0)
        return padded_wf


if __name__ == '__main__':
    p = Preprocessing()
    train, val, test = p.create_iterators()
    for x in train:
        print(x[0].shape)
        print(x[1])