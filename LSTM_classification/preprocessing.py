import os
from random import shuffle
import tensorflow as tf
import glob
from config import config
import soundfile as sf
from scipy.io.wavfile import write
from tensorflow.keras import layers
import numpy as np


# tf.keras.utils.set_random_seed(1)
# tf.config.experimental.enable_op_determinism()


class Preprocessing:
    def __init__(self):
        print('preprocessing instance creation started')

        self.dir_val_clean = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeechChuncked_v2/dev-clean'
        self.dir_test_clean = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeechChuncked_v2/test-clean'
        self.dir_train_clean = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeechChuncked_v2/train-clean-100'
        self.dir_background_noise = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/backround_noises_ESC50/ESC_50_16khz'
        self.prob = 0.5 # probability of noise augmentation
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.background_noise_files = None

        
    def create_iterators(self):
        val_clean_files = glob.glob(os.path.join(self.dir_val_clean, '**/*.wav'), recursive=True)
        test_clean_files = glob.glob(os.path.join(self.dir_test_clean, '**/*.wav'), recursive=True)
        train_clean_files = sorted(glob.glob(os.path.join(self.dir_train_clean, '**/*.wav'), recursive=True))
        self.background_noise_files = glob.glob(os.path.join(self.dir_background_noise, '**/*.wav'), recursive=True)
        # shuffle(train_clean_files)
        print('len(train_data)', len(train_clean_files))
        print('len(test_data)', len(test_clean_files))
        print('len(val_data)', len(val_clean_files))
        
        # make tf dataset object
        self.train_dataset = self.make_tf_dataset_from_list(train_clean_files)
        self.val_dataset = self.make_tf_dataset_from_list(val_clean_files, is_validation = True)
        self.test_dataset = self.make_tf_dataset_from_list(test_clean_files, is_validation = True)
        return self.train_dataset, self.val_dataset, self.test_dataset

    def make_tf_dataset_from_list(self, filenames_list, is_validation = False):
        """
        ARGS:
            filenames_list is a list of file_paths
            is_validation is a boolean which should be true when makeing val_dataset
        """   
        dataset = tf.data.Dataset.from_tensor_slices(filenames_list)
        dataset = dataset.map(self.map_get_waveform_and_label)
        dataset = dataset.cache()
        if not is_validation:
            dataset = dataset.shuffle(buffer_size=len(filenames_list)//2).map(self.noise_augmentation)
        dataset = dataset.map(self.map_normalize)
        dataset = dataset.padded_batch(batch_size=config['train_params']['batch_size'], 
                                       padded_shapes=([None], [])).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.map(lambda batch_wav, batch_label: (self.chunking(batch_wav), batch_label))
        return dataset
    
    def noise_augmentation(self, signal, label):
        noisy_signal = tf.py_function(func=self.add_noise, inp=[signal], Tout=tf.float32)
        return noisy_signal, label

    def add_noise(self, signal):
        if np.random.uniform() < self.prob:
            i_noise = np.random.randint(0, high=len(self.background_noise_files))
            noise_path = self.background_noise_files[i_noise]

            noise, sample_rate = sf.read(noise_path, dtype='float32')

            signal_len = signal.shape[0] 
            noise_len = noise.shape[0]

            if signal_len < noise_len:
                noise = noise[:signal_len]

            snr_db = np.random.randint(0, high=20) 
            noisy_signal = Preprocessing.mix(signal, noise, snr_db)
            return noisy_signal
        else:
            return signal
        
    
    def get_label(self, file_path):
        filename_wav= tf.strings.split(file_path, os.path.sep)[-1]
        filename = tf.strings.split(filename_wav, '.')[0]
        label = tf.strings.split(filename, '_')[0]
        label = tf.strings.to_number(label, tf.int32)
        return label
    
    def map_get_waveform_and_label(self, file_path):
        """
        Map function
        for every filepath return its waveform (use only tensorflow) and label 
        """
        label = self.get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform, sample_rate = tf.audio.decode_wav(audio_binary)
        waveform = tf.squeeze(waveform, axis=-1)
        return waveform, label 
    
    def map_normalize(self, wav, label):
        mean = tf.math.reduce_mean(wav)
        std = tf.math.reduce_std(wav)
        normalized_wav = (wav - mean)/std
        return normalized_wav, label

    def chunking(self, wav_batch):
        return tf.signal.frame(wav_batch, frame_length=int(config['frame_length']*config['sample_rate']/1000), frame_step=int(config['window_shift']*config['sample_rate']/1000), pad_end=True)
    
    @staticmethod
    def mix(signal, noise, snr_db):
        snr = 10**(snr_db/10)
        Es = np.sum(signal**2)
        En = np.sum(noise**2)
        alpha = np.sqrt(Es/(snr*En+10e-5))
        return signal + alpha*noise
    
    @staticmethod
    def read(file_path):
        audio_binary = tf.io.read_file(file_path)
        waveform, sample_rate = tf.audio.decode_wav(audio_binary)
        waveform = tf.squeeze(waveform, axis=-1)  
        return waveform

    # def noise_augmentation(self, signal):
    #     if  tf.random.uniform([]) < self.prob:
    #         i_noise = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(self.background_noise_files)[0], dtype=tf.int32)
    #         noise_path = self.background_noise_files[tf.cast(i_noise, tf.int32)]
    #         noise = Preprocessing.read(noise_path)

    #         signal_len = tf.shape(signal)[0] 
    #         noise_len = tf.shape(noise)[0]

    #         if signal_len < noise_len:
    #             noise = noise[:signal_len]

    #         snr_db = tf.random.uniform(shape=[], minval=0, maxval=20, dtype=tf.int32) # generate from a range
    #         snr_db = tf.cast(snr_db, tf.float32)
    #         noisy_signal = Preprocessing.mix(signal, noise, snr_db)
    #     else:
    #         return signal, label
    #     return noisy_signal, label



if __name__ == '__main__':
    p = Preprocessing()