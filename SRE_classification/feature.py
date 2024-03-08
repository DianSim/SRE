from SRE.preprocessing_TIMIT import Preprocessing
import tensorflow as tf
from config import config
import matplotlib.pyplot as plt

# We suggest MFCC (Mel Frequency Cepstral Ceofficients) 

config_feature = config['feature']

class FeatureMappings:
    def __init__(self, preprocessing):
        self.preprocessing = preprocessing
        self.sample_rate = config['sample_rate']
        self.output_feature = config_feature
        # fft params
        window_size_ms = config_feature['window_size_ms']
        self.frame_length = int(self.sample_rate * window_size_ms)
        frame_step = config_feature['window_stride']
        self.frame_step = int(self.sample_rate * frame_step)
        assert (self.frame_step == self.sample_rate * frame_step), \
            'frame step,  must be integer '
        self.fft_length = config_feature['fft_length']
        # mfcc params
        self.lower_edge_hertz = config_feature['mfcc_lower_edge_hertz']
        self.upper_edge_hertz = config_feature['mfcc_upper_edge_hertz']
        self.num_mel_bins = config_feature['mfcc_num_mel_bins']
        self.linear_to_mel_weight_matrix = self.get_linear_to_mel_weight_matrix()

    def create_features(self):
       """
       This is the main function
       it gets a preprocessing object instance as an argument and therefore has access to preprocessing.dataset
        add another mapping` we suggest MFCC, but you are free to use other feature, the other most common one is log mel spectrogram
       """
       self.preprocessing.create_iterators()
       train_dataset = self.preprocessing.train_dataset.map(self.map_input_to_mfcc)
       val_dataset = self.preprocessing.val_dataset.map(self.map_input_to_mfcc)
       test_dataset = self.preprocessing.test_dataset.map(self.map_input_to_mfcc)
       
       return train_dataset, val_dataset, test_dataset
         
    def get_linear_to_mel_weight_matrix(self):
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=self.fft_length//2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.lower_edge_hertz,
            upper_edge_hertz=self.upper_edge_hertz)
        return linear_to_mel_weight_matrix

    def get_stft(self, audio):
        stft= tf.signal.stft(audio, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.fft_length)
        return stft

    def stft_to_log_mel_spectrogram(self, stft):
        magnitude_spectrogram = tf.abs(stft)
        mel_spectrogram = tf.tensordot(magnitude_spectrogram, self.linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(magnitude_spectrogram.shape[:-1].concatenate(self.linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        return log_mel_spectrogram

    def map_input_to_mfcc(self, audio, label):  # (audio,label)
        log_mel_spectrogram = self.stft_to_log_mel_spectrogram(self.get_stft(audio))
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = tf.expand_dims(mfccs, axis=-1)
        return [mfccs, label]
