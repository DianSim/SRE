import tensorflow as tf
from scipy.io.wavfile import write



def read(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform, sample_rate = tf.audio.decode_wav(audio_binary)
    waveform = tf.squeeze(waveform, axis=-1)  
    return waveform


def mix(signal, noise, snr_db):
    """
    Args:
    snr_db: snr in dbs
    """

    snr = 10**(snr_db/10)
    Es = tf.reduce_sum(tf.square(signal))
    En = tf.reduce_sum(tf.square(noise))
    alpha = tf.sqrt(Es/(snr*En))
    # print(alpha)
    return signal + alpha*noise


sig_path = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeechChuncked_v2/dev-clean/84/121123/15_4.wav'
nois_path = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/backround_noises_ESC50/ESC_50_16khz/1-5996-A-6.wav'

signal = read(sig_path)
noise = read(nois_path)[:tf.shape(signal)[0]]
# print(signal)
# print(noise)
noisy_signal = mix(signal, noise, 10)
write('./noisy10.wav', 16000, noisy_signal.numpy())


# def noise_augmentation(inputs, factor=0.5):
#     augmented_batch = tf.TensorArray(size=tf.shape(inputs)[0], dtype=tf.float32)
#     for i, x in enumerate(inputs):
#         if  tf.random.uniform([]) < factor:
#             # get random noise index
#             # listdir and get noise path
#             noise_path = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/download.wav'
#             noise = Preprocessing.read(noise_path)

#             signal_len = tf.shape(x)[0] 
#             noise_len = tf.shape(noise)[0]

#             if signal_len < noise_len:
#                 noise = noise[16000:16000+signal_len]

#             snr_db = 7 # generate from a range
#             noisy_signal = Preprocessing.mix(x, noise, snr_db)
#             augmented_batch.write(i, noisy_signal)
#         else:
#             augmented_batch.write(i, x)
#     return augmented_batch