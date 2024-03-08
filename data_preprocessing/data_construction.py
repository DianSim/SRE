import os
import soundfile as sf
import pandas as pd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import math
import numpy as np
# from config import config

"""Builder of data version 1"""

def speaking_rate(transcript):
    """The function computes syllable based speaking rate 
    of an audio with given transcript
    transcript(str): transcript of the audio
    """
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    vowel_count = 0
    for s in transcript:
        if s.lower() in vowels:
            vowel_count += 1
    # return vowel_count/length
    return vowel_count


def chuncking_labeling(dir, chunck_length):
    """"
    Splits the audios of the given dir into chunks, computes the labels 
    and saves in a folder 'LibriSpeechChuncked' with the same structure as dir
    setting lables as audio titles

    chunck_length: chunk length with seconds
    """
    
    for root, dirs, files in os.walk(dir):
        word_trans_file = None
        for file in files:
            if file[-14:] == '.alignment.txt':
                word_trans_file = os.path.join(root, file)
        if word_trans_file is None:
            continue
        
        folder_file_title_count = {}
        df_word_trans = pd.read_csv(word_trans_file, delimiter='\t', header=None)
        for file in files:
            if file[-5:] == '.flac':
                signal, sr = sf.read(os.path.join(root, file))
                for row in df_word_trans[0]:
                    if row.split(' ')[0] == file.split('.')[0]:
                        words_str = row.split(' ')[1]
                        words_str = words_str.replace('"', '')
                        tm_stemps_str = row.split(' ')[2]
                        tm_stemps_str = tm_stemps_str.replace('"', '')
                        words = words_str.split(',')
                        tm_stemps = tm_stemps_str.split(',')
                        tm_stemps = [float(t) for t in tm_stemps]

                        print(words)
                        print(tm_stemps)
                        print()
                        print('-'*60+'chunking'+'-'*60)

                        i_begin = 0
                        while i_begin < len(words):
                            i_end = i_begin
                            chunck_begin =  tm_stemps[i_begin-1] if i_begin else 0
                            chunck_end =chunck_begin + chunck_length

                            while i_end < len(words) and tm_stemps[i_end] <= chunck_end:
                                i_end += 1
                            if i_end != i_begin:
                                i_end -= 1 # for i_end the condition deosn't hold

                            # test whether to add next word 
                            # because of this very long chunks arose 4 - 6 second
                            if (i_end + 1 < len(words)) and  (chunck_end - tm_stemps[i_end]) > (tm_stemps[i_end+1] - chunck_end):
                                i_end += 1

                            chunk_words = words[i_begin:i_end+1] 

                            # remove silence chunks
                            if len(chunk_words) == 1 and chunk_words[0] == '':
                                # pass to the next chunck
                                i_begin = i_end + 1
                                continue

                            transcript = ' '.join(chunk_words)
                            exact_chnk_len = float(tm_stemps[i_end]) - chunck_begin
                            print('transcripts: ', transcript)
                            print('length: ', exact_chnk_len)
                            speak_rate = speaking_rate(transcript)
                            print('sr:', speak_rate)
                    
                            # remove silence at the beggining of the chunck
                            if chunk_words[0] == '':
                                chunck_begin = tm_stemps[i_begin]

                            # remove silence at the end of chunck
                            if chunk_words[-1] == '':
                                i_end -= 1

                            # extract audio chunck with given timestemp [i_begin, i_end]
                            # the end of previous word is the beginning of the current
                            x = signal[math.floor(chunck_begin*sr): round(float(tm_stemps[i_end])*sr) + 1]

                            # pad silence in the beginning
                            silence_len = int(0.02*16000) # should be determined somehow
                            x = np.pad(x, (silence_len, 0), 'constant', constant_values=0)
                            save_dir = root.replace('LibriSpeech', 'LibriSpeechChuncked')

                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)

                            filename = str(speak_rate)
                            if filename not in folder_file_title_count:
                                folder_file_title_count[filename] = 0
                            else:
                                folder_file_title_count[filename] += 1
                                filename += f'_{folder_file_title_count[filename]}'
                            filename += '.wav'

                            # if x.shape[0] > (chunck_length+0.5)*16000:
                            #     # print(words)
                            #     # print(tm_stemps)
                            #     # print()
                            #     # print('-'*60+'chunking'+'-'*60)
                            #     # print('chunck_begin: ', chunck_begin)
                            #     # print("chunck end:", chunck_end)
                            #     # print('transcripts: ', transcript)
                            #     # print('length: ', exact_chnk_len)
                            #     # print(x.shape)
                            #     # print()
                            #     sf.write(os.path.join('/Users/dianasimonyan/Desktop/ASDS/Thesis/Implementation/datasets/LibriSpeechChuncked/long-chunks', filename), x, sr)
                            # else:
                            sf.write(os.path.join(save_dir, filename), x, sr)
                            print(os.path.join(save_dir, filename))
                            print()
                            
                            i_begin = i_end + 1


if __name__ == '__main__':
    dir = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeech/train-clean-100'
    chuncking_labeling(dir, 2)