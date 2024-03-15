import os
import soundfile as sf
import numpy as np
import pandas as pd


def Speed(dir, save_dir_name):
    for root, dirs, files in os.walk(dir):
        word_trans_file = None
        for file in files:
            if file[-14:] == '.alignment.txt':
                word_trans_file = os.path.join(root, file)

        if word_trans_file is None: # means there're no audios in the folder
            continue

        df_word_trans = pd.read_csv(word_trans_file, delimiter='\t', header=None)
        for file in files:
            if file[-5:] == '.flac': 
                signal, sr = sf.read(os.path.join(root, file))
                speed_factor = np.random.uniform(1.5, 2)
                save_dir = root.replace('LibriSpeech', save_dir_name)
                os.makedirs(save_dir, exist_ok=True)
                sf.write(os.path.join(save_dir, f'sp_factor_{speed_factor}.wav'), signal, int(sr*speed_factor))

                print('file:', os.path.join(save_dir, f'sp_factor_{speed_factor}.wav'))
                for row in df_word_trans[0]:
                    if row.split(' ')[0] == file.split('.')[0]:
                        words_str = row.split(' ')[1]
                        words_str = words_str.replace('"', '')
                        words = words_str.split(',')
                        print(f'transcript: {words}')
                        print()

                

dir = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeech/train-clean-100'
Speed(dir, 'LibriSpeech_Fast')