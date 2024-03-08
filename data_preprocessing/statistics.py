import os
import soundfile as sf
import matplotlib.pyplot as plt


def save(arr, xlabel, title, save_path):
    plt.figure()
    plt.hist(arr)#, bins=10) # bins=int(max(arr))-int(min(arr))+1) # ,, color='blue', edgecolor='black')
    # Add labels and title
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(save_path)


def stat(dir, save_dir, split):
    """Computes speaking rate distribution and chunck length distribution 
    of the audios in the given folder and saves in the 'save_dir' folder as png
    
    split: split name (train/val/test)"""

    os.makedirs(save_dir, exist_ok=True)
    lens = []
    spr = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file[-4:] == '.wav':
                signal, sr = sf.read(os.path.join(root, file))
                lens.append(len(signal)/sr)
                spr.append(int(file.split('.')[0].split('_')[0]))
    save(lens, 'chunck length', f'{split} set chunk length distribution', os.path.join(save_dir, f'{split}_length'))
    save(spr, 'speaking rate (#vowal)', f'{split} set speaking rate distribution', os.path.join(save_dir, f'{split}_sp_rate'))

    with open(os.path.join(save_dir,f'{split}.txt'), 'w') as file:
        # Write some text into the file
        file.write(f"max speaking rate({split}): {max(spr)}")



if __name__ == '__main__':
    save_dir = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/Statitics/clean_LB_Speech_stat_v2'
    val_dir = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeechChuncked_v2/dev-clean'
    test_dir = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeechChuncked_v2/test-clean'
    train_dir = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeechChuncked_v2/train-clean-100'

    stat(val_dir, save_dir, 'val')
    stat(test_dir, save_dir, 'test')
    stat(train_dir, save_dir, 'train')