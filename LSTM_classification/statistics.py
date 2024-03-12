import matplotlib.pyplot as plt
import os
import tqdm
import tensorflow as tf
from collections import Counter


def stat(arr, xlabel, title, image_name):
    plt.figure()
    plt.hist(arr, bins=int(max(arr))-int(min(arr))+1) # , bins=10, color='blue', edgecolor='black')
    # Add labels and title
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(image_name)


def class_distr(dataset, prefix, folder):
    """plots barplot of dataset(unbatched) classes and saves as a picture in the given folder"""
    sp_rates = [x[1].numpy() for x in dataset]
    value_counts = Counter(sp_rates)
    keys = value_counts.keys()
    values =  value_counts.values()
    plt.figure()
    plt.bar(keys, values)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(f"Classes distribution")
    plt.xticks(ticks=range(len(keys)), labels=sorted(keys))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'{prefix}_class_distr.png'))
    print(value_counts)
    

def compute_stat(dataset, prefix, folder):
    """"Computes sample rate distribution
    and chunk length distribution for the given dataset
    and saves as a png image
    
    dataset: tf Dataset object(unbatched)
    prefix: prefix to set before label name and saved image name: train, test or val
    folder: the folder to save histograms"""
    lens = [x[0].shape[0]/16000 for x in dataset]
    sp_rates = [x[1].numpy() for x in dataset]
    stat(lens, 'audio length', prefix+' data chunk lengths distribution', os.path.join(folder, f'{prefix}_len_distr.png'))
    stat(sp_rates, 'syl_count', prefix + ' data syllable count distribution', os.path.join(folder,f'{prefix}_sp_rate_distr.png'))


def upper_bound(corpus):
    """for the given corpus of datatstes computes max length of audio
    and max speaking rate:
    corpus: a list of datasets"""
    
    batch_max_len = []
    batch_max_sr = []
    for data in tqdm.tqdm(corpus):
        for x in data:
            batch_max_len.append(x[0].shape[1])
            batch_max_sr.append(tf.reduce_max(x[1]))
    return {'max_len:': max(batch_max_len), 'max_sr': max(batch_max_sr)}