# The dataset for this task is Goole Speech Commandc v2
# You can download it at http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

import os
from random import shuffle
import tensorflow as tf
import glob
from config import config
#dont import other libraries


class_names = ['right', 'eight', 'cat', 'tree', 'backward', 'learn', 'bed', 'happy', 'go', 'dog', 'no', 'wow', 'follow', 'nine', 'left', 'stop', 'three', 'sheila', 'one', 'bird', 'zero', 'seven', 'up', 'visual', 'marvin', 'two', 'house', 'down', 'six', 'yes', 'on', 'five', 'forward', 'off', 'four']
label_to_int = {c: num for c, num in zip(class_names, tf.range(len(class_names)))}

# Convert the dictionary to two separate lists: labels and numbers
class_labels, class_numbers = zip(*label_to_int.items())

# Create a TensorFlow StaticHashTable
table_initializer = tf.lookup.KeyValueTensorInitializer(class_labels, class_numbers, key_dtype=tf.string, value_dtype=tf.int32)
class_table = tf.lookup.StaticHashTable(table_initializer, default_value=-1)

# Function to map categorical labels to integers using the hash table
def map_labels_to_integers(wav, label):
    num_label = class_table.lookup(label)
    return (wav, num_label)


class Preprocessing:
    def __init__(self):
        print('preprocessing instance creation started')

        self.dir_name = config['data_dir']
        self.input_len = config['input_len']

        # optional/bonus points: add later noise augmentation
        
    def create_iterators(self):
        # get the filenames split into train test validation
        test_files = self.get_files_from_txt('testing_list.txt')
        val_files = self.get_files_from_txt('validation_list.txt')
        filenames = glob.glob(os.path.join(self.dir_name, '*/**.wav'), recursive=True)
        filenames = [filename for filename in filenames if 'background_noise' not in filename]
        train_files = list(set(filenames) - set(val_files) - set(test_files)) #from all files subtract test and validation
        shuffle(train_files)
        # get the commands and some prints
        self.commands = self.get_commands()
        self.num_classes = len(self.commands)
        print('len(train_data)', len(train_files))
        print('len(test_data)', len(test_files))
        print('len(val_data)', len(val_files))
        print('commands: ', self.commands)
        print('number of commands: ', len(self.commands))

        # make tf dataset object
        self.train_dataset = self.make_tf_dataset_from_list(train_files)
        self.val_dataset = self.make_tf_dataset_from_list(val_files, is_validation = True)
        self.test_dataset = self.make_tf_dataset_from_list(test_files)


    def get_files_from_txt(self, which_txt):
        """
         There are testing_list and validation_list txts and you should use those to get the the train_test_validation split
         this function must get the argument and return the paths of (for example validation) datapoints paths as a list
         you only need importet libraries
         dont forget to shuffle
        """
        assert which_txt == 'testing_list.txt' or which_txt == 'validation_list.txt', 'wrong argument'

        paths = []
        with open(os.path.join(self.dir_name, which_txt), "r") as file:
            for line in file:
                paths.append(os.path.join(self.dir_name, line.strip()))
        shuffle(paths)
        return paths

    def get_commands(self):
        dirs = glob.glob(os.path.join(self.dir_name, "*", ""))
        commands = [os.path.split(os.path.split(dir)[0])[1] for dir in dirs if 'background' not in dir]
        return commands

    def get_label(self, file_path):
        label = tf.strings.split(file_path, os.path.sep)[-2]

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
        dataset = dataset.map(self.map_add_padding)
        # dataset = dataset.map(lambda wav, label: (wav, map_labels_to_integers(label)))
        dataset = dataset.map(map_labels_to_integers)
        dataset = dataset.cache()
        if not is_validation:
            dataset = dataset.shuffle(buffer_size=len(filenames_list))
        dataset = dataset.batch(batch_size=config['train_params']['batch_size']).prefetch(tf.data.AUTOTUNE)
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
        return waveform, label

    def map_add_padding(self, audio, label):
        return [self.add_paddings(audio), label]

    def add_paddings(self, wav):
        """
        all the data should be 2 seconds (16000 points)
        pad with zeros to make every wavs lenght 16000 if needed.
        """
        zero_padding = tf.zeros(self.input_len - tf.shape(wav)[0], dtype=tf.float32)
        wav = tf.concat([wav, zero_padding], 0)
        return wav


# if __name__ == '__main__':
#     p = Preprocessing()
#     print(p.get_files_from_txt('testing_list.txt'))