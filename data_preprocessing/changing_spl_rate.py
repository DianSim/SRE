import os
import subprocess
import tqdm

def change_sample_rate(dir, sample_rate, out_dir_name):
    for root, dirs, files in os.walk(dir):
        out_dir = os.sep.join(root.split(os.sep)[:-1]+[out_dir_name])
        os.makedirs(out_dir, exist_ok=True)
        for file in tqdm.tqdm(files):
            command = f"ffmpeg -i '{os.path.join(root, file)}' -ar 16000 '{os.path.join(out_dir, file)}'"
            result = subprocess.run(['ffmpeg', '-i', f'{os.path.join(root, file)}', '-ar', '16000', f'{os.path.join(out_dir, file)}'], capture_output=True)


source = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/backround_noises_ESC50/audio'
change_sample_rate(source, 16000, 'ESC_50_16khz')