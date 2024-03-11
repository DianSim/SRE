config ={
    'model_name': 'MatchBox_3x2x64',
    'sample_rate': 16000,
    'input_len': 33601, 
    'train_params': {
        'batch_size': 128,
        'epochs':200,
        'steps_per_epoch': None,
        'latest_checkpoint_step': 1,
        'summary_step': 50, 
        'max_checkpoints_to_keep': 5,
    },
    'feature': {
        'window_size_ms': 25, 
        'window_stride': 15,
        'fft_length': 512,
        'mfcc_lower_edge_hertz': 0.0,
        'mfcc_upper_edge_hertz': 8000.0,  
        'mfcc_num_mel_bins': 64
    }
}