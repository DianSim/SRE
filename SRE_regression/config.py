config ={
    'model_name': 'LSTM64_dense1',
    'sample_rate': 16000,
    'frame_length': 35,
    'window_shift': 10,
    'train_params': {
        'batch_size': 16,
        'epochs':2,
        'steps_per_epoch': None,
        'latest_checkpoint_step': 1,
        'summary_step': 50, 
        'max_checkpoints_to_keep': 5,
    },
    'model_params':{
        'f_n': 8,
        'f_l': 30 # ms
    },
    'feature': {
        'window_size_ms': 0.020, 
        'window_stride': 0.01,
        'fft_length': 512,
        'mfcc_lower_edge_hertz': 0.0,
        'mfcc_upper_edge_hertz': 8000.0,  
        'mfcc_num_mel_bins': 13
    },
    'data_proc':{
        'silence': 30
    }
}