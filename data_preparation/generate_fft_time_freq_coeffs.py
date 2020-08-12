import os
import sys
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())

import platform
import argparse

import dill as pickle
import collections
from preprocess.preprocessing_library import FFTWithTimeFreqCorrelation
from utils.pipeline import Pipeline
import numpy as np
from joblib import Parallel, delayed
import warnings

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

def convert_to_fft(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency, file_path):

    warnings.filterwarnings("ignore")
    type_data = pickle.load(open(file_path, 'rb'))
    pipeline = Pipeline([FFTWithTimeFreqCorrelation(fft_min_freq, fft_max_freq, sampling_frequency, 'first_axis')])
    time_series_data = type_data.data
    start, step = 0, int(np.floor(window_step * sampling_frequency))
    stop = start + int(np.floor(window_length * sampling_frequency))
    fft_data = []

    while stop < time_series_data.shape[1]:
        signal_window = time_series_data[:, start:stop]
        fft_window = pipeline.apply(signal_window)
        fft_data.append(fft_window)
        start, stop = start + step, stop + step

    fft_data = np.array(fft_data)
    named_data = seizure_type_data(patient_id=type_data.patient_id, seizure_type=type_data.seizure_type, data=fft_data)

    return named_data, os.path.basename(file_path)

def main():
    parser = argparse.ArgumentParser(description='Generate FFT time&freq coefficients from seizure data')


    if platform.system() == 'Linux':
        parser.add_argument('-l','--save_data_dir', default='/slow1/out_datasets/tuh/seizure_type_classification/',
                            help='path from resampled seizure data')
        parser.add_argument('-b','--base_save_data_dir', default='/fast1/out_datasets/tuh/seizure_type_classification/',
                            help='path to processed data')
    elif platform.system() == 'Darwin':
        parser.add_argument('-l','--save_data_dir', default='/Users/jbtang/datasets/TUH/eeg_seizure/',
                            help='path from resampled seizure data')
        parser.add_argument('-b','--preprocess_data_dir',
                            default='/Users/jbtang/datasets/TUH/output/seizures_type_classification/',
                            help='path to processed data')
    else:
        print('Unknown OS platform %s' % platform.system())
        exit()

    parser.add_argument('-v', '--tuh_eeg_szr_ver',
                        default='v1.4.0',
                        help='path to output prediction')

    args = parser.parse_args()
    tuh_eeg_szr_ver = args.tuh_eeg_szr_ver
    save_data_dir = os.path.join(args.save_data_dir,tuh_eeg_szr_ver,'raw_seizures')
    preprocess_data_dir = os.path.join(args.preprocess_data_dir,tuh_eeg_szr_ver,'fft_with_time_freq_corr')

    fnames = []
    for (dirpath, dirnames, filenames) in os.walk(save_data_dir):
        fnames.extend(filenames)

    fpaths = [os.path.join(save_data_dir,f) for f in fnames]

    sampling_frequency = 250  # Hz
    fft_min_freq = 1  # Hz

    window_lengths = [1, 2, 4, 8, 16]#[0.25, 0.5, 1]#[1, 2, 4, 8, 16]
    fft_max_freqs = [12, 24, 48, 64, 96]#[12, 24]

    for window_length in window_lengths:
        window_steps = list(np.arange(window_length/4, window_length/2 + window_length/4, window_length/4))
        #window_steps = list(np.arange(window_length / 8, window_length / 2 + window_length / 8, window_length / 8))
        for window_step in window_steps:
            for fft_max_freq_actual in fft_max_freqs:
                fft_max_freq = fft_max_freq_actual * window_length
                fft_max_freq = int(np.floor(fft_max_freq))
                print('window length: ', window_length, 'window step: ', window_step, 'fft_max_freq', fft_max_freq)
                save_data_dir = os.path.join(preprocess_data_dir, 'fft_seizures_' + 'wl' + str(window_length) + '_ws_' + str(window_step) + '_sf_' + \
                                str(sampling_frequency) + '_fft_min_' + str(fft_min_freq) + '_fft_max_' + \
                                str(fft_max_freq_actual))
                if not os.path.exists(save_data_dir):
                    os.makedirs(save_data_dir)
                else:
                    exit('Pre-processed data already exists!')

                '''
                converted_data = Parallel(n_jobs=50)(delayed(convert_to_fft)(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency, file_path=item) for item in fpaths)
                count = 0
                for item in converted_data:
                    if item.data.ndim == 2:
                        pickle.dump(item, open(save_data_dir + 'fft_tf_corr_seiz_' + str(count) + '.pkl', 'wb'))
                        count += 1
                '''

                for file_path in sorted(fpaths):
                    converted_data,file_name_base = convert_to_fft(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency,file_path)
                    if converted_data.data.ndim == 2:
                        pickle.dump(converted_data, open(os.path.join(save_data_dir, file_name_base), 'wb'))

if __name__ == '__main__':
    main()


