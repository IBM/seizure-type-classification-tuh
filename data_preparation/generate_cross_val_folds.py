import os
import sys
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())

from sklearn.model_selection import KFold,StratifiedKFold
import pickle
import collections
import numpy as np
import platform
import argparse

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

def generate_seizure_wise_cv_folds(data_dir, num_split):

    seizures_by_type = collections.defaultdict(list)
    total_szr_num = 0
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            sz = pickle.load(open(os.path.join(data_dir,fname), 'rb'))
            if sz.seizure_type != 'MYSZ':
                seizures_by_type[sz.seizure_type].append(fname)
                total_szr_num += 1
    print('Total found Seizure Num =%d' % (total_szr_num))

    # Delete mycolnic seizures since there are only three of them
    # del seizures_by_type['MYSZ']
    kf = KFold(n_splits=num_split)

    cv_split = {}
    for i in range(1, num_split+1):
        cv_split[str(i)] = {'train': [], 'val': []}

    for type, fname_list in seizures_by_type.items():
        fname_list = np.array(fname_list)
        for index, (train_index, val_index) in enumerate(kf.split(fname_list), start=1):
            cv_split[str(index)]['train'].extend(fname_list[train_index])
            cv_split[str(index)]['val'].extend(fname_list[val_index])

    # check allocated szr num
    for i in range(1, num_split + 1):
        print('Fold %d Train Seizure Number = %d'%(i,len(cv_split[str(i)]['train'])))
        print('Fold %d Val Seizure Number = %d' %(i,len(cv_split[str(i)]['val'])))

    return cv_split

def generate_patient_wise_cv_folds(data_dir, num_split):

    #seizures_by_type = collections.defaultdict(list)
    szr_type_patient_list = collections.defaultdict(list)
    patient_file_list = collections.defaultdict(list)
    total_szr_num = 0
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            sz = pickle.load(open(os.path.join(data_dir,fname), 'rb'))
            # seizures_by_type[sz.seizure_type].append(fname)
            # Skip mycolnic seizures since there are only three of them
            if sz.seizure_type != 'MYSZ':
                szr_type_patient_list[sz.seizure_type].append(sz.patient_id)
                patient_file_list[sz.patient_id].append(fname)
                total_szr_num +=1

    print('Total found Seizure Num =%d'%(total_szr_num))
    szr_type_list = ['TNSZ', 'SPSZ', 'ABSZ', 'TCSZ', 'CPSZ', 'GNSZ', 'FNSZ']
    for type in szr_type_list:
        szr_type_patient_list[type] = np.unique(szr_type_patient_list[type])

    '''
    # try to get most balanced cross fold split
    min_diff = 99999
    for SEED in range(1,10000):
        kf = KFold(n_splits=num_split, random_state=SEED, shuffle=True)

        cv_split = {}
        for i in range(1, num_split+1):
            cv_split[str(i)] = {'train': [], 'val': []}

        allocated_patient_list = []
        for type in szr_type_list:
            all_patient_list = szr_type_patient_list[type]
            patient_list = np.setdiff1d(all_patient_list, allocated_patient_list)
            allocated_patient_list.extend(patient_list)
            for index, (train_index, val_index) in enumerate(kf.split(patient_list), start=1):
                train_patient_list = patient_list[train_index]
                val_patient_list = patient_list[val_index]

                for train_patient in train_patient_list:
                    cv_split[str(index)]['train'].extend(patient_file_list[train_patient])
                for val_patient in val_patient_list:
                    cv_split[str(index)]['val'].extend(patient_file_list[val_patient] )

        # check allocated szr num
        val_len = []
        for i in range(1, num_split+1):
            #print('Fold %d: Train Seizure Number = %d'%(i,len(cv_split[str(i)]['train'])))
            #print('Fold %d: Val Seizure Number = %d' %(i,len(cv_split[str(i)]['val'])))
            val_len.append(len(cv_split[str(i)]['val']))

        if min_diff>max(val_len)-min(val_len):
            min_diff = max(val_len)-min(val_len)
            min_seed = SEED
            print(min_diff,min_seed)
    
    '''
    min_seed = 2001 # found by the previous program
    kf = KFold(n_splits=num_split, random_state=min_seed, shuffle=True)

    cv_split = {}
    for i in range(1, num_split + 1):
        cv_split[str(i)] = {'train': [], 'val': []}

    allocated_patient_list = []
    for type in szr_type_list:
        all_patient_list = szr_type_patient_list[type]
        patient_list = np.setdiff1d(all_patient_list, allocated_patient_list)
        allocated_patient_list.extend(patient_list)
        for index, (train_index, val_index) in enumerate(kf.split(patient_list), start=1):
            train_patient_list = patient_list[train_index]
            val_patient_list = patient_list[val_index]

            for train_patient in train_patient_list:
                cv_split[str(index)]['train'].extend(patient_file_list[train_patient])
            for val_patient in val_patient_list:
                cv_split[str(index)]['val'].extend(patient_file_list[val_patient])

    # check allocated szr num
    for i in range(1, num_split + 1):
        print('Fold %d Train Seizure Number = %d'%(i,len(cv_split[str(i)]['train'])))
        print('Fold %d Val Seizure Number = %d' %(i,len(cv_split[str(i)]['val'])))

    return cv_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BCH Data Training')

    if platform.system() == 'Linux':
        parser.add_argument('--save_data_dir',
                            default='/slow1/out_datasets/tuh/seizure_type_classification/',
                            help='path to output prediction')
    elif platform.system() == 'Darwin':
        parser.add_argument('--save_data_dir',
                            default='/Users/jbtang/datasets/TUH/output/seizures_type_classification/fft/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_12',
                            help='path to output prediction')
    else:
        print('Unknown OS platform %s' % platform.system())
        exit()

    parser.add_argument('-v', '--tuh_eeg_szr_ver',
                        default='v1.5.2',
                        #default='v1.4.0',
                        help='version of TUH seizure dataset')

    args = parser.parse_args()
    tuh_eeg_szr_ver = args.tuh_eeg_szr_ver

    if tuh_eeg_szr_ver == 'v1.4.0': # for v1.4.0
        print('\nGenerating seizure wise cross validation for', tuh_eeg_szr_ver)
        fold_num = 5
        save_data_dir = os.path.join(args.save_data_dir, 'v1.4.0', 'raw_seizures')
        cv_random = generate_seizure_wise_cv_folds(save_data_dir, fold_num)
        pickle.dump(cv_random, open('cv_split_5_fold_seizure_wise_v1.4.0.pkl', 'wb'))

    elif tuh_eeg_szr_ver == 'v1.5.2': # for v1.5.2
        print('\nGenerating seizure wise cross validation for', tuh_eeg_szr_ver)
        fold_num = 5
        save_data_dir = os.path.join(args.save_data_dir, 'v1.5.2', 'raw_seizures')
        cv_random = generate_seizure_wise_cv_folds(save_data_dir, fold_num)
        pickle.dump(cv_random, open('cv_split_5_fold_seizure_wise_v1.5.2.pkl', 'wb'))

        print('\nGenerating seizure wise cross validation for', tuh_eeg_szr_ver)
        fold_num = 3
        save_data_dir = os.path.join(args.save_data_dir, 'v1.5.2', 'raw_seizures')
        cv_patient_wise = generate_patient_wise_cv_folds(save_data_dir, fold_num)
        pickle.dump(cv_patient_wise, open('cv_split_3_fold_patient_wise_v1.5.2.pkl', 'wb'))

    else:
        exit('Not supported version number %s'%tuh_eeg_szr_ver)












