import os,mne,pickle,torch
import numpy as np
from sklearn.utils import shuffle
from collections import Counter
import numpy as np
from sklearn.utils import shuffle

seed = 0
re_sfreq= 100
tmin = -0.15
tmax = 1.0
n_ses = 4
mvnn_dim = 'epochs'

project_dir = '/root/workspace/wht/multimodal_brain/datasets/things-eeg-small'
chan_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
				  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
				  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
				  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
				  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
				  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
				  'O1', 'Oz', 'O2']

def epoch_data(mode,sub):
    epoched_data = []
    img_conditions = []
    for s in range(n_ses):
        ### Load the EEG data and convert it to MNE raw format ###
        eeg_dir = os.path.join('Raw_data', 'sub-'+
            format(sub,'02'), 'ses-'+format(s+1,'02'), f"raw_eeg_{mode}.npy")
        eeg_data = np.load(os.path.join(project_dir, eeg_dir),
            allow_pickle=True).item()
        ch_names = eeg_data['ch_names']
        sfreq = eeg_data['sfreq']
        ch_types = eeg_data['ch_types']
        eeg_data = eeg_data['raw_eeg_data']
        # Convert to MNE raw format
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(eeg_data, info)

        ### Get events, drop unused channels and reject target trials ###
        events = mne.find_events(raw, stim_channel='stim')
        # # Select only occipital (O) and posterior (P) channels
        # chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'],
        # 	'^O *|^P *'))
        # new_chans = [raw.info['ch_names'][c] for c in chan_idx]
        # raw.pick_channels(new_chans)
        # * chose all channels
        raw.pick_channels(chan_order, ordered=True)
        # Reject the target trials (event 99999)
        idx_target = np.where(events[:,2] == 99999)[0]
        events = np.delete(events, idx_target, 0)
        ### Epoching, baseline correction and resampling ###
        # * [0, 1.0]
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=(None,0),
            preload=True)
        # Resampling
        if re_sfreq < 1000:
            epochs.resample(re_sfreq)
        ch_names = epochs.info['ch_names']
        times = epochs.times

        ### Sort the data ###
        data = epochs.get_data()
        events = epochs.events[:,2]
        img_cond = np.unique(events)
        # Select only a maximum number of EEG repetitions
        if mode == 'test':
            max_rep = 20
        else:
            max_rep = 2
        # Sorted data matrix of shape:
        # Image conditions × EEG repetitions × EEG channels × EEG time points
        sorted_data = np.zeros((len(img_cond),max_rep,data.shape[1],
            data.shape[2]))
        for i in range(len(img_cond)):
            # Find the indices of the selected image condition
            idx = np.where(events == img_cond[i])[0]
            # Randomly select only the max number of EEG repetitions
            idx = shuffle(idx, random_state=seed, n_samples=max_rep)
            sorted_data[i] = data[idx]
        print(sorted_data[:, :, :, int(abs(tmin)*re_sfreq)+1:].shape)
        epoched_data.append(sorted_data[:, :, :, int(abs(tmin)*re_sfreq)+1:].astype(np.float16))
        img_conditions.append(img_cond) 
    return epoched_data,img_conditions,ch_names,times


def mvnn( epoched_test, epoched_train):
	"""Compute the covariance matrices of the EEG data (calculated for each
	time-point or epoch/repetitions of each image condition), and then average
	them across image conditions and data partitions. The inverse of the
	resulting averaged covariance matrix is used to whiten the EEG data
	(independently for each session).
	
	zero-score standardization also has well performance

	Parameters
	----------
	args : Namespace
		Input arguments.
	epoched_test : list of floats
		Epoched test EEG data.
	epoched_train : list of floats
		Epoched training EEG data.

	Returns
	-------
	whitened_test : list of float
		Whitened test EEG data.
	whitened_train : list of float
		Whitened training EEG data.

	"""

	import numpy as np
	from tqdm import tqdm
	from sklearn.discriminant_analysis import _cov
	import scipy

	### Loop across data collection sessions ###
	whitened_test = []
	whitened_train = []
	for s in range(n_ses):
		session_data = [epoched_test[s], epoched_train[s]]

		### Compute the covariance matrices ###
		# Data partitions covariance matrix of shape:
		# Data partitions × EEG channels × EEG channels
		sigma_part = np.empty((len(session_data),session_data[0].shape[2],
			session_data[0].shape[2]))
		for p in range(sigma_part.shape[0]):
			# Image conditions covariance matrix of shape:
			# Image conditions × EEG channels × EEG channels
			sigma_cond = np.empty((session_data[p].shape[0],
				session_data[0].shape[2],session_data[0].shape[2]))
			for i in tqdm(range(session_data[p].shape[0])):
				cond_data = session_data[p][i]
				# Compute covariace matrices at each time point, and then
				# average across time points
				if mvnn_dim == "time":
					sigma_cond[i] = np.mean([_cov(cond_data[:,:,t],
						shrinkage='auto') for t in range(cond_data.shape[2])],
						axis=0)
				# Compute covariace matrices at each epoch (EEG repetition),
				# and then average across epochs/repetitions
				elif mvnn_dim == "epochs":
					sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]),
						shrinkage='auto') for e in range(cond_data.shape[0])],
						axis=0)
			# Average the covariance matrices across image conditions
			sigma_part[p] = sigma_cond.mean(axis=0)
		# # Average the covariance matrices across image partitions
		# sigma_tot = sigma_part.mean(axis=0)
		# ? It seems not fair to use test data for mvnn, so we change to just use training data
		sigma_tot = sigma_part[1]
		# Compute the inverse of the covariance matrix
		sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

		### Whiten the data ###
		whitened_test.append(np.reshape((np.reshape(session_data[0], (-1,
			session_data[0].shape[2],session_data[0].shape[3])).swapaxes(1, 2)
			@ sigma_inv).swapaxes(1, 2), session_data[0].shape))
		whitened_train.append(np.reshape((np.reshape(session_data[1], (-1,
			session_data[1].shape[2],session_data[1].shape[3])).swapaxes(1, 2)
				@ sigma_inv).swapaxes(1, 2), session_data[1].shape))

	### Output ###
	return whitened_test, whitened_train

if __name__=="__main__":
    
    sub=8
    mode='train'
    save_dir = os.path.join(project_dir,
    f'Preprocessed_data_{re_sfreq}Hz_no_whiten', 'sub-'+format(sub,'02'))
    
    if mode =='test': 
        epoched_data_test,img_conditions,ch_names,times = epoch_data('test',sub)
        session_list=np.zeros((200, 80))
        for s in range(n_ses):
            if s == 0:
                merged_test = epoched_data_test[s]
            else:
                merged_test = np.append(merged_test, epoched_data_test[s], 1)
            start_index = merged_test.shape[1]-epoched_data_test[s].shape[1]
            end_index = merged_test.shape[1]
            session_list[:,start_index:end_index]=s
        # 'img': duplicated_images,
        # 'label': label,
        img_directory = f'/root/workspace/wht/multimodal_brain/datasets/things-eeg-small/Image_set/train_images'
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()
        images = []  
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            images.extend(os.path.join(folder_path, img).rsplit("Image_set/")[-1] for img in all_images)
        img_list = np.tile(np.array(images)[:, np.newaxis], (1, 80))
        
        test_dict = {
            'eeg': merged_test,
            'img':img_list,
            'ch_names': ch_names,
            'times': times,
            'session_list': session_list,
        }

        file_name_test = 'test.pt'

        os.makedirs(save_dir, exist_ok=True)
        torch.save(test_dict, os.path.join(save_dir,file_name_test))
    
    else:
        epoched_data_train,img_conditions_train,ch_names,times = epoch_data('train',sub)
        
        ### Merge and save the training data ###
        ses_list=np.zeros((33080, 2))
        for s in range(n_ses):
            if s == 0:
                white_data = epoched_data_train[s]
                img_cond = img_conditions_train[s]
            else:
                white_data = np.append(white_data, epoched_data_train[s], 0)
                img_cond = np.append(img_cond, img_conditions_train[s], 0)
            start_index = white_data.shape[0] - epoched_data_train[s].shape[0]
            end_index = white_data.shape[0]
            ses_list[start_index:end_index] = s
        print('ses_list',len(ses_list))
        del epoched_data_train,img_conditions_train
        # Data matrix of shape:
        # Image conditions × EGG repetitions × EEG channels × EEG time points
        merged_train = np.zeros((len(np.unique(img_cond)), white_data.shape[1]*2,
            white_data.shape[2],white_data.shape[3]),dtype=np.float16)

        sorted_session_list = np.zeros((16540, 4),dtype=np.int32)
        for i in range(len(np.unique(img_cond))):
            # Find the indices of the selected category
            idx = np.where(img_cond == i+1)[0]
            
            for r in range(len(idx)):
                sorted_session_list[i][r*2:r*2+2]=ses_list[idx[r]]
                if r == 0:
                    ordered_data = white_data[idx[r]]
                else:
                    ordered_data = np.append(ordered_data, white_data[idx[r]], 0)
            merged_train[i] = ordered_data
            del ordered_data
            
        img_directory = f'/root/workspace/wht/multimodal_brain/datasets/things-eeg-small/Image_set/train_images'
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()
        images = []  
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            images.extend(os.path.join(folder_path, img).rsplit("Image_set/")[-1] for img in all_images)
        img_list = np.tile(np.array(images)[:, np.newaxis], (1, 4))
        
        train_dict = {
            'eeg': merged_train,
            'img':img_list,
            'session_list':sorted_session_list,
            'ch_names': ch_names,
            'times': times,
        }
        # Create the directory if not existing and save the data
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)

        file_name_train = 'train.pt'
        torch.save(train_dict, os.path.join(save_dir,file_name_train))