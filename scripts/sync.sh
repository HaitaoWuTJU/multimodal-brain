for i in {01..10}; do mkdir /dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten/sub-$i/ ; done


for i in {01..10}; do cp /root/workspace/wht/multimodal_brain/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten/sub-$i/*.pt /dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten/sub-$i/; done