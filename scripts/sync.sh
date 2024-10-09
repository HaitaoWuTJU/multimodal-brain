#!/bin/bash

mkdir -p /dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_1000Hz_whiten/

for i in {01..10}; do mkdir -p /dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_1000Hz_whiten/sub-$i/ ; done

for i in {01..10}; do cp /home/wht/multimodal_brain/datasets/things-eeg-small/Preprocessed_data_1000Hz_whiten/sub-$i/*.pt /dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_1000Hz_whiten/sub-$i/; done

# cp -r /home/wht/multimodal_brain/datasets/things-eeg-small/Image_set /dev/shm/wht/datasets/things-eeg-small/

# cp /home/wht/multimodal_brain/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten/*.pt /dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten/