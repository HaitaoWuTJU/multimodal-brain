# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-08','sub-10']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-08','sub-10','sub-07']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-08','sub-10','sub-07','sub-04']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-08','sub-10','sub-07','sub-04','sub-03']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-08','sub-10','sub-07','sub-04','sub-03','sub-09']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-08','sub-10','sub-07','sub-04','sub-03','sub-09','sub-02']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-08','sub-10','sub-07','sub-04','sub-03','sub-09','sub-02','sub-01']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-08','sub-10','sub-07','sub-04','sub-03','sub-09','sub-02','sub-01','sub-06']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-08','sub-10','sub-07','sub-04','sub-03','sub-09','sub-02','sub-01','sub-06','sub-05']"


# batch_sizes=(128 256 512 1024 2048 4096 8192)
batch_sizes=(64 128 256 512 1024 2048 4096 8192)

batch_sizes=(32)
for batch_size in "${batch_sizes[@]}"
do
    echo "Training with batch size: $batch_size"
    CUDA_VISIBLE_DEVICES=1 python train_multi_subject_desubject.py --name "clip_transformer_desubject_$batch_size" --batch_size $batch_size --subject "['sub-01','sub-02','sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']"

done
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-02']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-03']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-04']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-05']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-06']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-07']"

# batch_sizes=(4096)
# # Subject
# subject="['sub-08']"

# # Loop over each batch size
# for batch_size in "${batch_sizes[@]}"
# do
#     echo "Training with batch size: $batch_size"
#     CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "$subject" --batch_size $batch_size
# done

# Subject
# subject=("['sub-01']" "['sub-02']" "['sub-03']" "['sub-04']" "['sub-05']" "['sub-06']" "['sub-07']" "['sub-08']" "['sub-09']" "['sub-10']")

# # Loop over each batch size
# for subject in "${subject[@]}"
# do
#     echo "Training with batch size: $subject"
#     CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "$subject" --batch_size 4096
# done

# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-09']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-10']"
# CUDA_VISIBLE_DEVICES=0 python train_multi_subject.py --subject "['sub-01','sub-02','sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']"

# python class_subject.py --subjects "['sub-01','sub-02','sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']"