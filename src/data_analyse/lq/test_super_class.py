import pandas as pd
from sklearn.decomposition import PCA
import umap
from os.path import join as pjoin
from PIL import Image
import numpy as np
import torch

# 读取 CSV 文件
df = pd.read_csv('/root/workspace/wht/multimodal_brain/src/results/train_intersection.csv')

class_name = []

img_name = df['Img'].values
img_name_trimmed = [name[:-8] for name in img_name]

class_name = set(img_name_trimmed)
captions = [f'{name}' for name in class_name]

# concepts = pd.read_csv(path.replace('things-fmri', 'things_concepts.tsv'), sep='\t')
# super_con = concepts.loc[concepts['uniqueID'] == fl, 'Top-down Category (manual selection)'].values
#print(captions)


# ##########
# def get_letter_range(char):
#     # 获取首字母的ASCII码
#     char_ascii = ord(char.upper())
    
#     # 根据ASCII码判断首字母所在的范围
#     if 65 <= char_ascii <= 67:
#         return 'A-C'
#     elif 68 <= char_ascii <= 75:
#         return 'D-K'
#     elif 76 <= char_ascii <= 81:
#         return 'L-Q'
#     elif 82 <= char_ascii <= 83:
#         return 'R-S'
#     elif 84 <= char_ascii <= 90:
#         return 'T-Z'
#     else:
#         return '不在指定范围内'

# image_save = []
# for item, img_path in enumerate(img_name):
#     sub_dir = 'object_images_'+get_letter_range(img_path[0])
#     full_path = pjoin('../../../data/things/things-image', sub_dir, img_path[:-8], img_path)
#     print(full_path)
#     img = Image.open(full_path).convert('RGB')
#     img = img.resize((224, 224))
#     img = np.asarray(img,dtype='float32')
#     img = img / 255.0
#     image_save.append(img[np.newaxis, :])

# image_save = np.concatenate(image_save)
# print(image_save.shape)
# np.save('train_image.npy', image_save)
# ##########

# image_save = np.load('../union/train_image.npy')
# image_save = torch.from_numpy(image_save.transpose(0, 3, 1, 2))

# # Prepare CLIP
# clip_extractor = Clipper('ViT-L/14', device='cuda:0', hidden_state=True, norm_embs=True)

# #clip_text = clip_extractor.embed_text(captions).float()
# chunked_tensors = torch.chunk(image_save, chunks=10, dim=0)
# clip_image_list = []
# for chunk in chunked_tensors:
#     clip_image = clip_extractor.embed_image(chunk).float()
#     clip_image_list.append(clip_image)

# clip_image = torch.cat(clip_image_list, 0)
# #clip_text = clip_text[:, -1, :].cpu().numpy()
# clip_image = clip_image[:, 0, :].cpu().numpy()
# print(clip_image.shape)

# np.save('../union/img_feature.npy', clip_image)


# clip_image = np.load('../union/img_feature.npy')
# pca = PCA(n_components=32)
# data_pca = pca.fit_transform(clip_image)
# # 使用 reducer 进行降维
# reducer = umap.UMAP(n_neighbors=100, random_state=42)
# embedding_2d = reducer.fit_transform(data_pca)

# np.save('../union/2d.npy', embedding_2d)

# def draw(embedding_2d):
#     import matplotlib
#     import matplotlib.pyplot as plt
#     matplotlib.use('Agg')
#     plt.figure()
#     plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], color='gray', s=5)
#     plt.savefig(f'../union/cluster of image.png')

# draw(embedding_2d)

array = np.load('2d.npy')
print(array.shape)
# 根据条件将数组分成四块
block1_indices = (array[:, 0] < 10) & (array[:, 1] >= 6.5)
block2_indices = array[:, 0] > 10
block3_indices = (array[:, 0] <= 5.1) & (array[:, 1] < 6)
block4_indices = (array[:, 0] > 5.1) & (array[:, 0] < 10) & (array[:, 1] < 6.5)
block4_indices[3100] = True
test = (array[:, 0] > 4.6) & (array[:, 0] < 5.5) & (array[:, 1] > 2.2) & (array[:, 1] < 3)
block4_indices = np.where(test, True, block4_indices)
block3_indices = np.where(test, False, block3_indices)

block1 = array[block1_indices]
block2 = array[block2_indices]
block3 = array[block3_indices]
block4 = array[block4_indices]

# print(block1.shape)
# print(block2.shape)
# print(block3.shape)
# print(block4.shape)

# ids = np.arange(6330)
# selected_ids = ids[block1_indices | block2_indices | block3_indices | block4_indices]
# # 获取没有被选中的 ID
# not_selected_ids = np.setdiff1d(ids, selected_ids)
# print(not_selected_ids)
# print(array[not_selected_ids])

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')
# plt.scatter(block1[:, 0], block1[:, 1], color='red', label='Block 1', s=5)
# plt.scatter(block2[:, 0], block2[:, 1], color='blue', label='Block 2', s=5)
# plt.scatter(block3[:, 0], block3[:, 1], color='green', label='Block 3', s=5)
# plt.scatter(block4[:, 0], block4[:, 1], color='orange', label='Block 4', s=5)

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.savefig(f'../union/split_cluster.png')


# class 1 others  4326
# class 2 animal  787
# class 3 food   643
# class 4 fruit or vegetable  574
img_name_trimmed = np.array(img_name_trimmed)
class_4 = img_name_trimmed[block3_indices]
print(set(class_4))