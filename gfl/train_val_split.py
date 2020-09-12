import numpy as np
import shutil
import os


# 定义原始存放数据的路径
DATA_PIC = "../data/train/ill_data/"
ANN = '../data/train/ANN/'

# 定义存放划分后数据的路径
TRAIN_PIC = "../data/train/train/"
VAL_PIC = "../data/train/val/"

train_ann = '../data/train/train_ann/'
val_ann = '../data/train/val_ann/'

data_pic_list = np.array(sorted(os.listdir(DATA_PIC)))
ann_list = np.array(sorted(os.listdir(ANN)))

# shuffle
p = np.random.permutation(len(data_pic_list))
data_pic_list = data_pic_list[p]
ann_list = ann_list[p]

# 划分训练验证集
split_k = len(data_pic_list) // 8
train_pic_list = data_pic_list[split_k:]
val_pic_list = data_pic_list[: split_k]

ann_train = ann_list[split_k:]
ann_val = ann_list[: split_k]

# 创建目录
if not os.path.exists(TRAIN_PIC):
    os.makedirs(TRAIN_PIC)
if not os.path.exists(VAL_PIC):
    os.makedirs(VAL_PIC)
if not os.path.exists(train_ann):
    os.makedirs(train_ann)
if not os.path.exists(val_ann):
    os.makedirs(val_ann)

# do it
for train_pic in train_pic_list:
    shutil.move(DATA_PIC + train_pic, TRAIN_PIC + train_pic)

for val_pic in val_pic_list:
    shutil.move(DATA_PIC + val_pic, VAL_PIC + val_pic)

for t_ann in ann_train:
    shutil.move(ANN + t_ann, train_ann + t_ann)

for v_ann in ann_val:
    shutil.move(ANN + v_ann, val_ann + v_ann)


os.rmdir(DATA_PIC)
os.rmdir(ANN)
print("finished!")
