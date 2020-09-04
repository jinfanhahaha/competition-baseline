import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


All_Images = False

TRAIN_PIC = "/raid/ours_data/Competition/NAIC-Segmentation/data/my_dataset/img_dir/train/"
TRAIN_TAG = "/raid/ours_data/Competition/NAIC-Segmentation/data/my_dataset/ann_dir/train/"
# VAL_PIC = "./data/val_pic/"
# VAL_TAG = "./data/val_tag/"

if All_Images:
    train_pic_list = os.listdir(TRAIN_PIC)
    train_tag_list = os.listdir(TRAIN_TAG)
    # val_pic_list = os.listdir(VAL_PIC)
    # val_tag_list = os.listdir(VAL_TAG)
else:
    train_pic_list = os.listdir(TRAIN_PIC)[:100]
    train_tag_list = os.listdir(TRAIN_TAG)
    train_tag_list = np.array(train_tag_list)
    p = np.random.permutation(len(train_tag_list))
    # print(len(p))
    train_tag_list = list(train_tag_list[p])[:100]
    # val_pic_list = os.listdir(VAL_PIC)[:100]
    # val_tag_list = os.listdir(VAL_TAG)[:100]


def show_images_shape_distribution(train_pic_list):
    train_h_w_ratios = []
    val_h_w_ratios = []
    train_hs = []
    train_ws = []
    # val_hs = []
    # val_ws = []
    for train_pic in train_pic_list:
        train_img = cv2.imread(TRAIN_PIC + train_pic, cv2.IMREAD_UNCHANGED)
        print(train_img.shape)
        print(train_img)
        train_h, train_w, _ = train_img.shape
        train_hs.append(train_h)
        train_ws.append(train_w)
        train_h_w_ratios.append(train_h / train_w)

    print("train_mean_w: ", sum(train_ws) / len(train_ws))
    print("train_mean_h: ", sum(train_hs) / len(train_ws))
    print("train_h_w_ratios: ", sum(train_h_w_ratios) / len(train_h_w_ratios))
    # print("val_mean_w: ", sum(val_ws) / len(val_ws))
    # print("val_mean_h: ", sum(val_hs) / len(val_ws))
    # print("val_h_w_ratios: ", sum(val_h_w_ratios) / len(val_h_w_ratios))

    train_num_bins = 20
    val_num_bins = 20

    plt.figure()
    plt.subplot(121)
    plt.hist(train_h_w_ratios, train_num_bins)
    # plt.subplot(122)
    # plt.hist(val_h_w_ratios, val_num_bins)
    # plt.show()


def show_labels_distribution(DIR, tag_list):
    label_dict = {}
    print("begin computing...")
    for i, tag in enumerate(tag_list):
        # print(DIR + tag)
        img = cv2.imread(DIR + tag, cv2.IMREAD_UNCHANGED)
        # print(img.shape)
        img = np.array(img)[0]
        for pixel in img.flatten():
            if label_dict.get(pixel):
                label_dict.setdefault(pixel, 1)
                label_dict[pixel] += 1
            else:
                label_dict.setdefault(pixel, 1)
        print("doing...{:.2f}%".format((i+1) / len(tag_list) * 100))
    print(label_dict)
    keys = sorted(label_dict.keys())

    print(DIR.split("/")[-2][:-4] + " label distribution!")
    for k in keys:
        print(str(k) + ": " + str(label_dict[k]))


show_labels_distribution(TRAIN_TAG, train_tag_list)
print("\n" + "#"*100)
# show_labels_distribution(VAL_TAG, val_tag_list)
# print("\n" + "#"*100)
# show_images_shape_distribution(train_pic_list)
