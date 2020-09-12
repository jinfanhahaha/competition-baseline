import os
import cv2
import xml.etree.ElementTree as ET


# in
ILL_ANN = "./data/train/Annotations/"
ILL_IMAGE = './data/train/Illegal_adv_images/'
NOR_IMAGE = './data/train/Normal_images/'
ill_anns = os.listdir(ILL_ANN)
ill_images = os.listdir(ILL_IMAGE)
nor_images = os.listdir(NOR_IMAGE)
print(ill_images)

# out
OUT_NOR_ANN = './data/train/Normal_Annotations/'
OUT_NOR_IMAGES = './data/train/New_Normal_Images/'


for i, ill_ann in enumerate(ill_anns):
    tree = ET.parse(ILL_ANN+ill_ann)
    xml_filename = tree.find("filename")
    xml_size = tree.find("size")
    xml_size_width = xml_size.find("width")
    xml_size_height = xml_size.find("height")
    img1 = cv2.imread(ILL_IMAGE + xml_filename.text)
    img2 = cv2.imread(NOR_IMAGE + nor_images[i])
    print(xml_filename.text)
    print(nor_images[i])
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    xml_filename.text = nor_images[i]
    xml_size_width.text = str(w2)
    xml_size_height.text = str(h2)

    xml_objects = tree.findall("object")
    iswrite = False
    for xml_object in xml_objects:
        xml_bndboxs = xml_object.findall("bndbox")
        xml_bndboxs_xmin = xml_bndboxs[0].find("xmin")
        xml_bndboxs_ymin = xml_bndboxs[0].find("ymin")
        xml_bndboxs_xmax = xml_bndboxs[0].find("xmax")
        xml_bndboxs_ymax = xml_bndboxs[0].find("ymax")

        if int(xml_bndboxs_xmin.text) > w2:
            xml_bndboxs_xmin.text = str(w2)
        if int(xml_bndboxs_xmax.text) > w2:
            xml_bndboxs_xmax.text = str(w2)
        if int(xml_bndboxs_ymin.text) > h2:
            xml_bndboxs_ymin.text = str(h2)
        if int(xml_bndboxs_ymax.text) > h2:
            xml_bndboxs_ymax.text = str(h2)

        xmin = int(xml_bndboxs_xmin.text)
        ymin = int(xml_bndboxs_ymin.text)
        xmax = int(xml_bndboxs_xmax.text)
        ymax = int(xml_bndboxs_ymax.text)
        crop = img1[ymin: ymax, xmin: xmax]
        print(crop.shape)
        print(img2[ymin: ymax, xmin: xmax].shape)
        if 'train_1075.jpg' == nor_images[i]:
            break
        elif 'train_606.jpg' == nor_images[i]:
            break
        elif 'train_1067.jpg' == nor_images[i]:
            break
        elif 'train_1996.jpg' == nor_images[i]:
            break
        elif 'train_885.jpg' == nor_images[i]:
            break
        elif 'train_667.jpg' == nor_images[i]:
            break
        elif 'train_124.jpg' == nor_images[i]:
            break
        elif 'train_1426.jpg' == nor_images[i]:
            break
        elif 'train_1385.jpg' == nor_images[i]:
            break
        elif 'train_1727.jpg' == nor_images[i]:
            break
        elif 'train_1848.jpg' == nor_images[i]:
            break
        elif 'train_828.jpg' == nor_images[i]:
            break
        elif 'train_381.jpg' == nor_images[i]:
            break
        else:
            img2[ymin: ymax, xmin: xmax] = crop
            iswrite = True

    if iswrite:
        tree.write(OUT_NOR_ANN + nor_images[i][:-4] + ".xml")
        cv2.imwrite(OUT_NOR_IMAGES + nor_images[i], img2)
