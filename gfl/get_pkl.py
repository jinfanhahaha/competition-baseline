import xml.etree.ElementTree as ET
import os
import mmcv
import numpy as np

ILL_ANN = "../data/train/train_ann/"
ill_anns = os.listdir(ILL_ANN)
print(ill_anns)

res = []
key = {"window_shielding": 0, "non_traffic_sign": 1, "multi_signs": 2}

for i, ill_ann in enumerate(ill_anns):
    if ill_anns[i] == '.DS_Store':
        continue
    tree = ET.parse(ILL_ANN+ill_ann)
    xml_filename = tree.find("filename")
    xml_size = tree.find("size")
    xml_size_width = xml_size.find("width")
    xml_size_height = xml_size.find("height")
    tmp_dict = {}
    tmp_dict['filename'] = xml_filename.text
    tmp_dict['width'] = int(xml_size_width.text)
    tmp_dict['height'] = int(xml_size_height.text)

    tmp_ann = {}
    xml_objects = tree.findall("object")
    boxes = []
    labels = []
    flag = False
    for xml_object in xml_objects:
        box = []
        xml_bndboxs = xml_object.findall("bndbox")
        xml_bndboxs_xmin = xml_bndboxs[0].find("xmin")
        xml_bndboxs_ymin = xml_bndboxs[0].find("ymin")
        xml_bndboxs_xmax = xml_bndboxs[0].find("xmax")
        xml_bndboxs_ymax = xml_bndboxs[0].find("ymax")
        if int(xml_bndboxs_ymin.text) == int(xml_bndboxs_ymax.text):
            print('hh')
            flag = True
        box.append(int(xml_bndboxs_xmin.text))
        box.append(int(xml_bndboxs_ymin.text))
        box.append(int(xml_bndboxs_xmax.text))
        box.append(int(xml_bndboxs_ymax.text))
        boxes.append(box)
        labels.append(int(key[xml_object.find("name").text]))
    tmp_ann['bboxes'] = np.array(boxes, dtype=np.float32)
    tmp_ann['labels'] = np.array(labels, dtype=np.int64)
    tmp_ann["bboxes_ignore"] = None
    tmp_ann['labels_ignore'] = None

    tmp_dict['ann'] = tmp_ann
    if flag:
        print("hhh")
        continue
    else:
        res.append(tmp_dict)

mmcv.dump(res, 'train_ann.pkl')
