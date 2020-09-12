import xml.etree.ElementTree as ET
import os
from collections import Counter

ILL_ANN = "../data/train/train_ann/"
ill_anns = os.listdir(ILL_ANN)
print(ill_anns)

res = []
key = {"window_shielding": 0, "non_traffic_sign": 1, "multi_signs": 2}

for i, ill_ann in enumerate(ill_anns):
    if ill_ann == '.DS_Store':
        continue
    tree = ET.parse(ILL_ANN+ill_ann)
    xml_objects = tree.findall("object")
    for xml_object in xml_objects:
        res.append(int(key[xml_object.find("name").text]))

print(Counter(res))
