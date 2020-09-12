import os
import cv2
import mmcv


ILL_ANN = "./data/test/"
ill_anns = os.listdir(ILL_ANN)
print(ill_anns)

res = []
for img_ann in ill_anns:
    tmp_dict = {}
    tmp_dict["filename"] = img_ann
    img = cv2.imread(ILL_ANN + img_ann)
    h, w, _ = img.shape
    tmp_dict["height"] = h
    tmp_dict['width'] = w
    tmp_dict['ann'] = None
    res.append(tmp_dict)

mmcv.dump(res, 'test_ann.pkl')
