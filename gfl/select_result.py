import os
import json


DIR = "./result/"
NEW_DIR = "./new_result/"
if not os.path.exists(DIR):
    raise Exception("please check the path...")
if not os.path.exists(NEW_DIR):
    os.makedirs(NEW_DIR)

ann_list = os.listdir(DIR)
original_ill_data = 0
new_ill_data = 0
window_shielding = 0
non_traffic_sign = 0
multi_signs = 0
for ann in ann_list:
    new_tmp_res = []
    with open(DIR+ann, 'r') as f:
        content = json.loads(f.read())
        if content != []:
            original_ill_data += 1
        for c in content:
            if c['category'] == 'window_shielding' and c['score'] > 0.2:
                new_tmp_res.append(c)
                window_shielding += 1
            if c['category'] == 'non_traffic_sign' and c['score'] > 0.20:
                new_tmp_res.append(c)
                non_traffic_sign += 1
            if c['category'] == 'multi_signs' and c['score'] > 0.30:
                new_tmp_res.append(c)
                multi_signs += 1
    if new_tmp_res != []:
        new_ill_data += 1
    with open(NEW_DIR+ann, 'w') as w:
        json.dump(new_tmp_res, w, indent=2)

print('window_shielding: ', window_shielding)
print('non_traffic_sign: ', non_traffic_sign)
print('multi_signs: ', multi_signs)
print('original_ill_data: ', original_ill_data)
print('new_ill_data: ', new_ill_data)
print("平均检测个数要少于10个，考虑到数值的波动，平均检测个数在5-7个可能比较合适，\n"
      "通过调节每个类的score阈值，可以调节平均检测个数，也能找出理论上较为合适的阈值")
print("违法图片中的平均检测个数为: ", (window_shielding+non_traffic_sign+multi_signs) / new_ill_data)
