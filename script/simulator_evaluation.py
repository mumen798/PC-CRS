import json
import os
import numpy as np
path = 'D:\\code\\UserSimulator_A100\\iEvaLM\\win_rate\\data_annotation'

naturalness = []
usefulness = []

for file_name in os.listdir(path):
    file_path = os.path.join(path, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        naturalness.append(int(data['Naturalness']))
        usefulness.append(int(data['Usefulness']))

print(np.mean(naturalness))
print(np.mean(usefulness))