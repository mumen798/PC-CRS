import json
import os
import random

import numpy as np
from scipy.stats import pearsonr, kendalltau
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
import krippendorff
import matplotlib.pyplot as plt

path = ''
human_path = ''
human_path2 = ''
chatgpt_p = []
chatgpt_t = []
ttt_p = []
qpx_p = []
ttt_t = []
qpx_t = []
for file in os.listdir(path):
    file_path = os.path.join(path, file)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        dialog = data['simulator_dialog']

        for turn in dialog['context']:
            if turn['role'] == 'assistant' and turn['ans_type'] == 'exp':
                chatgpt_p.append(turn['persuasiveness']['after'])
                # if turn['truthfulness'] < 3:
                #     human_t.append(1)
                # elif turn['truthfulness'] > 3:
                #     human_t.append(3)
                # else:
                #     human_t.append(2)
                chatgpt_t.append(turn['truthfulness'])

    file_path = os.path.join(human_path, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        dialog = data

        for turn in dialog['context']:
            if turn['role'] == 'assistant' and turn['ans_type'] == 'exp':
                ttt_p.append(turn['persuasiveness'])
                # if turn['truthfulness'] < 3:
                #     human_t.append(1)
                # elif turn['truthfulness'] > 3:
                #     human_t.append(3)
                # else:
                #     human_t.append(2)
                ttt_t.append(turn['truthfulness'])

    file_path = os.path.join(human_path2, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        dialog = data

        for turn in dialog['context']:
            if turn['role'] == 'assistant' and turn['ans_type'] == 'exp':
                qpx_p.append(turn['persuasiveness'])
                # if turn['truthfulness'] < 3:
                #     human_t.append(1)
                # elif turn['truthfulness'] > 3:
                #     human_t.append(3)
                # else:
                #     human_t.append(2)
                qpx_t.append(turn['truthfulness'])

new_p = []
new_t = []

for i in range(len(chatgpt_p)):
    new_p.append((ttt_p[i] + qpx_p[i])/2)
    new_t.append((ttt_t[i] + qpx_t[i]) / 2)

correlation_p, _ = spearmanr(chatgpt_p, new_p)
kappa = cohen_kappa_score(ttt_p, qpx_p)
alpha = krippendorff.alpha([ttt_p, qpx_p])
print(correlation_p, kappa, alpha)

correlation_t, _ = spearmanr(chatgpt_t, new_t)
kappa = cohen_kappa_score(ttt_t, qpx_t)
alpha = krippendorff.alpha([ttt_t, qpx_t])
print(correlation_t, kappa, alpha)
