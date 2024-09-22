import os
import json
import numpy as np
import random
import csv
model_name = 'persuasive_chatgpt'
path = f'/data/qinpeixin/UserSimulator/iEvaLM/save_10/chat/{model_name}/redial_eval/'

strategy_dict = {"Logical Appeal": 0, "Emotion Appeal": 1, "Framing": 2, "Evidence-based Persuasion": 3, "Social Proof": 4}
exp_list = []

not_skip_list = [0, 8, 9, 19, 23, 28, 32, 38, 39, 42, 45, 47]
strategy_count = {}
strategy_success_count = {}
strategy_success_rate = {}
for id in not_skip_list:
    strategy_count[id] = {"Evidence-based Persuasion": 0, "Logical Appeal": 0, "Emotion Appeal": 0, "Social Proof": 0,
                          "Anchoring": 0, "Framing": 0}
    strategy_success_count[id] = {"Evidence-based Persuasion": 0, "Logical Appeal": 0, "Emotion Appeal": 0, "Social Proof": 0,
                          "Anchoring": 0, "Framing": 0}
    strategy_success_rate[id] = {"Evidence-based Persuasion": 0, "Logical Appeal": 0, "Emotion Appeal": 0,
                                  "Social Proof": 0,
                                  "Anchoring": 0, "Framing": 0}

for file in os.listdir(path):
    profile_id = eval(file.split('_profile')[-1].split('_')[0])
    if profile_id not in not_skip_list:
        continue
    file_path = os.path.join(path, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['simulator_dialog']['context']
        user_history = []
        for i, turn in enumerate(data):

            if turn['role'] == 'assistant' and turn['ans_type'] == 'exp':
                strategies = turn['strategies']

                if 'Anchoring' not in strategies:
                    strategies = [strategies[0]]
                    exp_list.append([turn['content'], strategy_dict[strategies[0]]])

                for s in strategies:
                    strategy_count[profile_id][s] += 1

                if turn['rec_success_dialogue']:
                    for s in strategies:
                        strategy_success_count[profile_id][s] += 1


for id in not_skip_list:
    for k, v in strategy_count[id].items():
        if v != 0:
            strategy_success_rate[id][k] = strategy_success_count[id][k] / v
        else:
            strategy_success_rate[id][k] = 0

random.shuffle(exp_list)
exp_list = exp_list[0:100]
# with open('/data/qinpeixin/UserSimulator/iEvaLM/strategy_label.csv', 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerows(exp_list)
print()
