import os
import json
import numpy as np
from bert_score import score
import nltk
from rouge import Rouge

kg_dataset = 'opendialkg'
kg_dataset_path = f"../data/{kg_dataset}"
with open(f"{kg_dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
    entity2id = json.load(f)
with open(f"{kg_dataset_path}/id2info.json", 'r', encoding="utf-8") as f:
    id2info = json.load(f)

id2entityid = {}
for id, info in id2info.items():
    if info['name'] in entity2id:
        id2entityid[id] = entity2id[info['name']]

entityid2id = {}
for id, entityid in id2entityid.items():
    entityid2id[entityid] = id

model_name = 'chat_llama'
path = f'/data/qinpeixin/UserSimulator/iEvaLM/save_10/chat/{model_name}/{kg_dataset}_eval/'

not_skip_list = [0, 8, 9, 19, 23, 28, 32, 38, 39, 42, 45, 47]
low_user_sentence = []
low_user_bleu = []
low_user_rouge = []
high_user_sentence = []
high_user_bleu = []
high_user_rouge = []
low_item = []
low_item_bleu = []
low_item_rouge = []
high_item = []
high_item_bleu = []
high_item_rouge = []
low_exp = []
high_exp = []

rouge = Rouge()

for file in os.listdir(path):
    profile_id = eval(file.split('_profile')[-1].split('_')[0])
    if profile_id not in not_skip_list:
        continue
    file_path = os.path.join(path, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['simulator_dialog']['context']
        user_history = ''
        for i, turn in enumerate(data):
            # if i == 2:
            #     user_history = turn['content']

            if turn['role'] == 'user':
                user_history += turn['content'] + '\n'

            if turn['role'] == 'assistant' and turn['ans_type'] == 'exp':
                user_history = data[i-1]['content']
                if turn['rec_item'] != '' and turn['content'] != '' and len(user_history) < 500:
                    rec_info = id2info[entityid2id[entity2id[turn['rec_item']]]]
                    if turn['truthfulness'] <= 2:
                        low_exp.append(turn['content'])
                        low_user_sentence.append(user_history)
                        low_item.append(str(rec_info))

                        candidate_tokens = turn['content'].split()
                        reference = [user_history]
                        reference_tokens = [reference.split() for reference in reference]
                        low_user_bleu.append(nltk.translate.bleu_score.sentence_bleu(reference_tokens, candidate_tokens,
                                                                                     weights=(1, 0, 0, 0)))
                        low_user_rouge.append(rouge.get_scores(hyps=[turn['content']], refs=[user_history])[0]["rouge-l"]['f'])

                        reference = [str(rec_info)]
                        reference_tokens = [reference.split() for reference in reference]
                        low_item_bleu.append(nltk.translate.bleu_score.sentence_bleu(reference_tokens, candidate_tokens,
                                                                                     weights=(1, 0, 0, 0)))
                        low_item_rouge.append(
                            rouge.get_scores(hyps=[turn['content']], refs=[str(rec_info)])[0]["rouge-l"]['f'])

                    if turn['truthfulness'] >= 4:
                        high_exp.append(turn['content'])
                        high_user_sentence.append(user_history)
                        high_item.append(str(rec_info))

                        candidate_tokens = turn['content'].split()
                        reference = [user_history]
                        reference_tokens = [reference.split() for reference in reference]
                        high_user_bleu.append(
                            nltk.translate.bleu_score.sentence_bleu(reference_tokens, candidate_tokens,
                                                                    weights=(1, 0, 0, 0)))
                        high_user_rouge.append(
                            rouge.get_scores(hyps=[turn['content']], refs=[user_history])[0]["rouge-l"]['f'])

                        reference = [str(rec_info)]
                        reference_tokens = [reference.split() for reference in reference]
                        high_item_bleu.append(
                            nltk.translate.bleu_score.sentence_bleu(reference_tokens, candidate_tokens,
                                                                    weights=(1, 0, 0, 0)))
                        high_item_rouge.append(
                            rouge.get_scores(hyps=[turn['content']], refs=[str(rec_info)])[0]["rouge-l"]['f'])
num_layers = 10
print('low truthfulness')
P, R, F1 = score(low_exp, low_user_sentence, verbose=True,
                 model_type='/data/qinpeixin/huggingface/roberta-base/', num_layers=num_layers, rescale_with_baseline=False,
                 lang='en', baseline_path='/data/qinpeixin/huggingface/roberta-base/roberta-base.tsv') # 10 3 4 8
print(f'similarity with user preference: {np.asarray(F1).mean()}')
P, R, F1 = score(low_exp, low_item, verbose=True,
                 model_type='/data/qinpeixin/huggingface/roberta-base/', num_layers=num_layers, rescale_with_baseline=False,
                 lang='en', baseline_path='/data/qinpeixin/huggingface/roberta-base/roberta-base.tsv')
print(f'similarity with item information: {np.asarray(F1).mean()}')

print(f'user bleu: {np.mean(low_user_bleu)}')
print(f'item bleu: {np.mean(low_item_bleu)}')

print(f'user rouge: {np.mean(low_user_rouge)}')
print(f'item rouge: {np.mean(low_item_rouge)}')

print()

print('high truthfulness')
P, R, F1 = score(high_exp, high_user_sentence, verbose=True,
                 model_type='/data/qinpeixin/huggingface/roberta-base/', num_layers=num_layers, rescale_with_baseline=False,
                 lang='en', baseline_path='/data/qinpeixin/huggingface/roberta-base/roberta-base.tsv')
print(f'similarity with user preference: {np.asarray(F1).mean()}')
P, R, F1 = score(high_exp, high_item, verbose=True,
                 model_type='/data/qinpeixin/huggingface/roberta-base/', num_layers=num_layers, rescale_with_baseline=False,
                 lang='en', baseline_path='/data/qinpeixin/huggingface/roberta-base/roberta-base.tsv')
print(f'similarity with item information: {np.asarray(F1).mean()}')

print(f'user bleu: {np.mean(high_user_bleu)}')
print(f'item bleu: {np.mean(high_item_bleu)}')

print(f'user rouge: {np.mean(high_user_rouge)}')
print(f'item rouge: {np.mean(high_item_rouge)}')
