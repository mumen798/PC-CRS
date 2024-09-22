import json
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
import openai
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import numpy as np
from collections import defaultdict

import sys

sys.path.append("..")

from src.model.barcor.kg_bart import KGForBART
from src.model.barcor.barcor_model import BartForSequenceClassification

GPT3_URL = "https://api.xiaoai.plus/v1"
# GPT3_URL = "https://ngedlktfticp.cloud.sealos.io/v1"
GPT4_URL = "https://ngedlktfticp.cloud.sealos.io/v1"
GPT4_KEY = "sk-1RtIIVK2V8OCkx6iCd70840f504d41Cf805c01F171FfD06e"
GPT3_KEY = "sk-3HzWsE3sPcwI7s1mB25076AaEeC4422fB1352d513fFeB82c"
GPT4 = "gpt-4o"  # gpt-4-turbo-2024-04-09
GPT3 = "gpt-3.5-turbo-0125"


def call_embedding(prompt):
    os.environ["OPENAI_BASE_URL"] = GPT3_URL
    os.environ["OPENAI_API_KEY"] = GPT3_KEY
    client = openai.OpenAI()
    response = client.embeddings.create(
        model='text-embedding-ada-002', input=prompt,
    )

    return response


class BARCOR():

    def __init__(self, seed, kg_dataset, debug, tokenizer_path, context_max_length,
                 rec_model, conv_model,
                 resp_max_length):
        self.seed = seed
        if self.seed is not None:
            set_seed(self.seed)
        self.kg_dataset = kg_dataset

        self.debug = debug
        # self.tokenizer_path = f"../src/{tokenizer_path}"
        self.tokenizer = AutoTokenizer.from_pretrained("/data/qinpeixin/huggingface/bart-base/")
        # self.tokenizer = AutoTokenizer.from_pretrained("D:\\llama\\bart-base")
        self.tokenizer.truncation_side = 'left'
        self.context_max_length = context_max_length

        self.padding = 'max_length'
        self.pad_to_multiple_of = 8

        self.accelerator = Accelerator(device_placement=False, mixed_precision='fp16')
        self.device = self.accelerator.device

        self.rec_model = f"../src/{rec_model}"
        self.conv_model = f"../src/{conv_model}"

        # conv
        self.resp_max_length = resp_max_length

        self.kg = KGForBART(kg_dataset=self.kg_dataset, debug=self.debug).get_kg_info()

        self.crs_rec_model = BartForSequenceClassification.from_pretrained(self.rec_model,
                                                                           num_labels=self.kg['num_entities']).to(
            self.device)
        self.crs_conv_model = AutoModelForSeq2SeqLM.from_pretrained(self.conv_model).to(self.device)
        self.crs_conv_model = self.accelerator.prepare(self.crs_conv_model)

        self.kg_dataset_path = f"../data/{self.kg_dataset}"
        with open(f"{self.kg_dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)
        with open(f"{self.kg_dataset_path}/id2info.json", 'r', encoding="utf-8") as f:
            self.id2info = json.load(f)

        self.id2entity = {}
        for k, v in self.entity2id.items():
            self.id2entity[int(v)] = k

        self.id2entityid = {}
        for id, info in self.id2info.items():
            if info['name'] in self.entity2id:
                self.id2entityid[id] = self.entity2id[info['name']]

        self.entityid2id = {}
        for id, entityid in self.id2entityid.items():
            self.entityid2id[entityid] = id

        self.last_rec = ''
        self.last_rec_info = ''

        self.item_embedding_path = f"../save/embed/item/{self.kg_dataset}"

        item_emb_list = []
        id2item_id = []
        for i, file in tqdm(enumerate(os.listdir(self.item_embedding_path))):
            item_id = os.path.splitext(file)[0]
            if item_id in self.id2entityid:
                id2item_id.append(item_id)

                with open(f'{self.item_embedding_path}/{file}', encoding='utf-8') as f:
                    embed = json.load(f)
                    item_emb_list.append(embed)

        self.id2item_id_arr = np.asarray(id2item_id)
        self.item_emb_arr = np.asarray(item_emb_list)

    def get_rec(self, conv_dict):

        # dataset
        text_list = []
        turn_idx = 0

        for utt in conv_dict['context']:
            if utt != '':
                text = ''
                if turn_idx % 2 == 0:
                    text += 'User: '
                else:
                    text += 'System: '
                text += utt
                text_list.append(text)
            turn_idx += 1

        context = f'{self.tokenizer.sep_token}'.join(text_list)
        context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)

        data_list = []

        for rec in conv_dict['rec']:
            if rec in self.entity2id:
                data_dict = {
                    'context': context_ids,
                    'entity': [self.entity2id[ent] for ent in conv_dict['entity'] if ent in self.entity2id],
                    'rec': self.entity2id[rec]
                }
                if 'template' in conv_dict:
                    data_dict['template'] = conv_dict['template']
                data_list.append(data_dict)

        # dataloader
        input_dict = defaultdict(list)
        label_list = []

        for data in data_list:
            input_dict['input_ids'].append(data['context'])
            label_list.append(data['rec'])

        input_dict = self.tokenizer.pad(
            input_dict, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )

        input_dict['labels'] = label_list

        for k, v in input_dict.items():
            if not isinstance(v, torch.Tensor):
                input_dict[k] = torch.as_tensor(v, device=self.device)

        labels = input_dict['labels'].tolist()
        self.crs_rec_model.eval()
        outputs = self.crs_rec_model(**input_dict)
        item_ids = torch.as_tensor(self.kg['item_ids'], device=self.device)
        logits = outputs['logits'][:, item_ids]
        ranks = torch.topk(logits, k=50, dim=-1).indices
        preds = item_ids[ranks].tolist()

        return preds, labels

    def get_conv(self, conv_dict):

        text_list = []
        turn_idx = 0
        for utt in conv_dict['context']:
            if utt != '':
                text = ''
                if turn_idx % 2 == 0:
                    text += 'User: '
                else:
                    text += 'System: '
                text += utt
                text_list.append(text)
            turn_idx += 1
        context = f'{self.tokenizer.sep_token}'.join(text_list)
        context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)

        if turn_idx % 2 == 0:
            user_str = 'User: '
        else:
            user_str = 'System: '
        resp = user_str + conv_dict['resp']
        resp_ids = self.tokenizer.encode(resp, truncation=True, max_length=self.resp_max_length)

        data_dict = {
            'context': context_ids,
            'resp': resp_ids,
        }

        input_dict = defaultdict(list)
        label_dict = defaultdict(list)

        input_dict['input_ids'] = data_dict['context']
        label_dict['input_ids'] = data_dict['resp']

        input_dict = self.tokenizer.pad(
            input_dict, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )

        label_dict = self.tokenizer.pad(
            label_dict, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )['input_ids']

        # input_dict['labels'] = label_dict

        for k, v in input_dict.items():
            if not isinstance(v, torch.Tensor):
                input_dict[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)

        self.crs_conv_model.eval()

        gen_args = {
            'min_length': 0,
            'max_length': self.resp_max_length,
            'num_beams': 1,
            'no_repeat_ngram_size': 3,
            'encoder_no_repeat_ngram_size': 3
        }

        gen_seqs = self.accelerator.unwrap_model(self.crs_conv_model).generate(**input_dict, **gen_args)
        gen_str = self.tokenizer.decode(gen_seqs[0], skip_special_tokens=True)

        conv_embed = call_embedding(gen_str).data[0].embedding
        conv_embed = np.asarray(conv_embed).reshape(1, -1)

        sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
        rank_arr = np.argsort(sim_mat, axis=-1).tolist()
        rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
        item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
        item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]

        meta_info = {}
        rec_info = self.id2info[self.entityid2id[item_rank_arr[0][0]]]
        print(rec_info)
        rec_item = rec_info['name']

        meta_info['rec_info'] = rec_info
        meta_info['strategies'] = None

        if rec_item.split('(')[0].strip() in gen_str:
            meta_info['rec_item'] = rec_item
            meta_info['ans_type'] = 'rec'
            self.last_rec = rec_item
            self.last_rec_info = rec_info
        else:
            if "?" in gen_str:
                meta_info['rec_item'] = ''
                meta_info['ans_type'] = 'none'
            else:
                meta_info['rec_item'] = self.last_rec
                meta_info['rec_info'] = self.last_rec_info

                meta_info['ans_type'] = 'exp'

        print(f"rec_item: {meta_info['rec_item']}")

        return meta_info, gen_str

    def get_choice(self, gen_inputs, options, state, conv_dict=None):
        outputs = self.accelerator.unwrap_model(self.crs_conv_model).generate(
            **gen_inputs,
            min_new_tokens=5, max_new_tokens=5, num_beams=1,
            return_dict_in_generate=True, output_scores=True
        )
        option_token_ids = [self.tokenizer.encode(f" {op}", add_special_tokens=False)[0] for op in options]
        option_scores = outputs.scores[-2][0][option_token_ids]
        state = torch.as_tensor(state, device=self.device, dtype=option_scores.dtype)
        option_scores += state
        option_with_max_score = options[torch.argmax(option_scores)]

        return option_with_max_score
