import typing
import torch
import json
import os

import nltk
import openai
import tiktoken
import numpy as np
import time

import transformers
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from accelerate.utils import set_seed
from tenacity import _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from thefuzz import fuzz
from tqdm import tqdm

GPT3_URL = ""
GPT4_URL = ""
GPT4_KEY = ""
GPT3_KEY = ""
GPT4 = "gpt-4o" # gpt-4-turbo-2024-04-09
GPT3 = "gpt-3.5-turbo-0125"


def call_embedding(prompt):
    if prompt is None or prompt == '':
        prompt = '.'
    os.environ["OPENAI_BASE_URL"] = GPT4_URL
    os.environ["OPENAI_API_KEY"] = GPT4_KEY
    client = openai.OpenAI()
    response = client.embeddings.create(
        model='text-embedding-ada-002', input=prompt,
    )

    return response


model_id = "/data/qinpeixin/huggingface/llama3-8b-instruct/"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


def call_chatgpt(messages, model_name, temperature, seed, json_mode):
    # # if "gpt-4" in model_name:
    # #     os.environ["OPENAI_BASE_URL"] = GPT4_URL
    # #     os.environ["OPENAI_API_KEY"] = GPT4_KEY
    # # else:
    # #     os.environ["OPENAI_BASE_URL"] = GPT3_URL
    # #     os.environ["OPENAI_API_KEY"] = GPT3_KEY
    #
    # model_name = '/data/qinpeixin/huggingface/llama3-8b-instruct/'
    #
    # openai_api_base = "http://localhost:8000/v1"
    #
    # client = openai.OpenAI(base_url=openai_api_base)
    #
    # if json_mode:
    #     completion = client.chat.completions.create(
    #         model=model_name,
    #         messages=messages,
    #         temperature=temperature,
    #         seed=seed,
    #         response_format={"type": "json_object"},
    #     )
    # else:
    #     completion = client.chat.completions.create(
    #         model=model_name,
    #         messages=messages,
    #         temperature=temperature,
    #         seed=seed,
    #     )

    outputs = pipeline(
        messages,
        max_new_tokens=512,
        eos_token_id=terminators,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )

    return outputs[0]["generated_text"][-1]['content']


class CHATLLAMA():

    def __init__(self, seed, debug, kg_dataset) -> None:
        self.seed = seed
        self.debug = debug
        if self.seed is not None:
            set_seed(self.seed)

        self.kg_dataset = kg_dataset

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

        self.id2entityid = {}
        for id, info in self.id2info.items():
            if info['name'] in self.entity2id:
                self.id2entityid[id] = self.entity2id[info['name']]

        self.entityid2id = {}
        for id, entityid in self.id2entityid.items():
            self.entityid2id[entityid] = id

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
        self.item_to_recommend = []

        self.movie_info = {"name": ""}

    def get_rec(self, conv_dict):

        rec_labels = [self.entity2id[rec] for rec in conv_dict['rec'] if rec in self.entity2id]

        context = conv_dict['context']
        context_list = []  # for model

        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            context_list.append({
                'role': role_str,
                'content': text
            })

        conv_str = ""

        for context in context_list[-2:]:
            conv_str += f"{context['role']}: {context['content']} "

        conv_embed = call_embedding(conv_str).data[0].embedding
        conv_embed = np.asarray(conv_embed).reshape(1, -1)

        sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
        rank_arr = np.argsort(sim_mat, axis=-1).tolist()
        rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
        item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
        item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]

        self.item_to_recommend = []
        for i in range(5):
            self.item_to_recommend.append(self.id2info[self.entityid2id[item_rank_arr[0][i]]])

        print(self.item_to_recommend)

        return item_rank_arr, rec_labels

    def get_conv(self, conv_dict):

        context = conv_dict['context']
        # context_list = []  # for model

        recommend_prompt = self.construct_recommend_prompt(context)

        print()
        print(recommend_prompt)
        print()

        meta_info = {}
        gen_str = call_chatgpt(recommend_prompt, model_name=GPT3, temperature=0.7, seed=0, json_mode=False)

        rec_item = ''
        rec_info = {'name': ''}
        for item in self.item_to_recommend:
            if item['name'] in gen_str and len(item['name']) > len(rec_item):
                rec_item = item['name']
                rec_info = item

        meta_info['rec_item'] = rec_item
        meta_info['rec_info'] = rec_info
        meta_info['strategies'] = None

        new_item = False
        if self.movie_info['name'] != rec_item:
            self.movie_info = {"name": rec_item}
            new_item = True
        if rec_item == '':
            new_item = True

        if gen_str.find('[EXP]') != -1 or (not new_item and gen_str.find('[REC]') == -1):
            ans_str = gen_str.split('[EXP]')[0]
            meta_info['ans_type'] = 'exp'
            meta_info['movie_info'] = rec_info
            meta_info['truth_info'] = rec_info
        elif gen_str.find('[REC]') != -1:
            ans_str = gen_str.split('[REC]')[0]
            meta_info['ans_type'] = 'rec'
        else:
            ans_str = gen_str
            meta_info['ans_type'] = 'none'

        return meta_info, ans_str

    def construct_recommend_prompt(self, context):
        recommend_info = []
        for item in self.item_to_recommend:
            recommend_info.append({'name': item['name'], 'genre': item['genre']})

        prompt = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
1. If you do not have enough information about user preference, you should ask the user for his preference.
2. If you have enough information about user preference, you can give recommendation. If you decide to give recommendation, you should choose 1 item to recommend from the candidate list.
3. If the user ask you questions or make requests, you should respond to the user's request.
4. If you recommending a new movie for the first time in the dialogue history, add a special token '[REC]' at the end of your response.
5. If you are answering user's questions or telling details about the recommendation movie, give engaging descriptions and add a special token '[EXP]' at the end of your response.
6. Make sure your response is consistent with the given information, your response should honestly reflecting the given information and do not contain any deception.
7. Be brief in your response!

Candidate List
#######
'''

        for r in recommend_info:
            prompt += str(r['name']) + '\n'
        prompt += '''#######'''
        messages = [{"role": "system", "content": prompt}]

        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            messages.append({
                'role': role_str,
                'content': text
            })

        return messages
