import typing
import torch
import json
import os

import nltk
import openai
import tiktoken
import numpy as np
import time
import requests

from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from accelerate.utils import set_seed
from tenacity import _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from thefuzz import fuzz
from tqdm import tqdm
import transformers

# os.environ["OPENAI_BASE_URL"] = "https://api.xiaoai.plus/v1"

GPT3_URL = "https://api.gptapi.us/v1"
# GPT3_URL = "https://ngedlktfticp.cloud.sealos.io/v1"
GPT4_URL = "https://api.gptapi.us/v1"
GPT4_KEY = "sk-USFpCc4Uk3OWh0XSBdAaCa440bB6441eA7125eB68dE56d3c"
GPT3_KEY = "sk-USFpCc4Uk3OWh0XSBdAaCa440bB6441eA7125eB68dE56d3c"
GPT4 = "gpt-4o-mini" # gpt-4-turbo-2024-04-09
GPT3 = "gpt-3.5-turbo"


def call_embedding(prompt):
    if prompt is None or prompt == '':
        prompt = '.'
    os.environ["OPENAI_BASE_URL"] = GPT3_URL
    os.environ["OPENAI_API_KEY"] = GPT3_KEY
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


class KGLLAMA():

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
        if context is None:
            context = ['']

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

        # print(self.item_to_recommend)

        return item_rank_arr, rec_labels

    def get_conv(self, conv_dict):

        context = conv_dict['context']
        # context_list = []  # for model

        goal_prompt = self.construct_goal_prompt(context)
        recommend_prompt = self.construct_recommend_prompt(context)

        print()
        print(goal_prompt)
        print()

        meta_info = {}

        local_end = False
        local_seed = 0
        while not local_end:
            try:
                gen_str = call_chatgpt(recommend_prompt, model_name=GPT3, temperature=0.7, seed=local_seed, json_mode=True)
                tmp = eval(gen_str)['Movie Recommendation']
                tmp = eval(gen_str)['Knowledge Triples']
                recommend = eval(gen_str)
                local_end = True
            except Exception:
                local_seed += 1

        local_end = False
        local_seed = 0
        while not local_end:
            try:
                gen_str = call_chatgpt(goal_prompt, model_name=GPT3, temperature=0.7, seed=local_seed, json_mode=True)
                goal = eval(gen_str)['Goal']
                local_end = True
            except Exception:
                local_seed += 1

        response_prompt = self.construct_response_prompt(goal, context, recommend)
        response = call_chatgpt(response_prompt, model_name=GPT3, temperature=0.7, seed=1, json_mode=False)

        rec_item = ''
        # rec_info = {'name': ''}
        # for item in self.item_to_recommend:
        #     if item['name'] in response and len(item['name']) > len(rec_item):
        #         rec_item = item['name']
        #         rec_info = item
        conv_embed = call_embedding(response).data[0].embedding
        conv_embed = np.asarray(conv_embed).reshape(1, -1)

        sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
        rank_arr = np.argsort(sim_mat, axis=-1).tolist()
        rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
        item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
        item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]

        meta_info['rec_item'] = ''
        # meta_info['rec_info'] = rec_info
        meta_info['strategies'] = None
        rec_info = self.id2info[self.entityid2id[item_rank_arr[0][0]]]
        print(rec_info)
        rec_item = rec_info['name']
        meta_info['rec_info'] = rec_info

        try:
            if rec_item.split('(')[0].strip() not in response:
                meta_info['truthfulness'] = 1
        except Exception:
            a = 0

        if goal.find('explanation') != -1:
            meta_info['ans_type'] = 'exp'
            meta_info['rec_item'] = rec_item
            # headers = {
            #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            # }
            # url = "http://www.omdbapi.com/?apikey=c97ae91&t="
            # if rec_item != '':
            #     for n in rec_item:
            #         url += n + '+'
            #     url = url[0:-1]
            #     r = requests.get(url, headers=headers)
            #     movie_data = eval(r.text)
            #     if "Error" in movie_data.keys():
            #         rec_info = {}
            #     else:
            #         rec_info = {'name': rec_item}
            #         rec_info['genre'] = movie_data['Genre'] if 'Genre' in movie_data.keys() else ''
            #         rec_info['director'] = movie_data['Director'] if 'Director' in movie_data.keys() else ''
            #         rec_info['writer'] = movie_data['Writer'] if 'Writer' in movie_data.keys() else ''
            #         rec_info['star'] = movie_data['Actors'] if 'Actors' in movie_data.keys() else ''
            #         rec_info['plot'] = movie_data['Plot'] if 'Plot' in movie_data.keys() else ''
            #         rec_info['imdb_id'] = movie_data['imdbID'] if 'imdbID' in movie_data.keys() else ''
            # else:
            #     rec_info = {}
            # print(rec_info)

            meta_info['movie_info'] = meta_info['rec_info']
        elif goal.find('recommendation') != -1:
            meta_info['ans_type'] = 'rec'
            meta_info['rec_item'] = rec_item
        else:
            meta_info['ans_type'] = 'none'

        ans_str = response

        return meta_info, ans_str

    def construct_recommend_prompt(self, context):
        prompt = '''You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations.
Given the dialogue history, your task is to first generate an appropriate knowledge triples and then generate appropriate 5 movie item Recommendations.

Dialogue History
##########
'''
        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user: ' + text + '\n'
            else:
                role_str = 'assistant: ' + text + '\n'

            prompt += role_str

        prompt += '''##########

Response in the following JSON format:
{"Knowledge Triples": <list>, "Movie Recommendation": <list>}
Return with the JSON only!'''

        return [{"role": "system", "content": prompt}]

    def construct_goal_prompt(self, context):
        prompt = '''You are an excellent goal planner and your task is to predict the next goal of the conversation given the dialogue history.
For the given dialogue, choose one of the goals for the next dialogue utterance from the given goal list:
["Greetings", "Ask about preference", "Give recommendation", "Make explanation", "Say goodbye"].

Dialogue History
##########
'''

        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user: ' + text + '\n'
            else:
                role_str = 'assistant: ' + text + '\n'

            prompt += role_str

        prompt += '''##########

Output your step-by-step reasoning process in "Thinking" then give your goal.
Response in the following JSON format.
{"Thinking": <string>, "Goal": <string>}
Response with the JSON only!'''

        return [{"role": "system", "content": prompt}]

    def construct_response_prompt(self, goal, context, recommend):
        prompt = '''You are an excellent conversational recommender who helps the user achieve recommendation-related goals through conversations.
Given the dialogue history, the conversation goal and candidate recommendation item for the next system response, your task is to generate an appropriate response. 
You can only choose one item to recommend in your response.

Dialogue History
##########
'''

        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user: ' + text + '\n'
            else:
                role_str = 'assistant: ' + text + '\n'

            prompt += role_str

        prompt += '''##########

Conversation Goal:'''

        prompt += goal + "\n\n"

        # rec_info = {'name': self.item_to_recommend[0]['name'], 'genre': self.item_to_recommend[0]['genre']}

        if 'recommendation' in goal:
            prompt += "Knowledge Triples: " + str(recommend['Knowledge Triples']) + "\n\n"
            prompt += "Candidate Movies: " + str(recommend['Movie Recommendation']) + "\n\n"

        prompt += "Your response:"

        return [{"role": "system", "content": prompt}]
