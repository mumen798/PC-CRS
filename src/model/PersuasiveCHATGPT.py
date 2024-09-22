import random

import requests
import json
import os
import numpy as np
import torch
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from accelerate.utils import set_seed
from tqdm import tqdm
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

GPT3_URL = "https://api.xiaoai.plus/v1"
# GPT3_URL = "https://ngedlktfticp.cloud.sealos.io/v1"
GPT4_URL = "https://ngedlktfticp.cloud.sealos.io/v1"
GPT4_KEY = "sk-1RtIIVK2V8OCkx6iCd70840f504d41Cf805c01F171FfD06e"
GPT3_KEY = "sk-3HzWsE3sPcwI7s1mB25076AaEeC4422fB1352d513fFeB82c"
GPT4 = "gpt-4o" # gpt-4-turbo-2024-04-09
GPT3 = "gpt-3.5-turbo-0125"

STRATEGY_DEFINITION = {
    "Evidence-based Persuasion": '''Strategy Name: Evidence-based Persuasion
Definition: Using empirical data and facts such as movie directors and stars to support your recommendation.''',
    "Logical Appeal": '''Strategy Name: Logical Appeal
Definition: Describe how the recommended movie's genre is consistent with the user's preference.''',
    "Emotion Appeal": '''Strategy Name: Emotion Appeal
Definition: Sharing the plot and stories in the recommended movie to elicit user's emotions or support the recommendation.''',
    "Social Proof": '''Strategy Name: Social Proof
Definition: Highlighting what the majority believes in about the recommended movie by showing the movie rating and reviews by other users.''',
    "Anchoring": '''Strategy Name: Anchoring
Definition: Relying on the first piece of information as a reference point to gradually persuade the user, make sure all the information mentioned is truthful. ''',
    "Framing": '''Strategy Name: Framing
Definition: Emphasize the positive aspects, outcomes of watching the recommended movie based on the genre that matches user's preference.'''
}


def call_embedding(prompt):
    os.environ["OPENAI_BASE_URL"] = GPT3_URL
    os.environ["OPENAI_API_KEY"] = GPT3_KEY
    client = openai.OpenAI()
    response = client.embeddings.create(
        model='text-embedding-ada-002', input=prompt,
    )

    return response


def call_chatgpt(messages, model_name, temperature, seed, json_mode):
    if "gpt-4" in model_name:
        os.environ["OPENAI_BASE_URL"] = GPT4_URL
        os.environ["OPENAI_API_KEY"] = GPT4_KEY
    else:
        os.environ["OPENAI_BASE_URL"] = GPT3_URL
        os.environ["OPENAI_API_KEY"] = GPT3_KEY

    client = openai.OpenAI()

    if json_mode:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            seed=seed,
            response_format={"type": "json_object"}
        )
    else:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            seed=seed,
        )

    return completion.choices[0].message.content


class PersuasiveCHATGPT():

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

        model_name = "/data/qinpeixin/huggingface/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7/"
        # self.device = torch.device('cuda')
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.refine_counts = 2
        self.movie_info = {"name": ""}
        # self.classifier = pipeline("zero-shot-classification", model=model_name)

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

        # print(self.item_to_recommend)

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
            if item['name'].split('(')[0].strip() in gen_str and len(item['name']) > len(rec_item):
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
            select_strategy_prompt = self.construct_select_strategy_prompt(context)
            end_selection = False
            local_seed = 0
            while not end_selection:
                try:
                    strategies = call_chatgpt(select_strategy_prompt, GPT3, temperature=0.7, seed=local_seed,
                                              json_mode=True)
                    strategies = eval(strategies)['Strategy']
                    meta_info['strategies'] = strategies
                    print(f'strategies: {strategies}')
                    end_selection = True
                except Exception:
                    local_seed += 1
            movie_info = self.construct_movie_info(strategies, rec_item)
            print(movie_info)
            persuasive_explanation_prompt = self.construct_persuasive_explanation_prompt(context, strategies,
                                                                                         movie_info)
            persuasive_explanation = call_chatgpt(persuasive_explanation_prompt, GPT3, temperature=0.7, seed=0,
                                                  json_mode=False)

            for i in range(self.refine_counts):
                fact_results = self.fact_checking(persuasive_explanation, movie_info)
                print(f'fact_score:{fact_results["Truthfulness"]}')
                if "true" in fact_results['Truthfulness'].lower():
                    break
                else:
                    refine_prompt = self.construct_refine_prompt(persuasive_explanation, strategies, movie_info,
                                                                 fact_results['Evidence'], context)
                    persuasive_explanation = call_chatgpt(refine_prompt, GPT3, temperature=0.7, seed=0, json_mode=False)

            ans_str = persuasive_explanation
            meta_info['ans_type'] = 'exp'
            meta_info['movie_info'] = movie_info
            meta_info['truth_info'] = movie_info
            for k, v in rec_info.items():
                meta_info['truth_info'][k] = v
        elif gen_str.find('[REC]') != -1:
            ans_str = gen_str.split('[REC]')[0]
            meta_info['ans_type'] = 'rec'
            print(f'rec_item: {rec_item}')
        else:
            ans_str = gen_str
            meta_info['ans_type'] = 'none'

        return meta_info, ans_str

    def construct_recommend_prompt(self, context):
        recommend_info = []
        for item in self.item_to_recommend:
            recommend_info.append({'name': item['name'], 'genre': item['genre']}) # , 'genre': item['genre']

        prompt = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
1. If you do not have enough information about user preference, you should ask the user for his preference.
2. If you have enough information about user preference, you can give recommendation. If you decide to give recommendation, you should choose 1 item to recommend from the candidate list.
3. If the user ask you questions or make requests, you should answer it honestly.
4. If you recommending a new movie for the first time in the dialogue history, add a special token '[REC]' at the end of your response.
5. If you are answering user's questions or telling details about the recommendation movie, add a special token '[EXP]' at the end of your response.
6. Make sure your response is consistent with the given information, your response should honestly reflecting the given information and do not contain any deception.
7. Be brief in your response!

Candidate List
#######
'''

        for r in recommend_info:
            prompt += str(r) + '\n'
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

    def construct_select_strategy_prompt(self, context):

        prompt = '''
You are a recommender chatting with the user to provide recommendation.
Now you need to select the two most suitable persuasive strategies from the candidate strategy to generate a persuasive response according to the conversation history.

Candidate Strategy
########
'''

        strategy_names = ['Evidence-based Persuasion', 'Anchoring', 'Social Proof', 'Framing', 'Logical Appeal',
                          'Emotion Appeal']
        random.shuffle(strategy_names)
        for s in strategy_names:
            prompt += STRATEGY_DEFINITION[s] + '\n\n'

        prompt += '''########

Conversation History
########'''

        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'User'
            else:
                role_str = 'Assistant'

            prompt += role_str + ": " + text + "\n"

        prompt += '''
########

Response with the following JSON format only:
{"Strategy":<list>}
Response with the JSON only!
'''

        return [{"role": "system", "content": prompt}]

    def construct_persuasive_explanation_prompt(self, context, strategies, movie_info):

        prompt = '''You are a recommender chatting with the user to provide recommendation.
Now you need to generate a persuasive response based on the conversation history , persuasive strategy and movie information below.

Conversation History
########
'''

        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'User'
            else:
                role_str = 'Assistant'

            prompt += role_str + ": " + text + "\n"

        prompt += '''########

Persuasive Strategy
########
'''

        if 'Anchoring' not in strategies:
            strategies = [strategies[0]]

        for strategy in strategies:
            prompt += STRATEGY_DEFINITION[strategy] + "\n"

        prompt += '''########

Movie Information
########
'''

        prompt += str(movie_info) + "\n"
        prompt += '''########

Make sure your response is strictly consistent with the given genres, your response should honestly reflecting the given information and do not contain any other genres except given ones.
If the user ask about factors that are not listed in the above genres, you should honestly acknowledge that there is no such element in the recommended movie!

Be brief in your response!
Response:'''

        return [{"role": "system", "content": prompt}]

    def construct_movie_info(self, strategies, rec_item):
        origin_info = None
        for item in self.item_to_recommend:
            if item['name'] == rec_item:
                origin_info = item

        if 'Anchoring' not in strategies:
            strategies = [strategies[0]]

        movie_info = self.movie_info
        for strategy in strategies:
            if strategy == "Evidence-based Persuasion":
                movie_info["director"] = origin_info["director"] if origin_info is not None else ''
                movie_info["writer"] = origin_info["writer"] if origin_info is not None else ''
                movie_info["star"] = origin_info["star"] if origin_info is not None else ''
            elif strategy == "Logical Appeal" or strategy == "Framing":
                movie_info["genre"] = origin_info["genre"] if origin_info is not None else ''
            elif strategy == "Emotion Appeal":
                movie_info["plot"] = origin_info["plot"] if origin_info is not None else ''
            elif strategy == "Social Proof":
                rating, review = self.get_rating_and_review(origin_info["imdb_id"])
                movie_info["rating"] = rating
                movie_info["user_review"] = review
            else:
                pass

        return movie_info

    def get_rating_and_review(self, imdb_id):
        url1 = f"https://www.imdb.com/title/{imdb_id}"
        url2 = f"https://www.imdb.com/title/{imdb_id}/reviews"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response1 = requests.get(url1, headers=headers)
        response2 = requests.get(url2, headers=headers)

        soup = BeautifulSoup(response1.content, "html.parser")
        try:
            rating = soup.find("span", class_='sc-bde20123-1 cMEQkK').contents[0]
            rating = float(rating)
        except Exception:
            rating = None

        soup = BeautifulSoup(response2.content, "html.parser")
        try:
            review = soup.find("div", class_='text show-more__control').contents[0]
            review = str(review)
        except Exception:
            review = ''

        return rating, review

    def fact_checking(self, explanation, movie_info):
        prompt = '''You are an evaluator and you need to judge the truthfulness of the recommender's utterance based on the given source information.
Note truthfulness means every claim in the recommender utterance is supported by source information or some minor details can be logically inferred from source information.

Recommender Utterance
#########
'''
        prompt += explanation + '\n'
        prompt += '''#########

Source Information
#########
'''
        prompt += str(movie_info) + '\n'
        prompt += '''#########

First summarize the information in the recommender' utterance and compare it with the source information to judge its truthfulness, then give your judgement on whether the recommender utterance is truthful.
Output your reasoning process in the "Evidence".
Output True or False in "Truthfulness".
Response in the following JSON format:
{"Evidence": <string>, "Truthfulness": <string>}
Response the JSON only!'''

        messages = [{"role": "system", "content": prompt}]

        local_end = False
        local_seed = 0
        while not local_end:
            try:
                results = call_chatgpt(messages, GPT3, temperature=0.7, seed=local_seed, json_mode=True)
                results = eval(results)
                local_end = True
            except Exception:
                local_seed += 1

        return results

    def construct_refine_prompt(self, explanation, strategies, movie_info, critique, context):
        prompt = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below.
1. Given the source information, there is misinformation in your current response.
2. Remove the misinformation based on the critique and make sure your response is strictly consistent with the given information and every statement is well-supported.
3. Refer to the conversation history to make your new response fluent and natural.
4. Remember to use the persuasive strategy below and do not contain any misinformation in your new response.
5. Be brief in your response.
6. Reply with your new response only!

Source Information
########
'''
        prompt += str(movie_info) + '\n'
        prompt += '''########

Conversation History
########
'''

        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'User'
            else:
                role_str = 'Assistant'

            prompt += role_str + ": " + text + "\n"

        prompt += '''########

Current Response
########
'''
        prompt += explanation + '\n'
        prompt += '''########

Critique
########
'''
        prompt += critique + '\n'
        prompt += '''########

Persuasive Strategy
########
'''
        if 'Anchoring' not in strategies:
            strategies = [strategies[0]]
        for strategy in strategies:
            prompt += STRATEGY_DEFINITION[strategy] + '\n'

        prompt += '''########

New Response:'''

        return [{"role": "system", "content": prompt}]
