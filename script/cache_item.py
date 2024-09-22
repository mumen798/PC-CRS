import json
import os
import random
import typing
from argparse import ArgumentParser

import openai
from loguru import logger
from tenacity import _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


GPT3_URL = "https://api.xiaoai.plus/v1"
# GPT3_URL = "https://ngedlktfticp.cloud.sealos.io/v1"
GPT4_URL = "https://ngedlktfticp.cloud.sealos.io/v1"
GPT4_KEY = "sk-1RtIIVK2V8OCkx6iCd70840f504d41Cf805c01F171FfD06e"
GPT3_KEY = "sk-3HzWsE3sPcwI7s1mB25076AaEeC4422fB1352d513fFeB82c"
GPT4 = "gpt-4o" # gpt-4-turbo-2024-04-09
GPT3 = "gpt-3.5-turbo-0125"


def call_embedding(prompt):
    os.environ["OPENAI_BASE_URL"] = GPT3_URL
    os.environ["OPENAI_API_KEY"] = GPT3_KEY
    client = openai.OpenAI()
    response = client.embeddings.create(
        model='text-embedding-ada-002', input=prompt,
    )

    return response


def get_exist_item_set():
    exist_item_set = set()
    for file in os.listdir(save_dir):
        user_id = os.path.splitext(file)[0]
        exist_item_set.add(user_id)
    return exist_item_set


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--api_key')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--dataset', type=str, choices=['redial', 'opendialkg'])
    args = parser.parse_args()

    openai.api_key = args.api_key
    batch_size = args.batch_size
    dataset = args.dataset

    save_dir = f'../save/embed/item/{dataset}'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'../data/{dataset}/id2info.json', encoding='utf-8') as f:
        id2info = json.load(f)

    # redial
    if dataset == 'redial':
        info_list = list(id2info.values())
        item_texts = []
        for info in info_list:
            item_text_list = [
                f"Title: {info['name']}", f"Genre: {', '.join(info['genre']).lower()}",
                f"Star: {', '.join(info['star'])}",
                f"Director: {', '.join(info['director'])}", f"Plot: {info['plot']}"
            ]
            item_text = '; '.join(item_text_list)
            item_texts.append(item_text)
        attr_list = ['genre', 'star', 'director']

    # opendialkg
    if dataset == 'opendialkg':
        item_texts = []
        for info_dict in id2info.values():
            item_attr_list = [f'Name: {info_dict["name"]}']
            for attr, value_list in info_dict.items():
                if attr != 'title':
                    item_attr_list.append(f'{attr.capitalize()}: ' + ', '.join(value_list))
            item_text = '; '.join(item_attr_list)
            item_texts.append(item_text)
        attr_list = ['genre', 'actor', 'director', 'writer']

    id2text = {}
    for item_id, info_dict in id2info.items():
        attr_str_list = [f'Title: {info_dict["name"]}']
        for attr in attr_list:
            if attr not in info_dict:
                continue
            if isinstance(info_dict[attr], list):
                value_str = ', '.join(info_dict[attr])
            else:
                value_str = info_dict[attr]
            attr_str_list.append(f'{attr.capitalize()}: {value_str}')
        item_text = '; '.join(attr_str_list)
        id2text[item_id] = item_text

    item_ids = set(id2info.keys()) - get_exist_item_set()
    while len(item_ids) > 0:
        logger.info(len(item_ids))

        # redial
        if dataset == 'redial':
            batch_item_ids = random.sample(tuple(item_ids), min(batch_size, len(item_ids)))
            batch_texts = [id2text[item_id] for item_id in batch_item_ids]

        # opendialkg
        if dataset == 'opendialkg':
            batch_item_ids = random.sample(tuple(item_ids), min(batch_size, len(item_ids)))
            batch_texts = [id2text[item_id] for item_id in batch_item_ids]

        batch_embeds = annotate(batch_texts)['data']
        for embed in batch_embeds:
            item_id = batch_item_ids[embed['index']]
            with open(f'{save_dir}/{item_id}.json', 'w', encoding='utf-8') as f:
                json.dump(embed['embedding'], f, ensure_ascii=False)

        item_ids -= get_exist_item_set()
