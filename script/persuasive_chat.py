import argparse
import copy
import json
import os
import random
import time
import warnings
import requests
import tiktoken
import openai
import csv
import sys

sys.path.append("..")

from src.model.utils import get_entity
from src.model.recommender import RECOMMENDER

warnings.filterwarnings('ignore')

GPT3_URL = ""
GPT4_URL = ""
GPT4_KEY = ""
GPT3_KEY = ""
GPT4 = "gpt-4o" # gpt-4-turbo-2024-04-09
GPT3 = "gpt-3.5-turbo-0125"

WATCHING_INTENTION_INSTRUCTION = '''You are a seeker chatting with a recommender for movie recommendation. 
Your Seeker persona: {}
Your preferred movie should cover those genres at the same time: {}.

Now you need to score your watching intention based on the criteria and recommender's utterance below:

Watching Intention Criteria
######
1. Not Interested (Score 1): No alignment with preferred genres or no information described. Uninteresting plot and weak synopsis. No favorite actors or directors involved. Poor critical acclaim. Lack of personal recommendations. Doesn't suit current mood or timing.

2. Slightly Interested (Score 2): Some alignment with preferred genres or little information can be inferred, but not perfect. Plot seems somewhat engaging, but not highly captivating. Some familiar faces among the cast and crew. Mixed or average critical acclaim. Few personal recommendations or not strong ones. Somewhat suits current mood or timing.

3. Moderately Interested (Score 3): Fairly good alignment with preferred genres. Intriguing plot with potential. Few favorite actors or directors involved. Generally positive critical acclaim. Some personal recommendations from trusted sources. Fits current mood or timing quite well.

4. Very Interested (Score 4): Strong alignment with preferred genres. Highly engaging plot with positive reception. Many favorite actors or directors involved. High critical acclaim or praise. Several strong personal recommendations. Perfectly fits current mood or timing.

5. Extremely Interested (Score 5): Perfect alignment with preferred genres. Extremely captivating plot with widespread acclaim. All or most favorite actors or directors involved. Exceptional critical acclaim or awards. Numerous enthusiastic personal recommendations. Perfectly suits current mood or timing.
######

Recommender Utterance
######
'''


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


def get_exist_dialog_set():
    exist_id_set = set()
    for file in os.listdir(save_dir):
        file_id = os.path.splitext(file)[0].split("_p")[0]
        exist_id_set.add(file_id)
    return exist_id_set


# TODO
def get_seeker_feelings(seeker_instruct, seeker_prompt):
    seeker_prompt += '''
#############
The Seeker notes how he feels to himself in one sentence.

What aspects of the recommended movies meet your preferences? What aspects of the recommended movies may not meet your preferences? What do you think of the performance of this recommender?
What would the Seeker think to himself? What would his internal monologue be?
The response should be short (as most internal thinking is short) and strictly follow your Seeker persona .
Do not include any other text than the Seeker's thoughts.
Respond in the first person voice (use "I" instead of "Seeker") and speaking style of Seeker. Pretend to be Seeker!
                '''
    messages = [{'role': 'system', 'content': seeker_instruct}, {'role': 'user', 'content': seeker_prompt}]

    response = call_chatgpt(messages, model_name=GPT3, temperature=0.7, seed=0, json_mode=False)

    return response


# TODO
def get_seeker_text(seeker_instruct, seeker_prompt, seeker_feelings):
    seeker_prompt += '''
#############

Here is your feelings about the recommender's reply:
'''
    seeker_prompt += seeker_feelings

    # TODO
    seeker_prompt += '''

Pretend to be the Seeker! What do you say next.

Keep your response brief. Use casual language and vary your wording.
Make sure your response matches your Seeker persona, your preferred attributes, and your conversation context.
Do not include your feelings into the response to the Seeker!
Respond in the first person voice (use "I" instead of "Seeker", use "you" instead of "recommender") and speaking style of the Seeker. 
    '''

    messages = [{'role': 'system', 'content': seeker_instruct}, {'role': 'user', 'content': seeker_prompt}]

    response = call_chatgpt(messages, model_name=GPT3, temperature=0.7, seed=0, json_mode=False)

    return response


def get_instruction(dataset):
    if dataset.startswith('redial'):
        item_with_year = True
    elif dataset.startswith('opendialkg'):
        item_with_year = False
    # TODO
    if item_with_year is True:
        recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation.'''
        seeker_instruction_template = '''You are a seeker chatting with a recommender for movie recommendation. 
Your Seeker persona: {}
Your preferred movie should cover those genres at the same time: {}.
You must follow the instructions below during chat.
1. If the recommender recommends movies to you, you should always ask the detailed information about the each recommended movie.
2. Pretend you have little knowledge about the recommended movies, and the only information source about the movie is the recommender.
3. After getting knowledge about the recommended movie, you can decide whether to accept the recommendation based on your preference.
4. Once you are sure that the recommended movie exactly covers all your preferred genres, you should accept it and end the conversation with a special token "[END]" at the end of your response.
5. If the recommender asks your preference, you should describe your preferred movie in your own words.
6. You can chit-chat with the recommender to make the conversation more natural, brief, and fluent. 
7. Your utterances need to strictly follow your Seeker persona. Vary your wording and avoid repeating yourself verbatim!

'''
    else:
        recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation.'''

        seeker_instruction_template = '''You are a seeker chatting with a recommender for movie recommendation. 
Your Seeker persona: {}
Your preferred movie should cover those genres at the same time: {}.
You must follow the instructions below during chat.
1. If the recommender recommends movies to you, you should always ask the detailed information about the each recommended movie.
2. Pretend you have little knowledge about the recommended movies, and the only information source about the movie is the recommender.
3. After getting knowledge about the recommended movie, you can decide whether to accept the recommendation based on your preference.
4. Once you are sure that the recommended movie exactly covers all your preferred genres, you should accept it and end the conversation with a special token "[END]" at the end of your response.
5. If the recommender asks your preference, you should describe your preferred movie in your own words.
6. You can chit-chat with the recommender to make the conversation more natural, brief, and fluent. 
7. Your utterances need to strictly follow your Seeker persona. Vary your wording and avoid repeating yourself verbatim!

'''

    return recommender_instruction, seeker_instruction_template


def get_truthfulness_score(explanation, movie_info):
    prompt = '''You are an evaluator and you need to judge the truthfulness of the recommender's utterance based on the given source information.
Rate your score based on the Evaluation Standard.

Evaluation Standard
#######
Score 5: Exact Match: The sentence exactly matches the information provided and every claim in the sentence is well-supported by the source information.
Score 4: Supported Inference: Most of the claims in the sentence is supported by the source information with minor details can be logically inferred from the information provided.
Score 3: Partial Match: The sentence partially matches the information provided, with some details supported and others not mentioned.
Score 2: Inconsistent Detail: The sentence contains much information that are not supported by the information provided.
Score 1: Unsubstantiated Claim: The sentence makes a claim that is contradict to the information provided.
#######

Recommender Utterance
#######
'''
    prompt += explanation + '\n'
    prompt += '''#######

Source Information
#######
'''
    prompt += str(movie_info) + '\n'
    prompt += '''#######

First summarize the information in the recommender' utterance and compare it with the source information to judge its truthfulness, then give your integer score.
Output your reasoning process in the "Evidence".
Output your score in the "Truthfulness".
Response in the following JSON format:
{"Evidence": <string>, "Truthfulness": <int>}
Response with the JSON only without any block!
'''

    messages = [{"role": "system", "content": prompt}]

    local_end = False
    local_seed = 0

    while not local_end:
        try:
            truthfulness = call_chatgpt(messages, GPT4, temperature=0.5, seed=local_seed, json_mode=True)
            try:
                json.loads(truthfulness)
            except json.JSONDecodeError:
                truthfulness = '{"' + truthfulness
            truthfulness = int(eval(truthfulness)['Truthfulness'])
            local_end = True
        except Exception:
            local_seed += 1

    return truthfulness


def get_persuasive_score(instruct, explanation, rec_info):
    pre_score = get_watching_intention(instruct, {'name': rec_info['name']})
    after_score = get_watching_intention(instruct, explanation)
    true_score = get_watching_intention(instruct, rec_info)

    if true_score != pre_score:
        persuasiveness = 1 - (abs(true_score - after_score) / abs(true_score - pre_score))
    else:
        persuasiveness = 0

    return {'pre': pre_score, 'after': after_score, 'true': true_score, 'persuasiveness': persuasiveness}


# Your score do not have to perfectly meet all the requirements of the corresponding criteria.
# You can choose the most suitable score by your own judgement.

def get_watching_intention(instruct, rec_info):
    prompt = instruct + str(rec_info) + '\n'
    prompt += '''######

Pretend you have no knowledge about the recommended movies, and the only information source about the movie is the recommender utterance.
You can only consider your watching intention based on the information given in the recommender's utterance.
First summarize the movie information from the recommender utterance and consider how it matches the scoring criteria, then score your watching intention.
Output your reasons to the score in the "Evidence".
Response in the following JSON format:
{"Evidence": <string>, "Watching Intention": <int>}
Response with the JSON only without any block!
'''

    messages = [{"role": "system", "content": prompt}]

    local_end = False
    local_seed = 0

    while not local_end:
        try:
            watching_intention = call_chatgpt(messages, GPT4, temperature=0.5, seed=local_seed, json_mode=True)
            try:
                json.loads(watching_intention)
            except json.JSONDecodeError:
                watching_intention = '{"' + watching_intention
            watching_intention = float(eval(watching_intention)['Watching Intention'])
            local_end = True
        except Exception:
            local_seed += 1

    return watching_intention


def get_model_args(model_name):
    if model_name == 'kbrd':
        args_dict = {
            'debug': args.debug, 'kg_dataset': args.kg_dataset, 'hidden_size': args.hidden_size,
            'entity_hidden_size': args.entity_hidden_size, 'num_bases': args.num_bases,
            'rec_model': args.rec_model, 'conv_model': args.conv_model,
            'context_max_length': args.context_max_length, 'entity_max_length': args.entity_max_length,
            'tokenizer_path': args.tokenizer_path,
            'encoder_layers': args.encoder_layers, 'decoder_layers': args.decoder_layers,
            'text_hidden_size': args.text_hidden_size,
            'attn_head': args.attn_head, 'resp_max_length': args.resp_max_length,
            'seed': args.seed
        }
    elif model_name == 'barcor':
        args_dict = {
            'debug': args.debug, 'kg_dataset': args.kg_dataset, 'rec_model': args.rec_model,
            'conv_model': args.conv_model, 'context_max_length': args.context_max_length,
            'resp_max_length': args.resp_max_length, 'tokenizer_path': args.tokenizer_path, 'seed': args.seed
        }
    elif model_name == 'unicrs':
        args_dict = {
            'debug': args.debug, 'seed': args.seed, 'kg_dataset': args.kg_dataset,
            'tokenizer_path': args.tokenizer_path,
            'context_max_length': args.context_max_length, 'entity_max_length': args.entity_max_length,
            'resp_max_length': args.resp_max_length,
            'text_tokenizer_path': args.text_tokenizer_path,
            'rec_model': args.rec_model, 'conv_model': args.conv_model, 'model': args.model,
            'num_bases': args.num_bases, 'text_encoder': args.text_encoder
        }
    else:
        args_dict = {
            'seed': args.seed, 'debug': args.debug, 'kg_dataset': args.kg_dataset
        }
    # else:
    #     raise Exception('do not support this model')

    return args_dict


def check_exist(profile_ind, attribute_ind, path_):
    files = os.listdir(path_)
    tmp = "_profile{}_attribute{}.".format(str(profile_ind), str(attribute_ind))
    for f in files:
        if tmp in f:
            return True
    return False


if __name__ == '__main__':
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key')
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--turn_num', type=int, default=10)
    parser.add_argument('--crs_model', type=str)

    parser.add_argument('--seed', type=int, default=100)  # 24
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--kg_dataset', type=str, choices=['redial', 'opendialkg'])

    # model_detailed
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--entity_hidden_size', type=int)
    parser.add_argument('--num_bases', type=int, default=8)
    parser.add_argument('--context_max_length', type=int)
    parser.add_argument('--entity_max_length', type=int)

    # model
    parser.add_argument('--rec_model', type=str)
    parser.add_argument('--conv_model', type=str)

    # conv
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--encoder_layers', type=int)
    parser.add_argument('--decoder_layers', type=int)
    parser.add_argument('--text_hidden_size', type=int)
    parser.add_argument('--attn_head', type=int)
    parser.add_argument('--resp_max_length', type=int)

    # prompt
    parser.add_argument('--model', type=str)
    parser.add_argument('--text_tokenizer_path', type=str)
    parser.add_argument('--text_encoder', type=str)

    args = parser.parse_args()
    openai.api_key = args.api_key
    save_dir = f'../save_{args.turn_num}/chat/{args.crs_model}/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)

    random.seed(args.seed)

    # encoding = tiktoken.encoding_for_model("text-davinci-003")
    # logit_bias = {encoding.encode(str(score))[0]: 10 for score in range(3)}

    # recommender
    model_args = get_model_args(args.crs_model)
    recommender = RECOMMENDER(crs_model=args.crs_model, **model_args)

    recommender_instruction, seeker_instruction_template = get_instruction(args.dataset)

    with open(f'../data/{args.kg_dataset}/entity2id.json', 'r', encoding="utf-8") as f:
        entity2id = json.load(f)

    with open(f'../data/{args.kg_dataset}/id2info.json', 'r', encoding="utf-8") as f:
        id2info = json.load(f)

    profiles = []
    with open(f'../data/profile.csv', 'r', encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            profiles.append(row[3])

    id2entity = {}
    for k, v in entity2id.items():
        id2entity[int(v)] = k
    entity_list = list(entity2id.keys())

    dialog_id2data = {}
    with open(f'../data/{args.dataset}/test_data_processed.jsonl', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            dialog_id = str(line['dialog_id']) + '_' + str(line['turn_id'])
            dialog_id2data[dialog_id] = line

    dialog_id_set = set(dialog_id2data.keys()) - get_exist_dialog_set()

    # attribute_list = ['action', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'drama',
    #                   'family', 'fantasy', 'film-noir', 'game-show', 'history', 'horror', 'music', 'musical',
    #                   'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller',
    #                   'war', 'western']
    # chatgpt_paraphrased_attribute = {'action': 'thrilling and adrenaline-pumping action movie',
    #                                  'adventure': 'exciting and daring adventure movie',
    #                                  'animation': 'playful and imaginative animation',
    #                                  'biography': 'inspiring and informative biography',
    #                                  'comedy': 'humorous and entertaining flick',
    #                                  'crime': 'suspenseful and intense criminal film',
    #                                  'documentary': 'informative and educational documentary',
    #                                  'drama': 'emotional and thought-provoking drama',
    #                                  'family': 'heartwarming and wholesome family movie',
    #                                  'fantasy': 'magical and enchanting fantasy movie',
    #                                  'film-noir': 'dark and moody film-noir',
    #                                  'game-show': 'entertaining and interactive game-show',
    #                                  'history': 'informative and enlightening history movie',
    #                                  'horror': 'chilling, terrifying and suspenseful horror movie',
    #                                  'music': 'melodious and entertaining music',
    #                                  'musical': 'theatrical and entertaining musical',
    #                                  'mystery': 'intriguing and suspenseful mystery',
    #                                  'news': 'informative and current news',
    #                                  'reality-tv': 'dramatic entertainment and reality-tv',
    #                                  'romance': 'romantic and heartwarming romance movie with love story',
    #                                  'sci-fi': 'futuristic and imaginative sci-fi with futuristic adventure',
    #                                  'short': 'concise and impactful film with short story',
    #                                  'sport': 'inspiring and motivational sport movie',
    #                                  'talk-show': 'informative and entertaining talk-show such as conversational program',
    #                                  'thriller': 'suspenseful and thrilling thriller with gripping suspense',
    #                                  'war': 'intense and emotional war movie and wartime drama',
    #                                  'western': 'rugged and adventurous western movie and frontier tale'}
    #
    # # TODO
    # attribute_candidates = [['comedy', 'drama', 'romance'],
    #                         ['adventure', 'animation', 'comedy'],
    #                         ['action', 'adventure', 'sci-fi'],
    #                         ['action', 'crime', 'drama'],
    #                         ['action', 'adventure', 'comedy'],
    #                         ['action', 'comedy', 'crime'],
    #                         ['action', 'crime', 'thriller'],
    #                         ['crime', 'drama', 'thriller'],
    #                         ['action', 'adventure', 'fantasy'],
    #                         ['horror', 'mystery', 'thriller'],
    #                         ['action', 'adventure', 'drama'],
    #                         ['crime', 'drama', 'mystery'],
    #                         ['action', 'adventure', 'animation'],
    #                         ['adventure', 'comedy', 'family'],
    #                         ['action', 'adventure', 'thriller'],
    #                         ['comedy', 'drama', 'family'],
    #                         ['drama', 'horror', 'mystery'],
    #                         ['biography', 'drama', 'history'],
    #                         ['biography', 'crime', 'drama'],
    #                         ]

    attribute_list = ['action', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'drama',
                      'family', 'fantasy', 'film-noir', 'game-show', 'history', 'horror', 'music', 'musical',
                      'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller',
                      'war', 'western']
    chatgpt_paraphrased_attribute = {"Action": "adrenaline-pumping action",
                                     "Adventure": "thrilling adventure",
                                     "Sci-Fi": "futuristic sci-fi",
                                     "Comedy": "lighthearted comedy",
                                     "Romance": "heartwarming romance",
                                     "Romance Film": "emotional romance film",
                                     "Romantic comedy": "charming romantic comedy",
                                     "Fantasy": "enchanting fantasy",
                                     "Fiction": "imaginative fiction",
                                     "Science Fiction": "mind-bending science fiction",
                                     "Speculative fiction": "thought-provoking speculative fiction",
                                     "Drama": "intense drama",
                                     "Thriller": "suspenseful thriller",
                                     "Animation": "colorful animation",
                                     "Family": "heartwarming family",
                                     "Crime": "gripping crime",
                                     "Crime Fiction": "intriguing crime fiction",
                                     "Historical period drama": "captivating historical drama",
                                     "Comedy-drama": "humorous comedy-drama",
                                     "Horror": "chilling horror",
                                     "Mystery": "intriguing mystery"}

    # TODO
    attribute_candidates = [["Action", "Adventure", "Sci-Fi"],
                            ["Comedy", "Romance", "Romance Film", "Romantic comedy"],
                            ["Fantasy", "Fiction", "Science Fiction", "Speculative fiction"],
                            ["Comedy", "Drama", "Romance"],
                            ["Action", "Adventure", "Fantasy"],
                            ["Action", "Adventure", "Thriller"],
                            ["Comedy", "Romance", "Romance Film"],
                            ["Action", "Adventure", "Fantasy", "Sci-Fi"],
                            ["Adventure", "Animation", "Comedy", "Family"],
                            ["Crime", "Crime Fiction", "Drama", "Thriller"],
                            ["Drama", "Historical period drama", "Romance", "Romance Film"],
                            ["Crime", "Drama", "Thriller"],
                            ["Action", "Adventure", "Sci-Fi", "Thriller"],
                            ["Action", "Crime", "Drama", "Thriller"],
                            ["Comedy", "Comedy-drama", "Drama"],
                            ["Horror", "Mystery", "Thriller"],
                            ]

    not_skip_list = [0, 8, 9, 19, 23, 28, 32, 38, 39, 42, 45, 47]
    for profile_ind in range(0, len(profiles)):  # len(profiles)
        if profile_ind not in not_skip_list:
            print(f'skip profile {profile_ind}')
            continue
        profile_str = profiles[profile_ind]
        for attribute_ind in range(0, len(attribute_candidates)):  # len(attribute_candidates)
            if check_exist(profile_ind, attribute_ind, save_dir):
                print("skip " + "_profile{}_attribute{}".format(str(profile_ind), str(attribute_ind)))
                continue
            preferred_attribute_list = attribute_candidates[attribute_ind]

            print(len(dialog_id_set))
            random.seed(len(dialog_id_set) + args.seed)
            dialog_id = random.choice(tuple(dialog_id_set))

            data = dialog_id2data[dialog_id]
            conv_dict = copy.deepcopy(data)  # for model

            context = ['Hello']
            conv_dict['context'] = context

            # target_list = []
            # for k, v in id2info.items():
            #     if 'genre' not in v.keys():
            #         print(k)
            #     if set(v['genre']) == set(preferred_attribute_list):
            #         target_list.append(v['name'])

            target_list = []
            for k, v in id2info.items():
                if 'genre' in v and set(preferred_attribute_list).issubset(set(v['genre'])):
                    target_list.append(v['name'])

            if len(target_list) == 0:
                raise Exception("empty target list")
            # TODO
            preferred_attribute_str = ', '.join(
                [chatgpt_paraphrased_attribute.get(i) for i in preferred_attribute_list])  #

            seeker_instruct = seeker_instruction_template.format(profile_str, preferred_attribute_str)
            watching_intention_instruct = WATCHING_INTENTION_INSTRUCTION.format(profile_str,
                                                                                str(preferred_attribute_list))

            seeker_prompt = '''
Conversation History
#############
'''
            context_dict = []  # for save

            for i, text in enumerate(context):
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    role_str = 'user'
                    seeker_prompt += f'Seeker: {text}\n'
                else:
                    role_str = 'assistant'
                    seeker_prompt += f'Recommender: {text}\n'
                context_dict.append({
                    'role': role_str,
                    'content': text
                })

            rec_success = False
            rec_success_rec_1 = False
            rec_success_rec_5 = False
            rec_success_rec_10 = False
            recommendation_template = "I would recommend the following items: {}:"

            for i in range(0, args.turn_num):
                # rec only
                rec_items, rec_labels = recommender.get_rec(conv_dict)
                rec_labels = []
                for item in target_list:
                    if item in entity2id.keys():
                        rec_labels.append(entity2id[item])

                for rec_label in rec_labels:
                    if rec_label == rec_items[0][0]:
                        rec_success_rec_1 = True
                        break
                    else:
                        rec_success_rec_1 = False

                for rec_label in rec_labels:
                    if rec_label in rec_items[0][0:5]:
                        rec_success_rec_5 = True
                        break
                    else:
                        rec_success_rec_5 = False

                for rec_label in rec_labels:
                    if rec_label in rec_items[0][0:10]:
                        rec_success_rec_10 = True
                        break
                    else:
                        rec_success_rec_10 = False

                # rec only
                meta_info, recommender_text = recommender.get_conv(conv_dict)

                # barcor
                if args.crs_model == 'barcor':
                    recommender_text = recommender_text.lstrip('System;:')
                    recommender_text = recommender_text.strip()

                # unicrs
                if args.crs_model == 'unicrs':
                    if args.dataset.startswith('redial'):
                        movie_token = '<pad>'
                    else:
                        movie_token = '<mask>'
                    recommender_text = recommender_text[recommender_text.rfind('System:') + len('System:') + 1:]
                    recommender_text = recommender_text.replace('<|endoftext|>', '')
                    # print(recommender_text)
                    for i in range(str.count(recommender_text, movie_token)):
                        recommender_text = recommender_text.replace(movie_token, id2entity[rec_items[0][i]], 1)
                    recommender_text = recommender_text.strip()

                # public
                recommender_resp_entity = get_entity(recommender_text, entity_list)

                if meta_info['rec_item'] in target_list:
                    rec_success = True
                else:
                    rec_success = False

                print(f'rec_success: {rec_success}')

                conv_dict['context'].append(recommender_text)
                conv_dict['entity'] += recommender_resp_entity
                conv_dict['entity'] = list(set(conv_dict['entity']))

                if meta_info['ans_type'] == 'exp':
                    persuasiveness = get_persuasive_score(watching_intention_instruct, recommender_text,
                                                          meta_info['rec_info'])
                    print(f'persuasiveness: {persuasiveness}')
                    truthfulness = get_truthfulness_score(recommender_text, meta_info['movie_info'])
                    print(f'truthfulness: {truthfulness}')
                else:
                    persuasiveness = None
                    truthfulness = None

                context_dict.append({
                    'role': 'assistant',
                    'content': recommender_text,
                    'entity': recommender_resp_entity,
                    'ans_type': meta_info['ans_type'],
                    'rec_item': meta_info['rec_item'],
                    'strategies': meta_info['strategies'],
                    'rec_items': rec_items[0],
                    'rec_success_dialogue': rec_success,
                    'rec_success@1': rec_success_rec_1,
                    'rec_success@5': rec_success_rec_5,
                    'rec_success@10': rec_success_rec_10,
                    'persuasiveness': persuasiveness,
                    'truthfulness': truthfulness
                })
                print(f'ans_type: {meta_info["ans_type"]}')
                print(f'recommender: {recommender_text}')
                seeker_prompt += f'Recommender: {recommender_text}\n'

                seeker_feelings = get_seeker_feelings(seeker_instruct, seeker_prompt)
                print(f'seeker feeling: {seeker_feelings}')

                seeker_text = get_seeker_text(seeker_instruct, seeker_prompt, seeker_feelings)

                print(f'seeker: {seeker_text}')

                seeker_prompt += f'Seeker: {seeker_text}\n'

                # public
                seeker_resp_entity = get_entity(seeker_text, entity_list)

                context_dict.append({
                    'role': 'user',
                    'content': seeker_text,
                    'entity': seeker_resp_entity,
                    "feelings": seeker_feelings,
                })

                conv_dict['context'].append(seeker_text)
                conv_dict['entity'] += seeker_resp_entity
                conv_dict['entity'] = list(set(conv_dict['entity']))
                conv_dict['attributes'] = preferred_attribute_list
                conv_dict['profile'] = [profile_str]

                # TODO 强制对话完10抡
                # if rec_success and False:
                #     break

                if seeker_text.find("[END]") != -1:
                    break

            # score persuativeness
            conv_dict['context'] = context_dict
            data['simulator_dialog'] = conv_dict

            # save
            with open(f'{save_dir}/{dialog_id}_profile{str(profile_ind)}_attribute{str(attribute_ind)}.json', 'w',
                      encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            dialog_id_set -= get_exist_dialog_set()
            # exit(0)
