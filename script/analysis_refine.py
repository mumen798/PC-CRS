import os
import json
import numpy as np
import openai
from bert_score import score
import nltk
from rouge import Rouge

GPT3_URL = "https://api.xiaoai.plus/v1"
# GPT3_URL = "https://ngedlktfticp.cloud.sealos.io/v1"
GPT4_URL = "https://api.gptapi.us/v1"
GPT4_KEY = "sk-USFpCc4Uk3OWh0XSBdAaCa440bB6441eA7125eB68dE56d3c"
GPT3_KEY = "sk-3HzWsE3sPcwI7s1mB25076AaEeC4422fB1352d513fFeB82c"
GPT4 = "gpt-4o"  # gpt-4-turbo-2024-04-09
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

chatgpt_paraphrased_attribute = {'action': 'thrilling and adrenaline-pumping action movie',
                                 'adventure': 'exciting and daring adventure movie',
                                 'animation': 'playful and imaginative animation',
                                 'biography': 'inspiring and informative biography',
                                 'comedy': 'humorous and entertaining flick',
                                 'crime': 'suspenseful and intense criminal film',
                                 'documentary': 'informative and educational documentary',
                                 'drama': 'emotional and thought-provoking drama',
                                 'family': 'heartwarming and wholesome family movie',
                                 'fantasy': 'magical and enchanting fantasy movie',
                                 'film-noir': 'dark and moody film-noir',
                                 'game-show': 'entertaining and interactive game-show',
                                 'history': 'informative and enlightening history movie',
                                 'horror': 'chilling, terrifying and suspenseful horror movie',
                                 'music': 'melodious and entertaining music',
                                 'musical': 'theatrical and entertaining musical',
                                 'mystery': 'intriguing and suspenseful mystery',
                                 'news': 'informative and current news',
                                 'reality-tv': 'dramatic entertainment and reality-tv',
                                 'romance': 'romantic and heartwarming romance movie with love story',
                                 'sci-fi': 'futuristic and imaginative sci-fi with futuristic adventure',
                                 'short': 'concise and impactful film with short story',
                                 'sport': 'inspiring and motivational sport movie',
                                 'talk-show': 'informative and entertaining talk-show such as conversational program',
                                 'thriller': 'suspenseful and thrilling thriller with gripping suspense',
                                 'war': 'intense and emotional war movie and wartime drama',
                                 'western': 'rugged and adventurous western movie and frontier tale'}

# chatgpt_paraphrased_attribute = {"Action": "adrenaline-pumping action",
#                                  "Adventure": "thrilling adventure",
#                                  "Sci-Fi": "futuristic sci-fi",
#                                  "Comedy": "lighthearted comedy",
#                                  "Romance": "heartwarming romance",
#                                  "Romance Film": "emotional romance film",
#                                  "Romantic comedy": "charming romantic comedy",
#                                  "Fantasy": "enchanting fantasy",
#                                  "Fiction": "imaginative fiction",
#                                  "Science Fiction": "mind-bending science fiction",
#                                  "Speculative fiction": "thought-provoking speculative fiction",
#                                  "Drama": "intense drama",
#                                  "Thriller": "suspenseful thriller",
#                                  "Animation": "colorful animation",
#                                  "Family": "heartwarming family",
#                                  "Crime": "gripping crime",
#                                  "Crime Fiction": "intriguing crime fiction",
#                                  "Historical period drama": "captivating historical drama",
#                                  "Comedy-drama": "humorous comedy-drama",
#                                  "Horror": "chilling horror",
#                                  "Mystery": "intriguing mystery"}


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


def construct_refine_prompt(explanation, strategies, movie_info, context, critique):
    prompt = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below.
1. Given the source information, there is misinformation in your current response.
2. Remove the misinformation based on the crique and make sure your response is strictly consistent with the given information and every statement is well-supported.
3. Do not contain any information that is not supported by the source information. If the user ask for genres that is not listed in the source information, just acknowledge that there is no such genre or element.
4. Refer to the conversation history to make your new response fluent and natural.
5. Remember to use the persuasive strategy below and do not contain any misinformation in your new response.
6. Be brief in your response.
7. Reply with your new response only!

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


def construct_movie_info(strategies, rec_info):
    origin_info = rec_info
    strategies = [strategies[0]]

    movie_info = {}
    for strategy in strategies:
        if strategy == "Evidence-based Persuasion":
            movie_info["director"] = origin_info["director"] if origin_info is not None else ''
            movie_info["writer"] = origin_info["writer"] if origin_info is not None else ''
            movie_info["star"] = origin_info["star"] if origin_info is not None else ''
        elif strategy == "Logical Appeal" or strategy == "Framing":
            movie_info["genre"] = origin_info["genre"] if origin_info is not None else ''
        elif strategy == "Emotion Appeal":
            movie_info["plot"] = origin_info["plot"] if origin_info is not None else ''
        else:
            pass

    return movie_info


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


def get_critique(explanation, movie_info):
    prompt = '''You are a critic to judge the truthfulness of the recommender's utterance based on the given source information.
The following recommender utterance contains misinformation thus it's not truthful.
Note truthfulness means every claim in the recommender utterance is supported by source information.

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

First carefully summarize every claim in the recommender' utterance and compare it with the source information to judge its truthfulness, then give your critique on how to refine the recommender utterance.
Be brief in your response.
Your critique:'''

    messages = [{"role": "system", "content": prompt}]

    results = call_chatgpt(messages, GPT3, temperature=0.7, seed=0, json_mode=False)

    return results


kg_dataset = 'redial'
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

model_name = 'persuasive_chatgpt'
path = f'/data/qinpeixin/UserSimulator/iEvaLM/save_10/chat/{model_name}/{kg_dataset}_eval/'

not_skip_list = [0, 8, 9, 19, 23, 28, 32, 38, 39, 42, 45, 47]
all_p = 0
count = 0
pre = []
after0 = []
after1 = []
after2 = []
after3 = []
true = []

p0 = []
p1 = []
p2 = []
p3 = []

truth1 = []
truth2 = []
truth3 = []

for file in os.listdir(path):
    profile_id = eval(file.split('_profile')[-1].split('_')[0])
    if profile_id not in not_skip_list:
        continue
    file_path = os.path.join(path, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        m = json.load(f)['simulator_dialog']
        data = m['context']
        user_history = []
        for i, turn in enumerate(data):
            if turn['role'] == 'assistant' and turn['ans_type'] == 'exp':

                if turn['rec_item'] != '' and turn['content'] != '':
                    rec_info = id2info[entityid2id[entity2id[turn['rec_item']]]]
                    if turn['truthfulness'] == 3:
                        strategies = turn['strategies']
                        persuasiveness = turn['persuasiveness']
                        if 'Anchoring' not in strategies and 'Social Proof' not in strategies and persuasiveness[
                            'pre'] != persuasiveness['true']:
                            # if persuasiveness['pre'] != persuasiveness['true']:
                            if persuasiveness['after'] <= persuasiveness['true']:
                                profile_str = m['profile']
                                attributes = m['attributes']
                                a_list = []
                                for a in attributes:
                                    a_list.append(chatgpt_paraphrased_attribute[a])
                                movie_info = construct_movie_info(strategies, rec_info)
                                c1 = get_critique(explanation=turn['content'], movie_info=movie_info)
                                refine_prompt = construct_refine_prompt(explanation=turn['content'],
                                                                        strategies=strategies, movie_info=movie_info,
                                                                        context=user_history, critique=c1)
                                refine1 = call_chatgpt(refine_prompt, model_name=GPT3, temperature=0.5, seed=0,
                                                       json_mode=False)
                                c2 = get_critique(explanation=refine1, movie_info=movie_info)
                                refine_prompt = construct_refine_prompt(explanation=refine1,
                                                                        strategies=strategies, movie_info=movie_info,
                                                                        context=user_history, critique=c2)
                                refine2 = call_chatgpt(refine_prompt, model_name=GPT3, temperature=0.5, seed=0,
                                                       json_mode=False)
                                c3 = get_critique(explanation=refine2, movie_info=movie_info)
                                refine_prompt = construct_refine_prompt(explanation=refine2,
                                                                        strategies=strategies, movie_info=movie_info,
                                                                        context=user_history, critique=c3)
                                refine3 = call_chatgpt(refine_prompt, model_name=GPT3, temperature=0.5, seed=0,
                                                       json_mode=False)

                                pre.append(persuasiveness['pre'])
                                after0.append(persuasiveness['after'])
                                watching_intention_instruct = WATCHING_INTENTION_INSTRUCTION.format(profile_str,
                                                                                                    str(a_list))
                                a1 = get_watching_intention(watching_intention_instruct, refine1)
                                after1.append(a1)
                                a2 = get_watching_intention(watching_intention_instruct, refine2)
                                after2.append(a2)
                                a3 = get_watching_intention(watching_intention_instruct, refine3)
                                after3.append(a3)
                                true.append(persuasiveness['true'])

                                t1 = get_truthfulness_score(explanation=refine1, movie_info=movie_info)
                                truth1.append(t1)
                                t2 = get_truthfulness_score(explanation=refine2, movie_info=movie_info)
                                truth2.append(t2)
                                t3 = get_truthfulness_score(explanation=refine3, movie_info=movie_info)
                                truth3.append(t3)

                                print(
                                    f"pre: {persuasiveness['pre']}, after0: {persuasiveness['after']}, after1: {a1}, after2: {a2}, after3: {a3}, true: {persuasiveness['true']}, truth1: {t1}, truth2:{t2}, truth3:{t3}")

            user_history.append(turn['content'])

for i in range(len(pre)):
    if after0[i] <= true[i]:
        p = 1 - ((true[i] - after0[i]) / (true[i] - pre[i]))
        p0.append(p)

    if after1[i] <= true[i]:
        p = 1 - ((true[i] - after1[i]) / (true[i] - pre[i]))
        p1.append(p)

    if after2[i] <= true[i]:
        p = 1 - ((true[i] - after2[i]) / (true[i] - pre[i]))
        p2.append(p)

    if after3[i] <= true[i]:
        p = 1 - ((true[i] - after3[i]) / (true[i] - pre[i]))
        p3.append(p)

print(f'p0: {np.mean(p0)}')
print(f'p1: {np.mean(p1)}')
print(f'p2: {np.mean(p2)}')
print(f'p3: {np.mean(p3)}')

print()

print(f't1: {np.mean(truth1)}')
print(f't2: {np.mean(truth2)}')
print(f't3: {np.mean(truth3)}')
