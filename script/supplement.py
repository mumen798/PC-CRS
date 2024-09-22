# coding=utf-8

import os
import csv
import copy
import json
import time
import openai
import collections
from src.model.recommender import RECOMMENDER
import jsonlines

DEBUG = 1
OPENAI = False


pool_keys = []

class FileResult:
    def __init__(self, profile, attribute_candidate, file_name):
        self.file_name = file_name
        self.profile = profile[0]
        self.profile_age = profile[1]
        self.attribute_candidate = attribute_candidate
        self.overall_performance = None
        self.user_satisfaction = None
        self.user_feeling_list = None
        self.overall_feelings = None
        self.relevance_score = None
        self.quality_score = None
        self.manner_score = None
        self.humanlike_score = None
        self.explanation_score = None

        self.consistency = None
        self.diag_recomds = None
        self.recomds = None
        self.rec_success_dialogues = None
        self.rec_success_recs = None
        self.chatgpt_success_list = None
        # raw data
        self.overall_response = None
        self.single_score = None
        self.user_feeling_score = None

    def __iter__(self):
        return iter([self.file_name, self.profile, self.profile_age, self.attribute_candidate, self.overall_performance,
                     self.user_satisfaction, self.user_feeling_list, self.overall_feelings,
                     self.relevance_score, self.quality_score, self.manner_score,
                     self.humanlike_score, self.explanation_score, self.consistency,
                     self.diag_recomds, self.recomds, self.rec_success_dialogues, self.rec_success_recs, self.chatgpt_success_list,
                     self.overall_response, self.single_score, self.user_feeling_score])

def find_all_diag(base):
    assert isinstance(base, str), "wrong type"
    assert len(base) > 0, "empty path " + base
    assert os.path.exists(base), "invalid path " + base
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.json') and "_profile" in f and "_attribute" in f:
                fullname = os.path.join(root, f)
                yield fullname


def get_conversation_history(j):
    format_ = '"user": "{}", "recommender system": "{}"'
    dialogs = j.get("simulator_dialog").get("context")
    res = "Conversation History = ["
    tmp = {}
    cnt = 1
    for dialog in dialogs:
        role = dialog.get("role").replace("assistant", "recommender system")
        content = dialog.get("content")
        tmp[role] = content
        if cnt % 2 == 0:
            res += str(dict(sorted(tmp.items(), key=lambda x: x[0], reverse=True))) + ","
        cnt += 1

    res_list = []
    for i in range(4):
        res_list.append(dialogs[i]['content'])

    res += "]"
    return res, res_list


def get_user_feelings(j):
    dialogs = j.get("simulator_dialog").get("context")
    res = {}
    ind = 1
    for dialog in dialogs:
        if dialog.get("role") == "user" and dialog.get("feelings"):
            res[ind] = dialog.get("feelings")
            ind += 1
    return "user feelings = " + str(res)


def go_chatgpt(instruction):
    assert len(instruction) > 0, "empty input"
    messages = [{'role': 'system', 'content': instruction}]#, {'role': 'user', 'content': seeker_prompt}]
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-1106', messages=messages, temperature=0, seed=1,
    )['choices'][0]['message']['content']
    if DEBUG:
        print("#############")
        print(instruction)
        print("-------------")
        print(response)
        print("#############")
    time.sleep(21)
    return response


def single_scoring(parsed_json, id):
    conversation_history, _ = get_conversation_history(parsed_json)
    prompt = seperator.join([single_score_prompt_head, bar, conversation_history, single_score_prompt_standard, bar])
    data = {"id": id, "input": prompt}
    with jsonlines.open('social_eval.jsonl', 'a') as f:
        f.write(data)
        return
    response = go_chatgpt(prompt)
    return response


def user_feeling_scoring(parsed_json):
    user_feelings = get_user_feelings(parsed_json)
    prompt = seperator.join([user_feeling_prompt_head, bar, user_feeling_standard, user_feelings, bar])

    response = go_chatgpt(prompt)
    return response


def overall_scoring(parsed_json):
    single_score = single_scoring(parsed_json)

    conversation_history, cs_list = get_conversation_history(parsed_json)

    conv_dict = copy.deepcopy(parsed_json)

    prompt = "paraphrase: " + cs_list[2]
    paraphrase = go_chatgpt(prompt)
    conv_dict['context'] = [cs_list[0], cs_list[1], paraphrase]


    rec_items, rec_labels = recommender.get_rec(conv_dict)
    _, recommender_text = recommender.get_conv(conv_dict)
    pass

    result = {
        'social-awareness': single_score,
        'paraphrase': paraphrase,
        'old_text': cs_list[3],
        'new_text': recommender_text
    }

    return result


def get_name(ss):
    res_all = []
    tt = list(filter(lambda x:len(x) > 0, ss.replace("!", ".").replace("?", ".").split(".")))
    for s in tt:
        res = []
        s_list = s.strip().split(" ")
        head_i = True
        begin = False
        for tmp in s_list:
            begin = False
            if tmp[0].isupper() and tmp != "I" and not head_i:
                res.append(tmp)
                begin = True
            try:
                if tmp[0] == "(" and tmp[-1] == ")" and int(tmp[1:-1]):
                    res.append(tmp)
                    begin = True
                    break
            except ValueError:
                continue
            if begin is False and len(res) > 0:
                res_all.append(" ".join(res))
                res = []
            head_i = False
        if len(res) > 0:
            res_all.append(" ".join(res))
    return list(set(res_all))

def get_recommendation_consistency_succ(j):
    diag_recomds = []
    recomds = []
    rec_success_dialogues = []
    rec_success_recs = []
    chatgpt_success_list = []
    dialogs = j.get("simulator_dialog").get("context")
    for dialog in dialogs:
        if dialog.get("role") == "assistant":
            entity = get_name(dialog.get("content"))
            entity = list(set(entity + dialog.get("entity")))
            diag_recomd = entity[0].lower() if len(entity) == 1 else "" if len(entity) == 0 else [e.lower() for e in entity]
            if dialog.get("rec_items")[0] not in id2info:
                print(dialog.get("rec_items")[0])
            recomd = id2info.get(dialog.get("rec_items")[0]).lower() if dialog.get("rec_items")[0] in id2info else ""
            diag_recomds.append(diag_recomd)
            recomds.append(recomd)
            rec_success_dialogues.append(dialog.get("rec_success_dialogue"))
            rec_success_recs.append(dialog.get("rec_success_rec"))
        if dialog.get("role") == "user":
            if dialog.get("content").find("[END]") > -1:
                chatgpt_success_list.append(1)

            else:
                chatgpt_success_list.append(0)

    # consistency rate
    consistency = [1 if diag_recomds[ind] == recomds[ind] or recomds[ind] in diag_recomds[ind] else 0 for ind in range(len(diag_recomds))]
    consistency = 1.0 * sum(consistency) / len(consistency)

    # success rate
    # TODO don't know how to calculate yet

    res = {"consistency": consistency, "diag_recomds": diag_recomds, "recomds": recomds,
           "rec_success_dialogues": rec_success_dialogues, "rec_success_recs": rec_success_recs, "chatgpt_success_list": chatgpt_success_list}
    return res


def get_stat(parsed_json):
    # {"consistency": consistency, "diag_recomds": diag_recomds, "recomds": recomds,
    #            "rec_success_dialogues": rec_success_dialogues, "rec_success_recs": rec_success_recs}
    recommendation_consistency = get_recommendation_consistency_succ(parsed_json)
    return recommendation_consistency


def get_json(s):
    ss = s[next(idx for idx, c in enumerate(s) if c in "{["):]
    return ss[: ss.rfind("}")+1]


# tranlate into csv line
def formation(overall_response, single_score, user_feeling_score, recommendation_consistency, profile, attribute_candidate, file_name):
    result = FileResult(profile, attribute_candidate, file_name)
    result.overall_response = overall_response.replace("\n", "  ")
    result.single_score = single_score.replace("\n", "  ")
    result.user_feeling_score = user_feeling_score.replace("\n", "  ")

    overall_response, single_score, user_feeling_score = eval(get_json(overall_response)), eval(get_json(single_score)), eval(get_json(user_feeling_score))
    result.humanlike_score = single_score.get("Human-like")[0]
    result.relevance_score = single_score.get("Relevance")[0]
    result.manner_score = single_score.get("Manner")[0]
    result.quality_score = single_score.get("Quality")[0]
    result.explanation_score = single_score.get("Explanation")[0]
    result.overall_feelings = user_feeling_score.get("overall feeling")
    result.user_feeling_list = [i[0] for i in user_feeling_score.get("sentence sentiment").values()]
    result.overall_performance = overall_response.get("Overall Performance")[0]
    result.user_satisfaction = overall_response.get("User Satisfaction")[0]
    result.consistency = recommendation_consistency.get("consistency")
    result.diag_recomds = recommendation_consistency.get("diag_recomds")
    result.recomds = recommendation_consistency.get("recomds")
    result.rec_success_dialogues = recommendation_consistency.get("rec_success_dialogues")
    result.rec_success_recs = recommendation_consistency.get("rec_success_recs")
    result.chatgpt_success_list = recommendation_consistency.get("chatgpt_success_list")

    if DEBUG:
        print(json.dumps(result.__dict__))
    return result


def filter_non_num(s):
    return "".join(filter(str.isdigit, s))

##################### Constant ##########################
crs_model = "chatgpt"
dialog_history_dir = "D:\\code\\UserSimulator_A100\\iEvaLM\\save_10\\chat\\chatgpt\\redial_eval"
api_key = "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"
openai.api_key = api_key

single_score_prompt_head = '''
You are an evaluator and you need to judge how does the recommender perform based on the following Conversation History. Please rate the recommender's performance based on the following Evaluation Standard.

Return the scores in a JSON format as follows:
{"Social-Awareness":[<int>, "<WHY>", "<CONCRETE EXAMPLE>"]}
'''

single_score_prompt_standard = '''
Evaluation Standard
#############
Social Awareness:
5: The recommender consistently shows its personal opinion and experience and offers help to the user with emotional support.
4: The recommender mostly shows its personal opinion and experience, with only a few instances of lacking engagement.
3: The recommender occasionally shows its personal opinion and experience or offering emotional support, but there are several instances of homologous response or lacking engagement.
2: The recommender rarely shows its personal opinion and experience or offering emotional support, with most responses lacking engagement or homologous.
1: The recommender consistently fails to offer social help or emotional support, with no personal opinions or experience in the recommender's utterances.
#############
'''

user_feeling_prompt_head = '''
The following sentences encode how the user feelings changes when using a recommender system. You need to identify the sentiment for each sentence and pick one sentiment for single sentence from the candidate sentiments. Finally, you need to summarize how user feeling changes and what is user's overall feeling

Return the results in a JSON format as follows: {"sentence sentiment": {"<SENTENCE INDEX>":["<SENTIMENT>", "<WHY>"]}, "overall feeling": "<OVERALL FEELING>", "feeling changes":"<HOW CHANGES>"]}
'''

user_feeling_standard = '''
candidate sentiments = ["Satisfaction", "Delight", "Disappointment", "Frustration", "Surprise", "Trust", "Curiosity", "Indifference", "Confusion", "Excitement"]
'''

overall_prompt_head = '''
You are an evaluator and you need to judge how does the recommender perform based on the following Conversation History, User Feelings, and Other Judgements. Please rate the recommender's performance based on the following Evaluation Standard.

Return the results in a JSON string as follows:
{"Overall Performance":[<int>, "<WHY>", "<CONCRETE EXAMPLE>"], "User Satisfaction":[<int>, "<WHY>", "<CONCRETE EXAMPLE>"]}
'''

overall_prompt_standard = '''
Evaluation Standard
#############
c1. Overall Performance:
5: Given the Other Judgements and User Feelings, the recommender's performance is excellent, meeting or exceeding expectations in all evaluation criteria.
4: Given the Other Judgements and User Feelings, the recommender's performance is good, with some minor areas for improvement in certain evaluation criteria.
3: Given the Other Judgements and User Feelings, the recommender's performance is average, with noticeable areas for improvement in several evaluation criteria.
2: Given the Other Judgements and User Feelings, the recommender's performance is below average, with significant areas for improvement in multiple evaluation criteria.
1: Given the Other Judgements and User Feelings, the recommender's performance is poor, failing to meet expectations in most or all evaluation criteria.

c2. User Satisfaction:
5: Given the User Feelings, the User thinks that the recommander system fully meets his/her needs, providing an exceptional user experience.
4: Given the User Feelings, the User thinks that the recommander system meets his/her needs. The user experience is good, but there are some areas that could be further improved.
3: Given the User Feelings, the User thinks that the recommander system performs adequately in recommendation. However, there is still room for improvement.
2: Given the User Feelings, the User thinks that the recommander system performs below average. The user experience is not ideal and requires improvement.
1: Given the User Feelings, the User thinks that the recommander system is very bad at recommendation. The user experience is extremely unsatisfactory
#############
'''

bar = "======================="
seperator = "\n\n"
save_dir = f'../scoring_{crs_model}'
os.makedirs(save_dir, exist_ok=True)

##################### Constant END ##########################

##################### Processing ##########################
profiles = []
profile_inds = []
with open(f'../data/profile.csv', 'r', encoding="utf-8") as f:
    reader = csv.reader(f)

    for row in reader:
        profiles.append(row[3])
        profile_inds.append(row[0].strip().split("---"))

attribute_candidates = [['comedy', 'drama', 'romance'],
                        ['adventure', 'animation', 'comedy'],
                        ['action', 'adventure', 'sci-fi'],
                        ['action', 'crime', 'drama'],
                        ['action', 'adventure', 'comedy'],
                        ['action', 'comedy', 'crime'],
                        ['action', 'crime', 'thriller'],
                        ['crime', 'drama', 'thriller'],
                        ['action', 'adventure', 'fantasy'],
                        ['horror', 'mystery', 'thriller'],
                        ['action', 'adventure', 'drama'],
                        ['crime', 'drama', 'mystery'],
                        ['action', 'adventure', 'animation'],
                        ['adventure', 'comedy', 'family'],
                        ['action', 'adventure', 'thriller'],
                        ['comedy', 'drama', 'family'],
                        ['drama', 'horror', 'mystery'],
                        ['biography', 'drama', 'history'],
                        ['biography', 'crime', 'drama'],
                        ]

with open(f'../data/redial/entity2id.json', 'r', encoding="utf-8") as f:
    id2info = json.load(f)
    id2info = {v:k for k,v in id2info.items()}

##################### Processing END ##########################

if __name__ == "__main__":
    # currently
    args_dict = {
        'seed': 0, 'debug': 0, 'kg_dataset': 'redial'
    }

    recommender = RECOMMENDER(crs_model='chatgpt', **args_dict)
    processed_file = []
    if os.path.exists(save_dir + "/" + crs_model + ".csv"):
        pass
        # with open(save_dir + "/" + crs_model + ".jsonl", 'r', encoding='utf-8') as csv_file:
        #     for line in csv_file:
        #         data = json.load(line)
        #         processed_file.append(data['id'])

    for p in find_all_diag(dialog_history_dir):
        with open(p, encoding="utf-8") as user_file:
            if p.strip().split("\\")[-1] in processed_file:
                print("skipping " + p)
                continue
            print("$$$$$$$$$$$$$")
            print("Processing " + p)
            print("$$$$$$$$$$$$$")
            _, _, profile, attribute = p.strip().split("\\")[-1].split("_")
            profile = profile_inds[int(filter_non_num(profile))]
            attribute_candidate = attribute_candidates[int(filter_non_num(attribute))]
            # read
            parsed_json = json.load(user_file)
            # stat.
            recommendation_consistency = get_stat(parsed_json)
            # chatgpt scoring
            # result = overall_scoring(parsed_json)
            single_scoring(parsed_json, p.strip().split("\\")[-1])
            # result['id'] = p.strip().split("\\")[-1]
            # with jsonlines.open(save_dir + "/" + crs_model + ".jsonl", 'a') as jsonl_file:
            #     jsonl_file.write(result)
            # print(result)
            # file_result = formation(response, single_score, user_feeling_score_tmp, recommendation_consistency, profile, attribute_candidate, p.strip().split("\\")[-1])
            # write
            # wr.writerow(list(file_result))
            # csv_file.flush()


        # wr = csv.writer(csv_file, delimiter='|')



