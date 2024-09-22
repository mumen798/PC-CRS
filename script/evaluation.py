import os
import json
import numpy as np

model_name = 'ma_llama'
path = f'/data/qinpeixin/UserSimulator/iEvaLM/save_10/chat/{model_name}/opendialkg_eval/'

rec_count = 0
rec_success_dialogue = 0
rec_success_1 = 0
rec_success_5 = 0
rec_success_10 = 0
user_accept = 0
true_accept = 0
false_accept = 0
true_reject = 0
low_truthful = 0
user_exp_accept = 0
p_accept = 0

true_truthfulness = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
false_truthfulness = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

success_accept = 0
fail_accept = 0

rec_count_turn = 0
rec_success_dialogue_turn = 0
rec_success_1_turn = 0
rec_success_5_turn = 0
rec_success_10_turn = 0
success_accept_turn = 0
fail_accept_turn = 0

exp_count = 0
total_truthfulness = []

success_truthfulness = []
success_truthfulness_y_3 = []
success_truthfulness_n_3 = []
success_truthfulness_y_4 = []
success_truthfulness_n_4 = []

fail_truthfulness = []
fail_truthfulness_y_3 = []
fail_truthfulness_n_3 = []
fail_truthfulness_y_4 = []
fail_truthfulness_n_4 = []

total_persuasiveness = []

success_persuasiveness = []
success_persuasiveness_y_3 = []
success_persuasiveness_n_3 = []
success_persuasiveness_y_4 = []
success_persuasiveness_n_4 = []

fail_persuasiveness = []
fail_persuasiveness_y_3 = []
fail_persuasiveness_n_3 = []
fail_persuasiveness_y_4 = []
fail_persuasiveness_n_4 = []

over_persuasive_count = 0
over_persuasive_count_when_success = 0
over_persuasive_count_when_fail = 0


def process_dialogue_level_rec(data):
    global rec_count, rec_success_dialogue, rec_success_1, rec_success_5, rec_success_10, user_accept, true_accept, false_accept, success_accept, fail_accept, true_reject, low_truthful, user_exp_accept, p_accept
    last_rec = data[-2]
    last_user = data[-1]
    rec_count += 1

    if last_rec['rec_success_dialogue']:
        rec_success_dialogue += 1
        if last_user['content'].find('[END]') != -1:
            success_accept += 1
    else:
        if last_user['content'].find('[END]') != -1:
            fail_accept += 1
    if last_rec['rec_success@1']:
        rec_success_1 += 1
    if last_rec['rec_success@5']:
        rec_success_5 += 1
    if last_rec['rec_success@10']:
        rec_success_10 += 1

    if last_user['content'].find('[END]') != -1:
        user_accept += 1
        if last_rec['rec_success_dialogue']:
            true_accept += 1
        else:
            false_accept += 1
        if last_rec['ans_type'] == 'exp' and last_rec['truthfulness'] >= 4:
            p_accept += 1

    else:
        if not last_rec['rec_success_dialogue']:
            true_reject += 1

    if last_user['content'].find('[END]') != -1 and last_rec['ans_type'] == 'exp':
        user_exp_accept += 1
        if last_rec['truthfulness'] >= 4:
            low_truthful += 1

        if last_rec['rec_success_dialogue']:
            true_truthfulness[last_rec['truthfulness']] += 1
        else:
            false_truthfulness[last_rec['truthfulness']] += 1


def process_turn_level_rec(turn, next_turn):
    global rec_count_turn, rec_success_dialogue_turn, rec_success_1_turn, rec_success_5_turn, rec_success_10_turn, success_accept_turn, fail_accept_turn
    rec_count_turn += 1
    if turn['rec_success_dialogue']:
        rec_success_dialogue_turn += 1
        if next_turn['content'].find('[END]') != -1:
            success_accept_turn += 1
    else:
        if next_turn['content'].find('[END]') != -1:
            fail_accept_turn += 1
    if turn['rec_success@1']:
        rec_success_1_turn += 1
    if turn['rec_success@5']:
        rec_success_5_turn += 1
    if turn['rec_success@10']:
        rec_success_10_turn += 1


def process_truthfulness(turn, next_turn):
    total_truthfulness.append(turn['truthfulness'])
    t = turn['truthfulness']
    if turn['rec_success_dialogue']:
        success_truthfulness.append(turn['truthfulness'])
        if next_turn['content'].find('[END]') != -1:
            success_truthfulness_y_3.append(t)
        else:
            success_truthfulness_n_3.append(t)

        if turn['persuasiveness']['after'] > 3:
            success_truthfulness_y_4.append(t)
        else:
            success_truthfulness_n_4.append(t)
    else:
        fail_truthfulness.append(turn['truthfulness'])
        if next_turn['content'].find('[END]') != -1:
            fail_truthfulness_y_3.append(t)
        else:
            fail_truthfulness_n_3.append(t)

        if turn['persuasiveness']['after'] > 3:
            fail_truthfulness_y_4.append(t)
        else:
            fail_truthfulness_n_4.append(t)


def process_persuasiveness(turn, next_turn):
    global over_persuasive_count, over_persuasive_count_when_success, over_persuasive_count_when_fail
    persuasiveness = turn['persuasiveness']
    if persuasiveness['after'] <= persuasiveness['true']:
        if persuasiveness['pre'] != persuasiveness['true']:
            p = 1 - ((persuasiveness['true'] - persuasiveness['after']) / (
                    persuasiveness['true'] - persuasiveness['pre']))
            total_persuasiveness.append(p)
            if turn['rec_success_dialogue']:
                success_persuasiveness.append(p)

                if next_turn['content'].find('[END]') != -1:
                    success_persuasiveness_y_3.append(p)
                else:
                    success_persuasiveness_n_3.append(p)

                if turn['persuasiveness']['after'] > 3:
                    success_persuasiveness_y_4.append(p)
                else:
                    success_persuasiveness_n_4.append(p)

            else:
                fail_persuasiveness.append(p)

                if next_turn['content'].find('[END]') != -1:
                    fail_persuasiveness_y_3.append(p)
                else:
                    fail_persuasiveness_n_3.append(p)

                if turn['persuasiveness']['after'] > 3:
                    fail_persuasiveness_y_4.append(p)
                else:
                    fail_persuasiveness_n_4.append(p)
    else:
        over_persuasive_count += 1
        if turn['rec_success_dialogue']:
            over_persuasive_count_when_success += 1
        else:
            over_persuasive_count_when_fail += 1


if __name__ == '__main__':
    not_skip_list = [0, 8, 9, 19, 23, 28, 32, 38, 39, 42, 45, 47]
    for file in os.listdir(path):
        if eval(file.split('_profile')[-1].split('_')[0]) not in not_skip_list:
            continue
        file_path = os.path.join(path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)['simulator_dialog']['context']

            process_dialogue_level_rec(data)

            for i, turn in enumerate(data):
                if turn['role'] == 'assistant':
                    if turn['ans_type'] == 'exp':
                        exp_count += 1
                        process_truthfulness(turn, data[i + 1])
                        process_persuasiveness(turn, data[i + 1])
                    elif turn['ans_type'] == 'rec':
                        process_turn_level_rec(turn, data[i + 1])

    print('dialogue level:')
    print(f'rec_success: {rec_success_dialogue / rec_count}')
    print(f'recall@1: {rec_success_1 / rec_count}')
    print(f'recall@5: {rec_success_5 / rec_count}')
    print(f'recall@10: {rec_success_10 / rec_count}')
    print(f'user_accept: {user_accept / rec_count}')
    print(f'  true_accept: {true_accept / user_accept}')
    print(f'  false_accept: {false_accept / user_accept}')
    print(f'  persuasive: {p_accept / user_accept}')
    print(f'  low_truthful: {low_truthful / user_exp_accept}')
    print(f'  true_reject: {true_reject / (rec_count-user_accept)}')
    print(f'  false_reject: {1-(true_reject / (rec_count-user_accept))}')
    print(true_truthfulness)
    print(false_truthfulness)

    print('')
    # print(f'success_accept: {success_accept / rec_success_dialogue}')
    # print(f'success_reject: {1-(success_accept / rec_success_dialogue)}')
    # print(f'fail_accept: {fail_accept / (rec_count-rec_success_dialogue)}')
    # print(f'fail_reject: {1-(fail_accept / (rec_count-rec_success_dialogue))}')

    print('')

    print('turn level:')
    print(f'rec_success: {rec_success_dialogue_turn / rec_count_turn}')
    print(f'recall@1: {rec_success_1_turn / rec_count_turn}')
    print(f'recall@5: {rec_success_5_turn / rec_count_turn}')
    print(f'recall@10: {rec_success_10_turn / rec_count_turn}')

    print('')
    print(f'success_accept_turn: {success_accept_turn / rec_success_dialogue_turn}')
    print(f'success_reject_turn: {1 - (success_accept_turn / rec_success_dialogue_turn)}')
    print(f'fail_accept_turn: {fail_accept_turn / (rec_count_turn - rec_success_dialogue_turn)}')
    print(f'fail_reject_turn: {1 - (fail_accept_turn / (rec_count_turn - rec_success_dialogue_turn))}')

    print('')

    print(f'truthfulness: {np.mean(total_truthfulness)}')

    print(f'success_truthfulness: {np.mean(success_truthfulness)}')
    # print(f'  acceptance by 3.5: {len(success_truthfulness_y_3) / len(total_truthfulness)}')
    print(f'  truthfulness accept by 3.5: {np.mean(success_truthfulness_y_3)}')
    print(f'  truthfulness reject by 3.5: {np.mean(success_truthfulness_n_3)}')
    # print(f'  acceptance by 4: {len(success_truthfulness_y_4) / len(total_truthfulness)}')
    # print(f'  truthfulness accept by 4: {np.mean(success_truthfulness_y_4)}')
    # print(f'  truthfulness reject by 4: {np.mean(success_truthfulness_n_4)}')

    print(f'fail_truthfulness: {np.mean(fail_truthfulness)}')
    # print(f'  acceptance by 3.5: {len(fail_truthfulness_y_3) / len(total_truthfulness)}')
    print(f'  truthfulness accept by 3.5: {np.mean(fail_truthfulness_y_3)}')
    print(f'  truthfulness reject by 3.5: {np.mean(fail_truthfulness_n_3)}')
    # print(f'  acceptance by 4: {len(fail_truthfulness_y_4) / len(total_truthfulness)}')
    # print(f'  truthfulness accept by 4: {np.mean(fail_truthfulness_y_4)}')
    # print(f'  truthfulness reject by 4: {np.mean(fail_truthfulness_n_4)}')

    print(f'accept avg: {np.mean(success_truthfulness_y_3 + fail_truthfulness_y_3)}')
    print(f'reject avg: {np.mean(success_truthfulness_n_3 + fail_truthfulness_n_3)}')

    print('')

    print(f'persuasiveness: {np.mean(total_persuasiveness)}')

    print(f'success_persuasiveness: {np.mean(success_persuasiveness)}')
    # print(f'  acceptance by 3.5: {len(success_persuasiveness_y_3) / len(total_persuasiveness)}')
    print(f'  persuasiveness accept by 3.5: {np.mean(success_persuasiveness_y_3)}')
    print(f'  persuasiveness reject by 3.5: {np.mean(success_persuasiveness_n_3)}')
    # print(f'  acceptance by 4: {len(success_persuasiveness_y_4) / len(total_persuasiveness)}')
    # print(f'  persuasiveness accept by 4: {np.mean(success_persuasiveness_y_4)}')
    # print(f'  persuasiveness reject by 4: {np.mean(success_persuasiveness_n_4)}')

    print(f'fail_persuasiveness: {np.mean(fail_persuasiveness)}')
    # print(f'  acceptance by 3.5: {len(fail_persuasiveness_y_3) / len(total_persuasiveness)}')
    print(f'  persuasiveness accept by 3.5: {np.mean(fail_persuasiveness_y_3)}')
    print(f'  persuasiveness reject by 3.5: {np.mean(fail_persuasiveness_n_3)}')
    # print(f'  acceptance by 4: {len(fail_persuasiveness_y_4) / len(total_persuasiveness)}')
    # print(f'  persuasiveness accept by 4: {np.mean(fail_persuasiveness_y_4)}')
    # print(f'  persuasiveness reject by 4: {np.mean(fail_persuasiveness_n_4)}')

    print('')
    print(f'accept avg: {np.mean(success_persuasiveness_y_3 + fail_persuasiveness_y_3)}')
    print(f'reject avg: {np.mean(success_persuasiveness_n_3 + fail_persuasiveness_n_4)}')

    print(f'over_persuasive: {over_persuasive_count / exp_count}')
    # print(f'  over_persuasive_when_success: {over_persuasive_count_when_success / (len(success_truthfulness))}')
    # print(f'  over_persuasive_when_fail: {over_persuasive_count_when_fail / (len(fail_truthfulness))}')
    print(f'  over_persuasive_when_success: {over_persuasive_count_when_success / over_persuasive_count}')
    print(f'  over_persuasive_when_fail: {over_persuasive_count_when_fail / over_persuasive_count}')
