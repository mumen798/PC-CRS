import csv

attr2idx = {"file_name": 0, "overall_performance": 4, "user_satisfaction": 5,
            "relevance_score": 8, "quality_score": 9, "manner_score": 10,
            "humanlike_score": 11, "explanation_score": 12, "consistency": 13,
            "dialogue_success": 16, "rec_success": 17, "chatgpt_success": 18}


def parse_filename(filename):
    sub_filenames = filename.strip(".json").split("_")
    profile_idx = sub_filenames[2].split("profile")[1]
    attr_idx = sub_filenames[3].split("attribute")[1]

    return int(profile_idx), int(attr_idx)


def get_score_matrix(processed_file, attr_name):
    score_matrix = []
    for i in range(48):
        init_score = [-1 for j in range(19)]
        score_matrix.append(init_score)
    for line in processed_file:
        profile_idx, attr_idx = parse_filename(line[0])
        if attr_name == "dialogue_success" or attr_name == "rec_success" or attr_name == "chatgpt_success":
            score_matrix[profile_idx][attr_idx] = int(line[attr2idx[attr_name]][-2])
        else:
            score_matrix[profile_idx][attr_idx] = line[attr2idx[attr_name]]

    return score_matrix


def main():
    result_file = "D:\\code\\UserSimulator_A100\\iEvaLM\\scoring_unicrs\\unicrs.csv"
    processed_file = []
    with open(result_file, "r", encoding="utf-8") as csv_file:
        for line in csv_file:
            if len(line) < 3:
                continue
            processed_file.append(line.strip().split("|"))
    score_matrix = get_score_matrix(processed_file, "user_satisfaction")
    analysis_file = "D:\\code\\UserSimulator_A100\\iEvaLM\\scoring_unicrs\\tmp.csv"
    with open(analysis_file, "w", encoding="utf-8", newline="") as file:
        wr = csv.writer(file)
        for line in score_matrix:
            wr.writerow(line)
    pass


if __name__ == "__main__":
    main()
