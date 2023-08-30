import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from collections import Counter
import json


def get_polls_list(polls_csv_path="/data/polls_synthetic.csv"):
    path = Path(__file__).parent.parent.resolve()
    polls = pd.read_csv(str(path) + polls_csv_path)
    return polls


def get_polls_list_json(polls_json_path="/data/elas_polls.json"):
    path = Path(__file__).parent.parent.resolve()
    polls = pd.read_csv(str(path) + polls_csv_path)
    return polls


# def get_polls_list(polls_json_path="/data/elas_polls.json"):
#    path = Path(__file__).parent.parent.resolve()
#    print(path)
#    polls = pd.read_json(str(path) + polls_json_path)
#    return polls


def encode_topics(df):
    # topics = df["topics"].str.get_dummies(sep=",")
    # topics = df["topics"].apply( topicfor topic in topics  )
    one_hot_encoded = (
        pd.get_dummies(df["topics"].apply(pd.Series).stack()).groupby(level=0).sum()
    )
    df = pd.concat([df, one_hot_encoded], axis=1)
    # print(df)
    return df


def set_index(df, index_column="poll_ID"):
    df.set_index(index_column, inplace=True)
    return df


def reset_index(df):
    df.reset_index()
    return df


def check_column_type(df, column_name, check_type):
    column_index = df.columns.get_loc(column_name)
    for i in range(len(df)):
        if not isinstance(df.iloc[i, column_index], check_type):
            print(
                f"error: {df.iloc[i, 0], df.iloc[i, 1],df.iloc[i, 2], df.iloc[i, 3], df.iloc[i, 4]}"
            )


def create_tf_idf_matrix(df, column):
    tf_idf = TfidfVectorizer(stop_words="english")
    tf_idf_matrix = tf_idf.fit_transform(df[column])
    return tf_idf_matrix


def calc_cosine_similarity_matrix(tf_idf_matrix_1, tf_idf_matrix_2):
    cosine_similarity_matrix = cosine_similarity(tf_idf_matrix_1, tf_idf_matrix_2)
    return cosine_similarity_matrix


def idx_from_title(df, title):
    return df[df["title"] == title].index.values[0]


def title_from_idx(df, idx):
    return df[df.index == idx]


def gen_recommendations(index, df, cosine_similarity_matrix, number_of_recommendations):
    # index = idx_from_title(df, original_title)
    similarity_scores = list(enumerate(cosine_similarity_matrix[index]))
    similarity_scores_sorted = sorted(
        similarity_scores, key=lambda x: x[1], reverse=True
    )

    recommendations_indices = [
        t[0] for t in similarity_scores_sorted[1 : (number_of_recommendations + 1)]
    ]
    recommendations = list(df["title"].iloc[recommendations_indices])
    # print(recommendations)
    # print(similarity_scores_sorted, type(similarity_scores_sorted))
    # recommendations_indices = [
    #    t[0] for t in similarity_scores_sorted[1 : (number_of_recommendations + 1)]
    # ]
    # recommendations_scores = [
    #    t[1] for t in similarity_scores_sorted[1 : (number_of_recommendations + 1)]
    # ]
    # return (df["title"].iloc[recommendations_indices], recommendations_scores)

    return recommendations


def gen_rec_from_list_of_polls(
    interacted_polls, polls, cosine_similarity_matrix, number_of_recommendations
):
    recommendations = []
    for poll_id in interacted_polls:
        # index = idx_from_title(df, original_title)
        similarity_scores = list(enumerate(cosine_similarity_matrix[poll_id]))
        similarity_scores_sorted = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True
        )

        recommendations_indices = [
            t[0] for t in similarity_scores_sorted[1 : (number_of_recommendations + 1)]
        ]
        recs = list(polls["id"].iloc[recommendations_indices])
        # print(f"recommended polls for {poll_id} are:{recs}")
        recommendations.append(recs)

    flattened_recommendations = [
        item for sublist in recommendations for item in sublist
    ]
    flattened_recommendations = Counter(flattened_recommendations)
    n_most_recommended = flattened_recommendations.most_common(
        number_of_recommendations
    )
    n_most_recommended = [t[0] for t in n_most_recommended]
    print(n_most_recommended)

    return n_most_recommended


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)

    path = Path(__file__).parent.parent.parent.resolve()
    path = str(path) + "/data/elas_polls.json"
    polls_list = []
    # polls = pd.read_json(str(path) + "/data/elas_polls.json")
    with open(path, "r") as infile:
        polls = json.load(infile)
        for poll in polls:
            # poll = poll["_source"]
            # print(f"poll:\n{poll}")
            polls_list.append(poll)

    polls = pd.DataFrame.from_records(polls_list)

    polls = encode_topics(polls)
    print(polls)
    # check_column_type(polls, 4, str)
    tf_idf_matrix = create_tf_idf_matrix(polls, "question")
    cosine_similarity_matrix = calc_cosine_similarity_matrix(
        tf_idf_matrix, tf_idf_matrix
    )

#
# liked_poll_title = "Do you think cryptocurrency is the future of finance?"
# liked_poll_index = idx_from_title(polls, liked_poll_title)
#
# liked_polls = []
# recommended_list = recommendations(
#    liked_poll_index, polls, cosine_similarity_matrix, 10
# )
# print(f"recommended_list: [{recommended_list}]")

# print(f"liked poll: [{liked_poll_title}] \nrecommended polls: \n{recommended_list}")
