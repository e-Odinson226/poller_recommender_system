import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

pd.set_option("max_colwidth", None)


def get_polls_list(polls_csv_path="/data/polls_synthetic.csv"):
    path = Path(__file__).parent.parent.resolve()
    polls = pd.read_csv(str(path) + polls_csv_path)
    return polls


def encode_topics(df):
    topics = df["topic"].str.get_dummies(sep="|")
    df = pd.concat([df, topics], axis=1)
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
                f"error: {df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3], df.iloc[i, 4]}"
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
    print(recommendations)
    # print(similarity_scores_sorted, type(similarity_scores_sorted))
    # recommendations_indices = [
    #    t[0] for t in similarity_scores_sorted[1 : (number_of_recommendations + 1)]
    # ]
    # recommendations_scores = [
    #    t[1] for t in similarity_scores_sorted[1 : (number_of_recommendations + 1)]
    # ]
    # return (df["title"].iloc[recommendations_indices], recommendations_scores)

    return recommendations


if __name__ == "__main__":
    polls = retrieve_data("/data/polls_synthetic.csv")
    polls = encode_topics(polls)
    check_column_type(polls, 4, str)
    tf_idf_matrix = create_tf_idf_matrix(polls, "title")
    cosine_similarity_matrix = calc_cosine_similarity_matrix(
        tf_idf_matrix, tf_idf_matrix
    )

    liked_poll_title = "Do you think cryptocurrency is the future of finance?"
    liked_poll_index = idx_from_title(polls, liked_poll_title)

    liked_polls = []
    recommended_list = recommendations(
        liked_poll_index, polls, cosine_similarity_matrix, 10
    )
    print(f"recommended_list: [{recommended_list}]")

    # print(f"liked poll: [{liked_poll_title}] \nrecommended polls: \n{recommended_list}")
