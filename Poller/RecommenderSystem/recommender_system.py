import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from pathlib import Path
from collections import Counter
import json
import nltk

nltk.download("punkt")
nltk.download("stopwords")
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)

tf_idf = TfidfVectorizer(stop_words="english")


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


def preprocess_text(text):
    tokens = nltk.tokenize.word_tokenize(text)
    # tokens = [word.lower() for word in tokens if type(word) is str]
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word not in string.punctuation]
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    processed_text = " ".join(tokens)

    return processed_text


def preprocess_list(field_list):
    ret_list = []
    stop_words = set(nltk.corpus.stopwords.words("english"))
    for item in field_list:
        tokens = nltk.tokenize.word_tokenize(item)
        # tokens = [word.lower() for word in tokens if type(word) is str]
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if word not in string.punctuation]
        tokens = [word for word in tokens if word not in stop_words]
        processed_text = " ".join(tokens)
        ret_list.append(processed_text)

    return ret_list


def create_tf_idf_matrix(df, column):
    # print(f"{df[column]} is {df[column].dtype} and {df[column].dtype is list} {list}: ")
    df[column] = df[column].apply(lambda x: " ".join(x))
    df[column] = df[column].apply(preprocess_text)

    return tf_idf.fit_transform(df[column])


def create_souped_tf_idf_matrix(df):
    df["topics"] = df["topics"].apply(preprocess_list)
    df["question"] = df["question"].apply(preprocess_text)

    # Create a new soup feature
    df["soup"] = df.apply(create_soup, axis=1)

    return tf_idf.fit_transform(df["soup"])


def create_soup(df):
    res = (
        df["question"]
        + " "
        + " ".join(df["options"])
        + " "
        + (4 * (" " + " ".join(df["topics"])))
    )
    # print(f"-----------------------------------\n* Processing: [{ }]")
    return res


def calc_cosine_similarity_matrix(tf_idf_matrix_1, tf_idf_matrix_2):
    return cosine_similarity(tf_idf_matrix_1, tf_idf_matrix_2)


def id_to_index(df, search_id):
    result = df[df["id"] == str(search_id)].index.values[0]
    print(result)

    if len(result) > 0:
        return result
    else:
        return None


def id_to_index2(df, id):
    try:
        if any(df["id"] == str(id)):
            # df.to_csv("df.csv", index=False)
            # print(
            #    f"---------------\nFound {id} at {df[df['id'] == str(id)].index.values[0]}"
            # )
            # print(f"\nWhich is equal to:\n{df[df['id'] == str(id)]}")
            return df[df["id"] == str(id)].index.values[0]

    except IndexError as e:
        print(f"erorrrrrrrrrrrrr:")
        print(f"{str(id)}")
        print(f"{df['id']==str(id)}")


def title_from_idx(df, idx):
    return df[df.index == idx]


def gen_recommendations(
    index,
    df,
    cosine_similarity_matrix,
    number_of_recommendations,
):
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
    interacted_polls,
    filtered_polls_df,
    cosine_similarity_matrix,
    number_of_recommendations,
):
    recommendations = []
    for poll_id in interacted_polls:
        index = id_to_index2(filtered_polls_df, poll_id)
        if index is not None:
            similarity_scores = list(enumerate(cosine_similarity_matrix[index]))
            similarity_scores_sorted = sorted(
                similarity_scores, key=lambda x: x[1], reverse=True
            )

            recommendations_indices = [
                t[0]
                for t in similarity_scores_sorted[1 : (number_of_recommendations + 1)]
            ]
            recs = list(filtered_polls_df["id"].iloc[recommendations_indices])

            # Filter out polls that have already been interacted with
            filtered_recs = [poll for poll in recs if poll not in interacted_polls]

            recommendations.append(filtered_recs)

        else:
            pass

        # index = id_to_index(polls, poll_id)
        # print(f"cosine_similarity_matrix:{len(cosine_similarity_matrix)}")
        # print(f"index:{index} | id:{poll_id}")

    flattened_recommendations = [
        item for sublist in recommendations for item in sublist
    ]
    flattened_recommendations = Counter(flattened_recommendations)
    n_most_recommended = flattened_recommendations.most_common(
        number_of_recommendations
    )
    n_most_recommended = [t[0] for t in n_most_recommended]
    # print(n_most_recommended)

    return n_most_recommended


def is_valid_limitations(limitations):
    if isinstance(limitations, dict):
        return (
            "allowedLocations" in limitations
            and "allowedGender" in limitations
            and "allowedAgeRange" in limitations
        )
    return False


# Function to filter polls with user-defined limitations
def filter_polls(row, user_limitations):
    # print(isinstance(row.get("pollLimitations"), dict))

    # if isinstance(row.get("pollLimitations"), dict) and all(
    #    k in user_limitations for k in ["Location", "Gender", "Age"]
    # ):
    if pd.notna(row.get("pollLimitations")) and all(
        k in user_limitations for k in ["Location", "Gender", "Age"]
    ):
        user_location = user_limitations.get("Location")

        allowed_locations = row.get("pollLimitations").get("allowedLocations")
        if len(allowed_locations) == 0 or any(
            user_location == loc for loc in allowed_locations
        ):
            allowed_gender = row["pollLimitations"]["allowedGender"]
            user_gender = user_limitations["Gender"]
            if allowed_gender == "All" or allowed_gender == user_gender:
                allowed_age_range = row["pollLimitations"]["allowedAgeRange"]
                user_age = user_limitations["Age"]
                if (
                    allowed_age_range["minimumAge"]
                    <= user_age
                    <= allowed_age_range["maximumAge"]
                ):
                    # print("All conditions met. Returning True")
                    return True
    #            else:
    #                print("Age condition not met.")
    #                return False
    #        else:
    #            print("Gender condition not met.")
    #            return False
    #
    #    else:
    #        print("No allowedLocations found.")
    #        return False
    # else:
    #    print("Invalid limitations or missing keys in user_limitations.")
    #    print(f"row:{row['id']}")
    #    return False
    return False


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
