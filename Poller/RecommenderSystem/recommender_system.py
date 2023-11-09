import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from pathlib import Path
from collections import Counter
import json
import nltk
import requests
from datetime import datetime, timedelta


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
    # if tf_idf_matrix_1 is not None and tf_idf_matrix_2 is not None:
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


def gen_rec_from_list_of_polls_df(
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

    filtered_df = filtered_polls_df[filtered_polls_df["id"].isin(n_most_recommended)]

    order_dict = {id: idx for idx, id in enumerate(n_most_recommended)}

    # Sort the filtered DataFrame based on the order
    filtered_df["order"] = filtered_df["id"].map(order_dict)
    filtered_df = filtered_df.sort_values("order")

    # Drop the 'order' column if not needed
    filtered_df = filtered_df.drop(columns=["order"])

    # Reset the index if needed
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df


def is_valid_limitations(limitations):
    if isinstance(limitations, dict):
        return (
            "allowedLocations" in limitations
            and "allowedGender" in limitations
            and "allowedAgeRange" in limitations
        )
    return False


def is_within_x_days_liifetime(timestamp):
    try:
        # Convert the timestamp to a datetime object
        time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")

        # Calculate the time difference
        time_difference = datetime.now() - time

        return True if time_difference <= timedelta(days=10) else False
    except ValueError:
        # If the timestamp doesn't match the expected format, return False
        return False


def filter_timestamp(timestamp):
    try:
        # Convert the timestamp to a datetime object
        time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")

        # Calculate the time difference
        time_difference = datetime.now() - time

        return time_difference
    except ValueError:
        # If the timestamp doesn't match the expected format, return None
        return None


def split_by_days(polls_df, days=10):
    filtered_df = polls_df[polls_df["createdAt"].apply(filter_timestamp).notna()]

    older_than_x_days = filtered_df[
        filtered_df["createdAt"].apply(filter_timestamp) >= timedelta(days=days)
    ]
    newer_than_x_days = filtered_df[
        filtered_df["createdAt"].apply(filter_timestamp) < timedelta(days=days)
    ]

    # Reset the index if needed
    older_than_x_days = older_than_x_days.reset_index(drop=True)
    newer_than_x_days = newer_than_x_days.reset_index(drop=True)

    return older_than_x_days, newer_than_x_days


def has_valid_date(date_str):
    # Convert the date string to a datetime object
    # date = pd.to_datetime(date_str)
    date = pd.to_datetime(date_str, utc=True).replace(tzinfo=None)

    # Get the current timestamp as a datetime object
    current_time = datetime.now()

    # Compare the date with the current timestamp
    return date > current_time


def remove_duplicates(input_list):
    seen = set()
    result = []

    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def validate_polls(polls_df, df_name, verbose=True):
    valid_polls = polls_df.loc[
        polls_df["valid"] & polls_df["endedAt"].apply(has_valid_date)
    ]

    invalid_polls = polls_df[
        ~polls_df["valid"] | ~polls_df["endedAt"].apply(has_valid_date)
    ]

    if verbose:
        print(f"valid_{df_name}_polls: {len(valid_polls)}")
        print(f"expired_{df_name}_polls: {len(invalid_polls)}")

    return valid_polls, invalid_polls
    return pd.concat([valid_polls, invalid_polls], ignore_index=False)


def order(recommended_polls_df=None, trend_polls_df=None, verbose=True):
    if recommended_polls_df is None:
        valid_trend_polls, invalid_trend_polls = validate_polls(
            trend_polls_df, "trend", verbose
        )
        recommended_polls_df = pd.concat(
            [
                valid_trend_polls,
                invalid_trend_polls,
            ],
            ignore_index=False,
        )

    elif trend_polls_df is None:
        valid_recommended_polls, invalid_recommended_polls = validate_polls(
            recommended_polls_df, "recommended", verbose
        )
        recommended_polls_df = pd.concat(
            [
                valid_recommended_polls,
                invalid_recommended_polls,
            ],
            ignore_index=False,
        )

    elif trend_polls_df is not None and recommended_polls_df is not None:
        valid_recommended_polls, invalid_recommended_polls = validate_polls(
            recommended_polls_df, "recommended", verbose
        )
        valid_trend_polls, invalid_trend_polls = validate_polls(
            trend_polls_df, "trend", verbose
        )

        recommended_polls_df = pd.concat(
            [
                valid_recommended_polls,
                valid_trend_polls,
                invalid_recommended_polls,
                invalid_trend_polls,
            ],
            ignore_index=False,
        )

    recommended_polls_df = recommended_polls_df.reset_index(drop=True)
    recommended_polls_df = recommended_polls_df["id"].tolist()
    recommended_polls_df = remove_duplicates(recommended_polls_df)

    return recommended_polls_df


def list_to_df(polls_list, polls_df):
    # Filter the DataFrame based on the id_list
    filtered_df = polls_df[polls_df["id"].isin(polls_list)]

    # Create a dictionary to preserve the order
    order_dict = {id: idx for idx, id in enumerate(polls_list)}

    # Sort the filtered DataFrame based on the order
    filtered_df["order"] = filtered_df["id"].map(order_dict)
    filtered_df = filtered_df.sort_values("order")

    # Drop the 'order' column if not needed
    filtered_df = filtered_df.drop(columns=["order"])

    # Reset the index if needed
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df


def filter_polls(row, user_limitations):
    if (
        row["pollType"] == "Public"
        and isinstance(row.get("pollLimitations"), dict)
        and all(k in user_limitations for k in ["Location", "Gender", "Age"])
        # and is_within_x_days_liifetime(row["createdAt"])
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
                    return True

    return False


def get_allowed_private_polls(
    params,
    # url="https://dev.pollett.io/api/Recommend/Polls/GetPrivatePollThatUserCanSee",
    url,
):
    # API URL
    # url = "https://dev.pollett.io/api/Recommend/Polls/GetPrivatePollThatUserCanSee"

    # Parameters
    # params = {"userId": "bbe64b34-ba34-4fbd-a62f-e6c84c0423b4"}

    # Send a GET request to the API
    response = requests.get(url, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        # allowed_polls_list = response.json().get("data")
        return response.json().get("data")
        # Process the data as needed

    else:
        # Handle the error
        print(f"Request failed with status code {response.status_code}")
        print(response.text)


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
