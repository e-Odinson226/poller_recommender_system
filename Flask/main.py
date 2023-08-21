from flask import Flask, redirect, url_for, jsonify, request
from rec_sys import *

app = Flask(__name__)


# @app.route("/get_recom", methods=["POST"])
@app.route("/get_recom")
def recom():
    # 1.Get list from the ElasticSearch query TODO. ---------------------------------------------
    polls = get_polls_list("/data/polls_synthetic.csv")

    # 2.encode topics to column. ---------------------------------------------
    polls = encode_topics(polls)

    # 3.check each column's datatype. ---------------------------------------------
    # check_column_type(polls, "topic", str)
    # check_column_type(polls, "option", str)
    # check_column_type(polls, "author_ID", int)

    # 4. get the list of polls, which the user has interacted with. TODO ---------------------------------------------
    user_interacted_polls = get_polls_list("/data/user_interacted_polls.csv")

    # 4.Calculate TF-IDF matrix based on "polls topics" for [Polls-to-Polls] or [Polls-to-UserProfile]
    polls_tf_idf_matrix = create_tf_idf_matrix(polls, "topic")
    cosine_similarity_matrix = calc_cosine_similarity_matrix(
        polls_tf_idf_matrix, polls_tf_idf_matrix
    )

    # liked_poll_title = "Do you think cryptocurrency is the future of finance?"
    # liked_poll_index = idx_from_title(polls, liked_poll_title)

    liked_polls = []
    recommended_list = gen_rec_from_list_of_polls(
        user_interacted_polls["poll_ID"], polls, cosine_similarity_matrix, 10
    )
    # print(f"recommended_list: [{recommended_list}]")

    # print(f"liked poll: [{liked_poll_title}] \nrecommended polls: \n{recommended_list}")
    # ---------------------------------------

    # ---------------------------------------

    # return jsonify({"recommended_polls": recommended_list})

    print(f"type(recommended_list): {type(list(recommended_list[0]))}")
    return list(recommended_list)

    # print(f"liked poll: [{liked_poll_title}] \nrecommended polls: \n{recommended_list}")


if __name__ == "__main__":
    app.run(debug=True)
