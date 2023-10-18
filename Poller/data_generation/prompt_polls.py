# %%
import os
import openai
import pickle
import json


choose_gpt_api = str(
    input(
        "OpenAI API-KEY \noptions:\nr: read from env vars; should be exported as (OPENAI_API_KEY)\nk: enter the key itself\nchoose option[(r)/k]:"
    )
)
if choose_gpt_api == "k":
    openai.api_key = str(input("Enter OpenAI API-KEY: "))
else:
    print("Reading API-KEY from environment variables...")
    openai.api_key = os.getenv("OPENAI_API_KEY")
models = openai.Model.list()
print(models.data[0].id)

# %%
# *if new combination list required...
topics = [
    "Tech",
    "Sport",
    "Politics",
    "Science",
    "Activity",
    "Art",
    "Beauty",
    "Economy",
    "Fashion",
    "Food&Drinks",
    "Fun",
    "Health",
    "Music",
    "Pets",
    "Travel",
    "Movies & TV shows",
    "Home Décor",
]

answer = []

prompt = f"""
    generate 30 distinct, meaningful, and relatable combination of topics,
    for example: [travel, activity] or [Fun, Art],
    from this list of topics:    
    {topics}
    
    only and only use topics from presented topic list,
    make sure you do not repeat any combination,
    give your answer inside a json.

"""

chosen_gpt_model = str(input("GPT model [(3)/4]:"))


if chosen_gpt_model == "4":
    gpt_model = "gpt-4"
else:
    gpt_model = "gpt-3.5-turbo"


get_topic = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo",
    model=gpt_model,
    messages=[
        {"role": "user", "content": prompt},
    ],
)


comb_dic = json.loads(get_topic["choices"][0]["message"].content)  # .split('\n')
get_topic["choices"][0]["message"].content

# %%
combinations = comb_dic.get("combinations")
combinations = list(comb_dic.values())
print(f"generated combinations:{combinations}")


# %%
all_polls = []

for comb in combinations:
    prompt = f"""\
        Create a list of at least 100 meaningful, natural, and profound "polls" inside JSONs that seamlessly blend topics of "{comb[0]}" and "{comb[1]}" within the same question.\
            with the following fields: question(string), options(a list of options), topics(a list of topics which where geiven to you).\
            example:
        {{
            question: "What type of headphones do you prefer?",
            options:["Wireless earbuds", "Over-ear headphones", "On-ear headphones"],
            topics:["Tech", "Activity"]
        }}
        """
    print(
        f"\n-----------------------------\nPrompting: for polls with topic of {comb}..."
    )

    gen_polls = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model=gpt_model,
        messages=[{"role": "user", "content": prompt}],
    )

    print(f"Prompt response retreived successfully.")
    resp = gen_polls["choices"][0]["message"]["content"]
    polls = [r for r in resp.split("\n")]

    with open(f"{comb[0]}_{comb[1]}.pickle", "wb") as file:
        pickle.dump({f"{comb[0]}_{comb[1]}": polls}, file)
    all_polls.append({f"{comb[0]}_{comb[1]}": polls})

# %%


jsons_dir = str(os.getcwd()) + "/jsons"

print(f"extract retrieved polls to json files inside:{jsons_dir}")
polls = []

for filename in os.listdir(jsons_dir):
    file_path = os.path.join(jsons_dir, filename)
    new_file_path = file_path

    # if 'pickle' in filename:
    new_filename = filename.replace(".pickle", "")

    try:
        # print(f"renamin: {file_path}")
        # os.rename(filename, new_filename)
        os.rename(file_path, new_file_path)
        # print(f"File '{filename}' has been renamed to '{new_filename}'")
        with open(new_file_path, "rb") as file:
            loaded_object = json.load(file)
            polls.append(loaded_object)
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except FileExistsError:
        print(f"File '{new_filename}' already exists.")
    except Exception as e:
        print(f"renamin: {file_path}")
        print(f"An error occurred: {e}")

with open("polls.json", "w") as outfile:
    json.dump(polls, outfile)
    print(
        f" [polls.json] generated, containing a list of all generated polls WITH a hierarchy structure. "
    )


# %%
polls_list = []
for cat in polls:
    cat_title = list(cat.keys())[0]
    for poll in cat.get(cat_title):
        polls_list.append(poll)


with open("list_of_polls.json", "w") as outfile:
    json.dump(polls_list, outfile)
    print(
        f" [list_of_polls.json] generated, containing a list of all generated polls without a hierarchy structure. "
    )
    # jsoned = json.dumps(file)
