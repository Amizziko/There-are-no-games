import pandas as pd
import random
import string
import nltk
from nltk.corpus import stopwords
import re

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

IMPORTANT_ABBREVIATIONS = {
    # Gameplay / Modes
    "ai", "fps", "rpg", "rts", "pvp", "pve", "mmo", "coop",
    "solo", "mp", "sp",

    # Tech / Platform
    "vr", "ar",

    # Genres / Subgenres
    "jrpg", "crpg", "rogue", "roguelike", "roguelite",
    "moba", "tps",

    # Perspective / Style
    "2d", "3d",

    # Other common gaming terms
    "dlc", "npc", "rng"
}


def generate_queries(documents):
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    queries = []
    valid_documents = []
    for _, row in documents.iterrows():
        lowercase_doc = row["short_description"].lower()
        lowercase_doc = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", lowercase_doc)
        no_punctuation = lowercase_doc.translate(str.maketrans('','',string.punctuation))
        tokens = no_punctuation.split()
        
        filtered_tokens = [word for word in tokens if word not in stop_words 
                           and (len(word) >= 3 or word in IMPORTANT_ABBREVIATIONS) and not word.isdigit()
            and word.isascii()]
        if not filtered_tokens or len(filtered_tokens) < 2:
            continue
        name_tokens = row["name"].lower().split()
        filtered_tokens += name_tokens

        len_query = random.randint(2, min(6, len(filtered_tokens)))
        query_token = random.sample(filtered_tokens,len_query)
        query = " ".join(query_token)
        queries.append(query)
        valid_documents.append(row["name"] + " " + row["short_description"])
    return queries, valid_documents

def prepare_training(path="steam_games_cleaned.csv", seed=42, training_size=40000,test_size=5000,validation_size=5000):
    random.seed(seed)
    df = pd.read_csv(path)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    #We are using a 40k/5k/5k split for trianing/validating/testing, as anything more will contain a too significant
    # percentage of the dataset, which might lead to overfitting.

    training_documents = shuffled[:training_size]
    validation_documents = shuffled[training_size:training_size+validation_size]
    test_documents = shuffled[training_size+validation_size:training_size+validation_size+test_size]

    our_test_queries,test_documents = generate_queries(test_documents)
    training_queries,training_documents = generate_queries(training_documents)
    validation_queries,validation_documents = generate_queries(validation_documents)

    return training_queries, training_documents, validation_queries, validation_documents, our_test_queries,test_documents 

