import json
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import re


stop_words = {"the", "a", "an", "and", "or", "but", "if", "in", "on", "of", "for", "with", "is", "are", "he", "she", "it", "they", "we", "your", "me", "this", "that"}

def tokenize(text):

    text = re.sub(r"'s\b", "", text)

    tokenizer = RegexpTokenizer(r"[A-Za-z]+(?:\.[A-Za-z]+)*")
    tokens = tokenizer.tokenize(text)
    return tokens


def preprocess(tokens):

    #normalization
    tokens_lower = [token.lower() for token in tokens]


    #stemming
    stemmer = SnowballStemmer("english")
    tokens_stem = [stemmer.stem(token) for token in tokens_lower]

    #deleting stop words
    processed_text = [token for token in tokens_stem if token not in stop_words]

    return processed_text



def compare_goldstandard(preproccesed_text, gold):
    precision = len(set(preproccesed_text) & set(gold)) / len(preproccesed_text)
    recall = len(set(preproccesed_text) & set(gold)) / len(gold)

    print(f"Precision: {precision}  Recall: {recall} \n")



if __name__ == "__main__":
    with open("402715255.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for text in data:
        raw_text = text["original_text"]
        gold = text["goldstandard"]
        tokens = tokenize(raw_text)
        preproccesed_text = preprocess(tokens)


        print(preproccesed_text) 
        print(gold)
        compare_goldstandard(preproccesed_text, gold )
