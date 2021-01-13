import re
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import json
import pandas as pd
from preprocess import preprocess_sentence


def init_data():
    labeled_ids = pd.read_csv(
        'data/NAACL_SRW_2016.csv', names=['id', 'label'])
    data = []
    with open('data/tweets.txt') as f:
        for line in f:
            data.append(json.loads(line))
    df_api = pd.DataFrame(data)
    # select columns of interest
    columns_of_interest = ['id', 'full_text']
    df_api = df_api[columns_of_interest]

    # print(labeled_ids.shape[0])
    # print(df_api.shape[0])
    df = df_api.set_index("id").join(labeled_ids.set_index("id"))
    # print(df.shape[0])

    df.dropna(how='any', inplace=True)
    df = df[df["label"] != "racism"]

    # writes to train.dec all the tabs, i.e Offensive for all the lines
    # writes all the words after being preprocess on each line.
    vocab = []
    r_count = 0
    s_count = 0
    stupid_count = 0
    with open("preprocessed/train.dec", "w") as dec:
        with open("preprocessed/train.enc", "w") as enc:
            is_first_line = True
            for index, row in df.iterrows():
                label = df.at[index, "label"]
                if not isinstance(label, str):
                    # continue
                    stupid_count += 1
                    print(label)
                    label = label[0]
                    print(label)
                if label == "racism":
                    r_count += 1
                if label == "sexism":
                    s_count += 1
                words = preprocess_sentence(row["full_text"])
                if words == "":
                    continue
                df.at[index, "full_text"] = words
                # word_tokens = nltk.word_tokenize(words)
                vocab.extend(words.split(" "))
                if is_first_line:
                    is_first_line = False
                else:
                    enc.write("\n")
                    dec.write("\n")
                enc.write(words)
                dec.write(label)
    print(r_count)
    print(s_count)
    print(stupid_count)
    sorted_vocab = sorted(list(set(vocab)))
    with open('config.py', 'a') as cf:
        cf.write("VOCAB_SIZE = "+str(len(sorted_vocab)) + "\n")
    # writes the sorted vocab to file and adds a <pad> token on the first line because of the padding that is done later
    with open("preprocessed/vocab.enc", "w") as file:
        is_first_line = True
        file.write("<pad>\n")
        for item in sorted_vocab:
            if not is_first_line:
                file.write("\n")
            file.write(item)
            is_first_line = False
    # df.dropna(inplace=True)
    df.to_csv("preprocessed/labeled_tweets.csv", index=None)
    # writes the vocab of decoder.
    with open("preprocessed/vocab.dec", "w") as f:
        f.write("none"+"\n")
        f.write("sexism"+"\n")

    write_enc_ids()
    write_dec_ids()


def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}


def write_enc_ids():
    words, word_set = load_vocab("preprocessed/vocab.enc")
    with open("preprocessed/train_ids.enc", "w") as ids:
        is_first_line = True
        with open("preprocessed/train.enc", "r") as wor:
            for line in wor:
                if is_first_line:
                    is_first_line = False
                else:
                    ids.write("\n")
                word_tokens = nltk.word_tokenize(line)
                for token in word_tokens:
                    if token in words:
                        ids.write(str(word_set[token])+" ")


def write_dec_ids():
    words, word_set = load_vocab("preprocessed/vocab.dec")
    with open("preprocessed/train_ids.dec", "w") as ids:
        with open("preprocessed/train.dec", "r") as wor:
            is_first_line = True
            for line in wor:
                if is_first_line:
                    is_first_line = False
                else:
                    ids.write("\n")
                word_tokens = nltk.word_tokenize(line)
                for token in word_tokens:
                    if token in words:
                        ids.write(str(word_set[token])+" ")


def load_enc_ids():
    res = []
    with open("preprocessed/train_ids.enc", "r") as ids:
        index = 0
        for line in ids:
            index += 1
    return res


def load_dec_ids():
    res = []
    with open("preprocessed/train_ids.dec", "r") as ids:
        for line in ids:
            res.append(int(line.strip()))
    return res


def ids_to_words(ids):
    res = ""
    vocab, s = load_vocab("preprocessed/vocab.enc")
    for item in ids:
        if item == 0:
            continue
        res += vocab[item]
        res += " "
    return res


def words_to_ids(words):
    res = []
    v, w = load_vocab("preprocessed/vocab.enc")
    for word in nltk.word_tokenize(words):
        res.append(w[word])
    return res


def labels_to_ids(labels):
    res = []
    v, w = load_vocab("preprocessed/vocab.dec")
    for label in labels:
        res.append(w[label])
    return res


def id_to_label(id):
    print(id)
    if id[0] > 0.5:
        return "non-offensive"
    elif id[0] > 0.5:
        return "sexism"


if __name__ == '__main__':
    init_data()
