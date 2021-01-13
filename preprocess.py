from nltk.corpus import stopwords
import re
import spacy
import json
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
import preprocessor as p

from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
# Load English tokenizer, tagger,
# parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")


def preprocess_sentence(sentence):
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY,
                  p.OPT.MENTION, p.OPT.NUMBER)
    sentence = p.clean(sentence)
    #sentence = remove_words_multiple_chars(sentence)
    sentence = remove_non_char(sentence)
    sentence = remove_rt(sentence)
    #sentence = remove_stopwords(sentence)
    # sentence = lemmatize(sentence)
    sentence = " ".join(sentence.split())
    sentence = sentence.lower()
    # print(sentence)
    return sentence


def removeNan(words):
    if words != words:
        print("yy")
        return ""


def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(words)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)

# bruker nlp lemmatizer.


def lemmatize(words):
    doc = nlp(words)
    doc2 = ""
    doc2 = " ".join([token.lemma_ for token in doc])
    patterns = "[-]\w+[-]"
    kk = re.sub(patterns, "", doc2)
    kk = " ".join(kk.split())
    return kk

# Fjerner alt som ikke er a til z eller mellomrom.


def remove_non_char(words):
    pattern = '[^a-zA-Z@ ]'
    # First parameter is the replacement, second parameter is your input string
    return re.sub(pattern, "", words)

# fjerner RT som hver linje begynner på "retweet"


def remove_rt(words):
    pattern = re.compile(r"^(RT)")
    return pattern.sub("", words)

# Fjerner bokstaver hvis det er fler enn 2 av samme på rad. altså heeey blir til heey


def remove_words_multiple_chars(words):
    # pattern = r"([a-z])\1{3,}"
    pattern = re.compile("(\\w)\\1{2,}")
    kk = pattern.sub(r"\1", words)
    # kk = re.sub(pattern, "", words)
    return kk


def isNaN(num):
    return num != num
