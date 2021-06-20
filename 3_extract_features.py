#!/anaconda3/bin/python3.7
# -*- coding: UTF-8 -*-
import re
import pandas as pd
from conllu import *
import language_tool_python
from normalization import *


# ----------load conll file---------

def get_sentences_tagged(conll_file="../1_data/3_data_tagged/data_tagged.conll"):
    """
    input: str(path_to_conll_file)
    ouput: list(tagged sentences)
    """
    with open(conll_file, "r", encoding="utf-8") as f:
        conll_sentences = re.split("\n\n", f.read())
    return conll_sentences

def parse_conll_to_dict(conll_text):
    """
    stock conll text in dictionary
    input: str(conll_text_one_sentence)
    output: dict(token
    {
        'id': 1,
        'form': 'The',
        'lemma': 'the',
        ...
    })
    """
    return parse(conll_text)

# ----------lexical features---------

# lexical diversity
def get_ttr(conll_sentence):
    """
    input: str(conll_sentence)
    output: float(type_token_ratio)
    """
    forms = list()
    sentence_parsed = parse_conll_to_dict(conll_sentence)
    for i in range(len(sentence_parsed)):
        sentence = sentence_parsed[i]
        for j in range(len(sentence)):
            token = sentence[j]
            if token["upostag"] != "PONCT":
                forms.append(token["form"])
    try:
        return len(set(forms))/len(forms)
    except:
        return 0

# keyword1 "ça"
def if_ca(conll_sentence):
    """
    detect if "ça" is present in sentence
    input: str(conll_sentence)
    output: int(1/0)
    """
    test = 0
    sentence_parsed = parse_conll_to_dict(conll_sentence)
    for i in range(len(sentence_parsed)):
        sentence = sentence_parsed[i]
        for j in range(len(sentence)):
            token = sentence[j]
            if token["form"] == "ça":
                test = 1
    return test

# keyword2 "cela"
def if_cela(conll_sentence):
    """
    detect if "cela" is present in sentence
    input: str(conll_sentence)
    output: int(1/0)
    """
    test = 0
    sentence_parsed = parse_conll_to_dict(conll_sentence)
    for i in range(len(sentence_parsed)):
        sentence = sentence_parsed[i]
        for j in range(len(sentence)):
            token = sentence[j]
            if token["form"] == "ça":
                test = 1
    return test

# keywords3 SMS
def if_sms(conll_sentence):
    """
    detect if there are sms words in sentence
    input: str(conll_sentence)
    output: int(1/0)
    """
    test = 0
    sms_list = list(get_data('../sms.csv')['sms'])
    sentence_parsed = parse_conll_to_dict(conll_sentence)
    for i in range(len(sentence_parsed)):
        sentence = sentence_parsed[i]
        for j in range(len(sentence)):
            token = sentence[j]
            if any(token["form"] == w for w in sms_list):
                test = 1
    return test

# keywords4 emoji
def if_emoji(conll_sentence):
    """
    detect if there is emoji in sentence
    input: str(conll_sentence)
    output: int(1/0)
    """
    test = 0
    sentence_parsed = parse_conll_to_dict(conll_sentence)
    for i in range(len(sentence_parsed)):
        sentence = sentence_parsed[i]
        for j in range(len(sentence)):
            token = sentence[j]
            if token["form"] == "emoji":
                test = 1
    return test
# ----------morphosyntactic features---------

# tense diversity
def get_tense_diversity(conll_sentence):
    """
    input: str(conll_sentence)
    output: float(nbrtense_nbrverb_ratio)
    """
    tense = list()
    verbs = list()
    sentence_parsed = parse_conll_to_dict(conll_sentence)
    for i in range(len(sentence_parsed)):
        sentence = sentence_parsed[i]
        for j in range(len(sentence)):
            token = sentence[j]
            if token["upostag"]=='VERB':
                tense.append(token["feats"].get('Tense', 'None'))
                verbs.append(token["form"])
    try:
        return len(set(tense))/len(verbs)
    except:
        return 0

#

# ----------syntactic features---------

def get_avg_distance(conll_sentence):
    """
    input: str(conll_sentence)
    output: float(average_token_head_distance)
    """
    sentence_parsed = parse_conll_to_dict(conll_sentence)
    dis = list()
    for i in range(len(sentence_parsed)):
        sentence = sentence_parsed[i]
        for j in range(len(sentence)):
            token = sentence[j]
            distance = abs(token["id"]-token["head"])
            dis.append(distance)
    try:
        return sum(dis)/len(dis)
    except:
        return 0

def combine_features_to_csv(conll_sentences):
    """
    input: list(conll_sentences)
    ouput: csv(features_extracted)
    """
    dict_features = dict()
    for sentence in conll_sentences:
        # lemma
        ttr = get_ttr(sentence)
        if "ttr" not in dict_features:
            dict_features["ttr"] = list()
        else:
            dict_features["ttr"].append(ttr)
        ca = if_ca(sentence)
        if "ca" not in dict_features:
            dict_features["ca"] = list()
        else:
            dict_features["ca"].append(ca)
        cela = if_cela(sentence)
        if "cela" not in dict_features:
            dict_features["cela"] = list()
        else:
            dict_features["cela"].append(cela)
        sms = if_sms(sentence)
        if "sms" not in dict_features:
            dict_features["sms"] = list()
        else:
            dict_features["sms"].append(sms)
        emoji = if_emoji(sentence)
        if "emoji" not in dict_features:
            dict_features["emoji"] = list()
        else:
            dict_features["emoji"].append(emoji)
        # morphosyntaxe
        diversity_tense = get_tense_diversity(sentence)
        if "diversity_tense" not in dict_features:
            dict_features["diversity_tense"] = list()
        else:
            dict_features["diversity_tense"].append(diversity_tense)
        # syntaxe
        avg_distance = get_avg_distance(sentence)
        if "avg_distance" not in dict_features:
            dict_features["avg_distance"] = list()
        else:
            dict_features["avg_distance"].append(avg_distance)

    df = pd.DataFrame.from_dict(dict_features)
    return df.to_csv("../1_data/4_features_csv/features.csv", sep="\t", encoding="utf-8")


# ----------------------error features-----------------------------
"""
1. obtain all typing errors with their sources(tweet/comment), an error may occur in one sentence for many times
2. combine typing error features with above features (csv+csv)
"""
# typing error
def get_typing_error(path_csv="../1_data/2_normalized_data/normalized_data.csv"):
    """
    input: csv(normalized_data)
    output: df({error1: ['tweet_1', 'tweet_2'], error2:['tweet_2'] ...})
    """
    dico_dir_error = dict()
    tool = language_tool_python.LanguageTool('fr-FR')
    df = pd.read_csv(path_csv, encoding="utf-8", sep="\t")
    # check typing error for each tweet/comment
    for index, row in df.iterrows():
        matches = tool.check(row['content'])
        for i in range(len(matches)):
            # ex. [Match({'ruleId': 'FR_SPELLING_RULE', 'message': 'Faute de frappe possible trouvée.', 'replacements': ['bonjour', 'bon gour'], 'offsetInContext': 0, 'context': 'bongour', 'offset': 0, 'errorLength': 7, 'category': 'TYPOS', 'ruleIssueType': 'misspelling', 'sentence': 'bongour'})]
            error = matches[i].ruleId  # type of error
            # stock infos in a dict(error: file_list)
            if error not in dico_dir_error:
                dico_dir_error[error] = list()
            else:
                dico_dir_error[error].append(row['id'])
    # the length of file list is not always the same, fill in df
    # transpose dataframe
    df = pd.DataFrame(dict([(k, pd.Series(v))
                            for k, v in dico_dir_error.items()])).T
    return df


def get_error_proportion(df):
    """
    input: df({error1: ['tweet_1', 'tweet_2'], error2:['tweet_2'] ...})
    output: csv('tweet_1',0.25(proportion of typing_erro1), 0.50(proportion of typing_error2)...)
    """
    # convert df into dictionary
    dico_dir_error = df.to_dict('list')
    # get all error types
    all_type_error = list(dico_dir_error.keys())
    # new dictionary to stock {id:[proportion_error1, ...]}
    dico_proportion = dict()
    for index, key in enumerate(dico_dir_error):
        for id in dico_dir_error[key]:
            if id not in dico_proportion:
                dico_proportion[id] = [0]*len(dico_dir_error)
                dico_proportion[id][index] += 1
            else:
                dico_proportion[id][index] += 1
    for k, v in dico_proportion.items():
        # replace numbers by their percentages in error list
        dico_proportion[k] = [1.*nbr_error/sum(v) for nbr_error in v]
    # add 'id' column name in all type errors' list to get the whole column names
    all_type_error.insert(0, 'id')
    # keep dictionary key (file id in our case) as first column
    df = pd.DataFrame.from_dict(dico_proportion, orient='index').reset_index()
    df.columns = all_type_error
    return df.to_csv("1_data/5_features_csv/error_feature.csv", encoding="utf-8")


def concatenate_features(features_lemme_morpho_synt="../1_data/5_features_csv/features.csv", typing_error="../1_data/5_features_csv/error_feature.csv"):
    """
    input: 2 csv files with different features which have a same column (id)
    output: 1 csv file concatenated
    """
    # all lemme, morphosyntactic and syntactic features are written in a csv file before
    feature1 = pd.read_csv(features_lemme_morpho_synt, encoding="utf-8")
    feature2 = pd.read_csv(typing_error, encoding="utf-8")
    features = feature1.merge(feature2, on="id")
    return features.to_csv("../1_data/4_features_csv/all_features_500.csv", encoding="utf-8")


# ------------------------------------------------------------
if __name__ == "__main__":

    # Part 1: linguistic features
    conll_sentences = get_sentences_tagged(
        "../1_data/3_data_tagged/data_tagged.conll")
    combine_features_to_csv(conll_sentences)
    # Part 2: typing error features
    typing_error_df = get_typing_error(
        "1_data/2_normalized_data/normalized_data_500.csv")
    get_error_proportion(typing_error_df)
    # Part 3: concatenate features
    concatenate_features("1_data/4_features_csv/features.csv",
                         "1_data/4_features_csv/error_feature.csv")

