#!/anaconda3/bin/python3.7
# -*- coding: UTF-8 -*-
import pandas as pd
import emoji

def get_data(path_file, sep=";"):
    """
    load csv file
    input: str(file path)
    output: dataframe
    """
    return pd.read_csv(path_file, sep=sep)

def save_data(df, path_file, sep=";"):
    """
    save dataframe to csv
    input: dataframe
    output: obj(csv file)
    """
    return df.to_csv(path_file, sep=sep)

def replace_emoji(text):
    """
    replace emoji signs by "emoji"
    input: str(text)
    output: str(text with emoji signs replaced by "emoji")
    """
    word_list = text.split(' ')
    for i,w in enumerate(word_list):
        # replace 1 or consecutive emojis
        if w[0] in emoji.UNICODE_EMOJI:
            word_list[i] = "emoji"
    text = ' '.join(word_list)
    return text

def replace_at(text):
    """
    replace @ sign by "user" text
    input: str(text)
    output: str(text with @ replaced by "user")
    """
    return text.replace("@", "user")

def replace_hashtag(text):
    """
    replace # sign by "hashtag_" text
    input: str(text)
    output: str(text with # replaced by "hashtag_")
    """
    return text.replace("#", "hashtag_")

def rm_url(text):
    """
    input: str(text)
    output: str(text without url)
    """
    word_list = text.split(' ')
    for i,w in enumerate(word_list):
        if w.startswith('http'):
            word_list[i] = ""
    text = ' '.join(word_list)
    return text

def tokenisation_punct(text):
    """
    for easier tokenisation
    input: str(text)
    output: str(spaces added before and after punctuation(s) in text)
    """
    list_symbole = ['.', '?', '?!', '!?', '...',
                    '!', '!!', '!!!', ':', '??', '???']
    for symbole in list_symbole:
        words = text.split(symbole)
        text = (" "+symbole+" ").join(words)
    return text

def normalization(text):
    """
    including all preprocessing
    input: str(text)
    output: str(text normalized)
    """
    # space before and after punctuation
    text = tokenisation_punct(text)
    # replace emoji by "emoji"
    text = replace_emoji(text)
    # replace # by "hashtag_"
    text = replace_hashtag(text)
    # replace @ by "user"
    text = replace_at(text)
    # remove urls
    text = rm_url(text)
    return text

def data_cleaning(df, col_name="Tweet"):
    """
    normalize all rows in text column
    input: dataframe
    output: dataframe(with text column normalized)
    """
    df_new = df.copy()
    df_new[col_name] =df_new[col_name].apply(normalization)
    return df_new

if __name__ == "__main__":
    df = get_data("../1_data/1_tweets_csv/tweet_annotes.csv", ";")
    df_new = data_cleaning(df, "Tweet")
    save_data(df_new, "../1_data/2_normalized_data/normalized_data_seed.csv")
