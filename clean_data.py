import os
import re
import pandas as pd
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer #Tokenizer for preprocessing
preprop_tokenizer = RegexpTokenizer(r'\w+')

from utils import Timer

def get_clean_letters_mapper(clean_letter_text_file):
    clean_word_dict = {}
    with open(clean_letter_text_file, "r", encoding="utf-8") as cl:
        for line in cl:
            line = line.strip("\n")
            typo_letter, correct_letter = line.split(",")
            clean_word_dict[typo_letter] = correct_letter
    return clean_word_dict

def clean_text(text, clean_word_dict, remove_stopwords=True, remove_special_chars=True, remove_contractions=True ,  clean_wiki_tokens=False, stem_words=False ):
    """Clean the text, with the option to remove stopwords and to stem words."""
    # # dirty words
    text = text.lower()
    text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "", text)
    text = re.sub(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}", "", text)

    ## Regex to remove wiki special characters
    if clean_wiki_tokens:
        # Drop the image
        text = re.sub(r"image:[a-zA-Z0-9]*\.jpg", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.png", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.gif", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.bmp", " ", text)

        # Drop css
        text = re.sub(r"#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})", " ",text)
        text = re.sub(r"\{\|[^\}]*\|\}", " ", text)
        
        # Clean templates
        text = re.sub(r"\[?\[user:.*\]", " ", text)
        text = re.sub(r"\[?\[user:.*\|", " ", text)        
        text = re.sub(r"\[?\[wikipedia:.*\]", " ", text)
        text = re.sub(r"\[?\[wikipedia:.*\|", " ", text)
        text = re.sub(r"\[?\[special:.*\]", " ", text)
        text = re.sub(r"\[?\[special:.*\|", " ", text)
        text = re.sub(r"\[?\[category:.*\]", " ", text)
        text = re.sub(r"\[?\[category:.*\|", " ", text)
    
    ## Replace special characters with proper characters
    for typo_letter, correct_letter in clean_word_dict.items():
        text = re.sub(typo_letter, f" {correct_letter} ", text)

    # Regex to replace contractions
    if remove_contractions:
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"´", "'", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"doesn't", "does not ", text)
        text = re.sub(r"dont't", "do not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"\!", " ! ", text)
        text = re.sub(r"\"", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        # regex to replace all numerics
        replace_numbers = re.compile(r'\d+',re.IGNORECASE)
        text = replace_numbers.sub(" ", text)
    
    # Regex to remove all Non-Alpha Numeric and space
    if remove_special_chars:
        special_character_removal = re.compile(r'[^?!.,:a-z\d ]',re.IGNORECASE)
        text = special_character_removal.sub(' ',text)

    ## Tokenize text and remove stop words
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        token_list = preprop_tokenizer.tokenize(text)
        filtered_token_list = [word for word in token_list if word not in stop_words]
        text = " ".join(filtered_token_list).strip()
     
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer("english")
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return text

if __name__ == "__main__":

    ## INPUT FILES
    DATA_DIR = "data"
    TRAIN_DATA_FILE = os.path.join(DATA_DIR, "train.csv")
    TEST_DATA_FILE = os.path.join(DATA_DIR, "test.csv")
    clean_letter_text_file = os.path.join(DATA_DIR, "clean_letters.txt")

    ## OUTPUT FILES
    TRANSFORMED_TRAIN_DATA_FILE = os.path.join(DATA_DIR, "transformed_train.csv")
    TRANSFORMED_TEST_DATA_FILE = os.path.join(DATA_DIR, "transformed_test.csv")

    ## Load data
    print("Loading the data")
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    ## Load letter mapper
    print("Loading the clean letter mapper")
    clean_word_dict = get_clean_letters_mapper(clean_letter_text_file)

    ## Process Text Dataset
    t = Timer("Processing Text Data")
    t.start()
    train_comments_list = train_df["comment_text"].tolist()
    test_comments_list = test_df["comment_text"].tolist()
    transformed_train_comments_list = [clean_text(comment_text, clean_word_dict) for comment_text in train_comments_list]    
    t.toc("Processing Train Data")
    transformed_test_comments_list =  [clean_text(comment_text, clean_word_dict) for comment_text in test_comments_list]
    t.stop()
    ## It takes 3 mins
    print(train_comments_list[:2])
    print(transformed_train_comments_list[:2])

    ## Export transformed text to files
    train_df["comment_text"] = transformed_train_comments_list
    test_df["comment_text"] = transformed_test_comments_list
    train_df.to_csv(TRANSFORMED_TRAIN_DATA_FILE, index=False)
    test_df.to_csv(TRANSFORMED_TEST_DATA_FILE, index=False)
    print("Export transformed files !")