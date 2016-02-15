"""
__file__
    nlp_utils.py
__description__
    Adopted from @Chenglong Chen's code < https://github.com/ChenglongChen/Kaggle_CrowdFlower >
    This file provides functions to perform NLP task, e.g., TF-IDF and POS tagging.
__author__
    Venkata Ravuri < venkat@nikhu.com >
"""
import re
import nltk
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

import config

################
## Stop Words ##
################
stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)

replace_dict = {
    "in.": "inch",
    "sq.": "square",
    "ft.": "feet",
    "cu.": "cubic",
    "lbs.": "pounds",
    "gal.": "gallons",
    "w x": "width ",
    "h x": "height ",
    "ah ": "ampere",
    "gal.": "gallons",
    "yds.": "yards",
    "yd.": "yard",
    "o.d.": "od",
    "oz.": "ounce",
    "in.": "inch",
    "lb.": "pounds",
    "dia.": "diameter",
    "sch.": "schedule",
    "qt. ": "quintal"
}

##############
## Stemming ##
##############
if config.stemmer_type == "porter":
    english_stemmer = nltk.stem.PorterStemmer()
elif config.stemmer_type == "snowball":
    english_stemmer = nltk.stem.SnowballStemmer('english')


def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed


# TODO
# TODO
# TODO
# 13.33  7,0202 SHOULD BE EXCLUDED.

remove_punctuations_regex = re.compile("[ &<>)(_,.;:!?/-]+")
remove_apostrophe_s_regex = re.compile("'s\\b")
remove_apostrophe_regex = re.compile("[']+")
remove_double_quotes_regex = re.compile("[\"]+")


################
## Text Clean ##
################
def clean_text(line):
    # Remove HTML tags
    line = BeautifulSoup(line, "lxml").get_text(separator=" ")

    # Remove '&nbsp;' from the text content before HTML tags strip off.
    line.replace('&nbsp;', ' ')

    # Convert to lower case
    line = line.lower()

    # replace other words, do this only after lower case conversion as it effects mixed case.
    for k, v in replace_dict.items():
        line = re.sub(k, v, line)

    # Replace all punctuation and special characters by space
    line = re.sub(remove_punctuations_regex, " ", line)
    # Remove the apostrophe's
    line = re.sub(remove_apostrophe_s_regex, "", line)
    # Remove the apostrophe
    line = re.sub(remove_apostrophe_regex, "", line)
    # Remove the double quotes
    line = re.sub(remove_double_quotes_regex, "", line)

    return line


token_pattern = r"(?u)\b\w\w+\b"
tfidf__norm = "l2"
tfidf__max_df = 0.75
tfidf__min_df = 3


############
## TF-IDF ##
############
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def getTFV(token_pattern=token_pattern,
           norm=tfidf__norm,
           max_df=tfidf__max_df,
           min_df=tfidf__min_df,
           ngram_range=(1, 1),
           vocabulary=None,
           stop_words='english'):
    tfv = StemmedTfidfVectorizer(min_df=min_df, max_df=max_df, max_features=None,
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                                 stop_words=stop_words, norm=norm, vocabulary=vocabulary)
    return tfv
