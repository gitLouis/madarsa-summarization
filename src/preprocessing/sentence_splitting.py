from nltk import sent_tokenize
import re

def split_into_sentences(text):
    custom_pattern = r'[.!?\\n]'
    text = re.sub(custom_pattern, " . ", text)
    sentences_ = sent_tokenize(text)
    return sentences_

def explode_sentences(docs_col):
    sentences = docs_col.apply(split_into_sentences)
    sentences = sentences.explode()
    sentences.dropna(inplace=True)
    sentences = sentences.str.replace('.','')
    sentences = sentences.str.strip()
    fltr = sentences.apply(len) < 3
    sentences.drop(index=fltr[fltr].index, inplace=True)
    # col_name = sentences.name
    sentences = sentences.reset_index(drop=True)
    # sentences.columns = ['record_id',col_name]
    return sentences

