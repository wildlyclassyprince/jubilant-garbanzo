'''
analyse.py
----------

Reusable sentiment analysis methods.
'''

# Header
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

# The usual suspects ...
import os
import sys
import glob
import logging
import string
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# And their accomplices ...
from scipy import stats
from matplotlib.ticker import FuncFormatter
from textblob import TextBlob
from gensim import corpora
from gensim import models
from gensim import similarities
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from collections import defaultdict
from collections import Counter
from pprint import pprint

# Variables
LR= re.compile(r'((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)',
                            re.DOTALL)
UPDATES = ['-', '=', '+', '*','.', ',', '"',
           "'", '?', '!', ':', ';', '(', ')',
           '[', ']', '{', '}', 'amp', 'kkk',
           'hahaha', 'haha', 'ha', 'RT', 'i’m',
           '…', '–', 'http']

# Emoticons
HAPPY = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)',
    '=)', ':}', ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D',
    'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)", ":')", ':*',
    ':^*', '>:P', ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p',
    ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3'])

SAD = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L',
    ':<', ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(",
    ':\\', ':-c', ':c', ':{', '>:\\', ';('])

EMOJIS = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)

EP = ['@', '#']

# Read in the data
def format_data(csv_name):
    '''Converts multiple JSON files to CSV.'''
    path = os.path.join('data/json/', '*')
    files = glob.glob(path)
    data = pd.DataFrame()
    for file in files:
        # Takes a while ...
        temp = pd.read_json(file, lines=True)
        data = data.append(temp, sort=True)
    data.reset_index(inplace=True)
    data.to_csv('data/csv/' + csv_name, index=False)
    return data

# Removing @user references and links
def strip_links(text):
    '''Removes links in text.'''
    links = re.findall(LR, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text

def strip_all_entities(text):
    '''Removes @user references and hashtags.'''
    for separator in string.punctuation:
        if separator not in EP:
            text = text.replace(separator, ' ')
    words = list()
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in EP:
                words.append(word)
    return ' '.join(words)

# Text corpus
def create_document_corpus(df, column):
    '''Creates document corpus.'''
    return [i for i in df[column]]

# Removing common words and tokenize
def remove_common_words_and_tokenize(document_corpus):
    '''Removes common words and tokenizes text.'''
    stop_words = set(stopwords.words('english'))
    stop_words.update(UPDATES)
    # including lower case letters ...
    stop_words.update([i for i in string.ascii_lowercase])
    # and upper-case
    stop_words.update([i for i in string.ascii_uppercase])
    for doc in document_corpus:
        list_of_words = [i.lower() for i in wordpunct_tokenize(doc)
                         if i.lower() not in stop_words]
    stop_words.update(list_of_words)

    # Removing common words
    return [[word for word in doc.lower().split() if word not in
             stop_words] for doc in document_corpus]

# Removing words that appear only once
def remove_words_appearing_only_once(text_corpus):
    '''Removes words that appear only once.'''
    frequency = defaultdict(int)
    for text in text_corpus:
        for token in text:
            frequency[token] += 1

    return [[token for token in text if frequency[token] > 1]
            for text in text_corpus]

# Removing emojis
def remove_emojis(text_corpus):
    '''Removes emojis and emoticons from text corpus.'''
    # Combine
    emoticons = HAPPY.union(SAD)
    return [[token for token in text if token not in emoticons]
            for text in text_corpus]

# Removing empty tokens
def remove_empty_corpus_tokens(text_corpus):
    '''Removes empty text corpus tokens.'''
    return [text for text in text_corpus if len(text) > 1]

# Initializing tf-idf
def create_tfidf_model(corpus):
    '''Transform text to tf-idf model.'''
    # Initialization
    tfidf = models.TfidfModel(corpus)
    # Applying the transformation to the whole corpus
    return tfidf, tfidf[corpus]

# Initializing an LSI transformation
def create_lsi_model(corpus_tfidf, model_name, id2word, num_topics):
    '''Initializes an LSI transformation of tf-idf model.'''
    lsi = models.LsiModel(corpus_tfidf, id2word, num_topics)
    # Model persistence: save(), load()
    lsi.save('models/' + model_name + '.lsi')
    lsi = models.LsiModel.load('models/' + model_name + '.lsi')
    return lsi, lsi[corpus_tfidf]

# Initialize an LDA transformation
def create_lda_model(corpus_tfidf, id2word, num_topics):
    '''Initializes LDA transformation.'''
    # LDA Transformation
    lda = models.LdaModel(corpus_tfidf, id2word, num_topics)
    return lda, lda[corpus_tfidf]

# Queries
def query_similarity(doc, model_name, dict_of_doc,
                     lsi_model, doc_corpus, corpus):
    # Transform corpus to LSI space and index it
    index = similarities.MatrixSimilarity(lsi_model[corpus])

    # Index persistence
    index.save('models/' + model_name + '.index')
    index = similarities.MatrixSimilarity.load('models/' +
                                               model_name +
                                               '.index')
    
    # Performing queries
    vec_bow = dict_of_doc.doc2bow(doc.lower().split())

    # Convert the query to LSI space
    vec_lsi = lsi_model[vec_bow]

    # Perform a similarity query against the corpus
    sims = index[vec_lsi]

    # Ranking the tweets by their weights of similarity
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    # Printing the associated Tweets:
    for i in range(10):
        print("Tweet Rank #{}:" +
              "\tWeight: {}\n" +
              "Raw text: {}\n".format(i+1,
                                      sims[i][1],
                                      doc_corpus[sims[i][0]]))
        
def format_topics_sentences(ldamodel, corpus, texts):
    '''
    Returns dominant topics and respective percentage contributions
    with original text.
    '''
    # Initialize output
    sent_topics_df = pd.DataFrame()
    
    # Get main topic for each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ', '.join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num),
                               round(prop_topic, 4),
                               topic_keywords]),
                    ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['dominant_topic',
                              'percentage_distribution',
                              'topic_keywords']
    
    # Add original text to the end of output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df

# Formatting
def word_counts_of_topic_keywords(lda_model, text_corpus):
    '''Returns the word counts of topic keywords.'''
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in text_corpus for w in w_list]
    counter = Counter(data_flat)
    
    out = list()
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])
    return pd.DataFrame(out, columns=['word',
                                      'topic_id',
                                      'importance',
                                      'word_count'])

# Sentence Coloring of N Sentences
def topics_per_document(model,lda, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics, topic_percentages = list(), list()
    for i, corp in enumerate(corpus_sel):
        topic_id_percentages = lda[corp]
        dominant_topic = sorted(topic_id_percentages,
                                key=lambda x: x[1],
                                reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_id_percentages)
    return dominant_topics, topic_percentages

def append_cleaned_text(texts, df, column_name):
    '''Added cleaned text column to dataframe.'''
    cleaned_text = list()
    for i in range(len(texts)):
        if len(texts[i]) != 0:
            cleaned_text.append(' '.join(i for i in texts[i]))
        else:
            cleaned_text.append(' ')
    df[column_name] = cleaned_text

# Create polarity and subjectivity columns
def polarity_and_subjectivity(df, column):
    '''Creates the polarity and subjectivity columns.'''
    polarity = list(map(lambda tweet:TextBlob(tweet).polarity,
                        df[column]))
    subjectivity = list(map(lambda tweet: TextBlob(tweet).subjectivity,
                            df[column]))
    df['polarity'] = polarity
    df['subjectivity'] = subjectivity

# Correlations
def correlations(x, y, **kws):
    '''Calculate correlation coefficients.'''
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .6),
                xycoords=ax.transAxes,
                size=24)
