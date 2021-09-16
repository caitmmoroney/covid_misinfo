# A LIME integration
import pandas as pd
import numpy as np
import nltk
import torch
import transformers as ppb # pytorch transformers
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.base import TransformerMixin
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from lime import submodular_pick
import re
import pickle
import time
import joblib
from scipy.io import loadmat
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
print('loaded imports')

# Load tweets dataset
tweets = pd.read_csv('COVID19_Dataset-text_labels_only.csv')
print('loaded data file')
targets = np.asarray(tweets['Is_Unreliable'].copy())

# Load word embeddings
A_mat_contents = loadmat('./A.mat')
A = A_mat_contents['A'] # ICA word embeddings from A matrix, shape num_words x 250

# Load vocabulary list
with open('tweet_vocab_list', 'rb') as f:
    tweet_vocab_list = pickle.load(f)

# Create vocabulary dictionary
vocabulary_dict = dict()
for i in range(len(tweet_vocab_list)):
    word = tweet_vocab_list[i]
    vocabulary_dict[word] = i

# to convert contractions picked up by word_tokenize() into full words
contractions = {
    "n't": 'not',
    "'ve": 'have',
    "'s": 'is',  # note that this will include possessive nouns
    'gonna': 'going to',
    'gotta': 'got to',
    "'d": 'would',
    "'ll": 'will',
    "'re": 'are',
    "'m": 'am',
    'wanna': 'want to'
}


# to convert nltk_pos tags to wordnet-compatible PoS tags
def convert_pos_wordnet(tag):
    tag_abbr = tag[0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }

    if tag_abbr in tag_dict:
        return tag_dict[tag_abbr]

def get_text_vectors(word_embeddings, # numpy array
                     word_index_dict, # dictionary mapping words to index in array
                     text_list, # list of strings to derive embeddings for
                     remove_stopwords = True,
                     lowercase = True,
                     lemmatize = True,
                     add_start_end_tokens = True):
    
    lemmatizer = WordNetLemmatizer()
    
    for k in range(len(text_list)):
        text = text_list[k]
        text = re.sub(r'[_~`@$%^&*[\]+=\|}{\"\'<>/]+', '', text)
        text_vec = np.zeros(word_embeddings.shape[1])
        words = word_tokenize(text)
        tracker = 0 # to track whether we've encountered a word for which we have an embedding (in each tweet)
        
        if remove_stopwords:
            clean_words = []
            for word in words:
                if word.lower() not in set(stopwords.words('english')):
                    clean_words.append(word)
            words = clean_words

        if lowercase:
            clean_words = []
            for word in words:
                clean_words.append(word.lower())

            words = clean_words

        if lemmatize:
            clean_words = []
            for word in words:
                PoS_tag = pos_tag([word])[0][1]

                # to change contractions to full word form
                if word in contractions:
                    word = contractions[word]

                if PoS_tag[0].upper() in 'JNVR':
                    word = lemmatizer.lemmatize(word, convert_pos_wordnet(PoS_tag))
                else:
                    word = lemmatizer.lemmatize(word)

                clean_words.append(word)

            words = clean_words

        if add_start_end_tokens:
            words = ['<START>'] + words + ['<END>']
        
        for i in range(len(words)):
            word = words[i]
            if word in word_index_dict:
                word_embed_vec = word_embeddings[word_index_dict[word],:]
                if tracker == 0:
                    text_matrix = word_embed_vec
                else:
                    text_matrix = np.vstack((text_matrix, word_embed_vec))
                    
                # only increment if we have come across a word in the embeddings dictionary
                tracker += 1
                    
        for j in range(len(text_vec)):
            text_vec[j] = text_matrix[:,j].mean()
            
        if k == 0:
            full_matrix = text_vec
        else:
            full_matrix = np.vstack((full_matrix, text_vec))
            
    return full_matrix


class Text2Embed(TransformerMixin):
    """ Description:
        Transformer that takes in a list of strings, calculates the word-context matrix
        (with any specified transformations), reduces the dimensionality of these word
        embeddings, and then provides the text embeddings of a (new) list of texts
        depending on which words in the "vocab" occur in the (new) strings.
    """

    # initialize class & private variables
    def __init__(self):

        self.text_embeddings = None

    def fit(self, corpus, y=None):

        """ Does nothing. Returns self.

            Params:
                corpus: list of strings

            Returns: self
        """
        

        return self

    def transform(self, new_corpus=None, y=None):

        """ Get text embeddings for given corpus, using predefined term dictionary and word embeddings.

            Returns: text embeddings (shape: num texts by embedding dimensions)
        """
        full_matrix = get_text_vectors(A, vocabulary_dict, new_corpus)

        self.text_embeddings = full_matrix

        return self.text_embeddings.copy()

# instantiate embedder
embedder = Text2Embed()
print('instantiated embedder class')
embedder.fit(tweets['Tweet'])
embedded_tweets = embedder.transform(tweets['Tweet'])
print('fitted embedder to tweets')

# load array of embedded tweets
A_embeddings = np.load('tweet_embed_A.npy')
print('loaded numpy A embeddings')

# if these don't match the prior bert embeddings I'm working with, there's an issue
if not np.array_equal(embedded_tweets, A_embeddings):
	print('!!!!! WARNING: embedded_tweets do not match A embeddings file !!!!!')


# load trained classification algorithm
svc = joblib.load('svm_A_Fold 3.pkl')
print('loaded pretrained model')

# get predictions
predictions = svc.predict(embedded_tweets)
print('made class predictions')

# create pipeline
c = make_pipeline(embedder, svc)

# instantiate LIME explainer
explainer = LimeTextExplainer(class_names = ['Reliable', 'Unreliable'])

# Establish list for explanations objects
explanations = []
# Establish list for explanations as list objects
explanation_tuples = []
# Establish list to track LIME explanation time
lime_time = np.zeros(np.shape(embedded_tweets)[0])

# Loop through test set of unreliable tweets:
for idx in range(np.shape(embedded_tweets)[0]):
    tweet = tweets['Tweet'][idx]
    y_true = targets[idx]
    y_predict = predictions[idx]
    num_words = len(re.split("\W+", tweet))
    startt = time.process_time() # to track how long it takes for LIME to form the explanation
    exp = explainer.explain_instance(tweet, c.predict_proba, num_features = num_words)
    endt = time.process_time()
    lime_time[idx] = endt - startt

    # save explanations
    explanation_tuples.append(exp.as_list())
    explanations.append(exp)

    # create csv for each tweet explanation
    fname = 'tweet{}expl_true'.format(idx) + '{}_predict'.format(y_true) + '{}.csv'.format(y_predict)
    expl_words = [tup[0] for tup in exp.as_list()]
    expl_netvals = [tup[1] for tup in exp.as_list()]
    df = pd.DataFrame({'term': expl_words, 'net_val': expl_netvals})
    df.to_csv(fname)

    # print idx every 2 tweets completed to track progress
    if (idx + 1) % 2 == 0:
    	print('completed tweet {}'.format(idx + 1))

# create list of all words in tweets
vocab_list = []
for tweet_exp in explanations:
    for tup in tweet_exp:
        if tup[0] not in vocab_list:
            vocab_list.append(tup[0])

# save explanations
with open('a_explanations.pkl', 'wb') as f:
    pickle.dump(explanations, f)

# save explanations as lists
with open('a_explanations_aslist.pkl', 'wb') as f:
	pickle.dump(explanation_tuples, f)

# print average LIME explanation time
print(np.mean(lime_time))

