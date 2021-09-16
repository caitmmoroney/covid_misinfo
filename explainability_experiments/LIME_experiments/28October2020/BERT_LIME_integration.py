# BERT LIME integration
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

data_file = pd.read_csv(r'COVID19_Dataset-text_labels_only.csv')
tweets = np.asarray(data_file['Tweet'].copy())
targets = np.asarray(data_file['Is_Unreliable'].copy())

for i in range(np.shape(tweets)[0]):
    tweets[i] = tweets[i].lower()

class Text2Embed(TransformerMixin):
    """ Description:
        Transformer that takes in a list of strings, constructs word embeddings
        using BERT, and then provides the text embeddings of a (new) list of texts
        depending on which words in the "vocab" occur in the (new) strings.
    """

    # initialize class & private variables
    def __init__(self):

        self.corpus = None
        #self.X = None
        self.text_embeddings = None
        
        # Load pretrained model/tokenizer
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)

    def fit(self, corpus, y=None):

        """ Do nothing because BERT will be used in transform method. Return self.

            Params:
                corpus: list of strings

            Returns: self
        """

        return self

    def transform(self, new_corpus=None, y=None):

        """ Get text embeddings for given corpus, using BERT embeddings.

            Returns: text embeddings (shape: num texts by embedding dimensions)
        """
        
        # Load pretrained model/tokenizer
        tokenizer = self.tokenizer
        model = self.model
        
        for k in range(len(new_corpus)):
            text = new_corpus[k]
            #text = re.sub(r'[_~`@$%^&*[\]+=\|}{\"\'<>/]+', '', text)
            text_vec = np.zeros(768) # num features in bert base
            
            tokenized = tokenizer.encode(text.lower(), add_special_tokens=True)
            #tokenized = np.array(tokenized)
            
            # max length of tweet tokens is 83 (from Saswat's code); pad all vectors
            maxi = 83
            #padded = list()
            #padded.append(np.array(tokenized + [0]*(maxi - len(tokenized))))
            padded = np.array(tokenized + [0]*(maxi - len(tokenized)))
            
            segment_ids = [1]*len(padded)
            
            # create tensors
            tokens_tensor = torch.tensor([padded])
            segments_tensor = torch.tensor([segment_ids])
            
            with torch.no_grad():
                last_hidden_states = model(tokens_tensor, segments_tensor)[0] # pull out only the last hidden state
            
            last_hidden_states = last_hidden_states.numpy() # dim: tweets x words x features (where tweets = 1)
            
            word_embeddings = last_hidden_states[0,:,:] # dim: words x features (where features = 768)
            
            for j in range(768):
                text_vec[j] = word_embeddings[:, j].mean() # should be of dimension 1 x 768

            if k == 0:
                full_matrix = text_vec
            else:
                full_matrix = np.vstack((full_matrix, text_vec))

        self.text_embeddings = full_matrix

        return self.text_embeddings.copy()


# instantiate embedder
embedder = Text2Embed()
embedder.fit(data_file['Tweet'])


embedded_tweets = embedder.transform(data_file['Tweet'])

# save array of embedded tweets
np.save('bert_embeddings', embedded_tweets)


# instantiate classification algorithm

# round 1 winner
svc = SVC(C = 1, kernel = 'rbf', probability = True)

class1_train_indices = list(range(100))
class0_train_indices = list(range(280,380))

train_X = embedded_tweets[[class1_train_indices + class0_train_indices],:][0]

hundred_ones = [1]*100
hundred_zeros = [0]*100
train_Y = hundred_ones + hundred_zeros

# fit SVC model on training subset of tweet embeddings
svc.fit(train_X, train_Y)


class1_test_indices = list(range(100,280))
class0_test_indices = list(range(380,560))
test_X = data_file['Tweet'][class1_test_indices + class0_test_indices]
test_X = test_X.reset_index(drop = True)

# create pipeline
c = make_pipeline(embedder, svc)

# instantiate LIME explainer
explainer = LimeTextExplainer(class_names = ['Reliable', 'Unreliable'])

# Establish list
explanations = []

# Loop through test set of unreliable tweets:
for idx in range(100,280):
    tweet = data_file['Tweet'][idx]
    num_words = len(re.split("\W+", tweet))
    exp = explainer.explain_instance(tweet, c.predict_proba, num_features = num_words)
    explanations.append(exp.as_list())

# create list of all words in tweets
vocab_list = []
for subList in explanations:
    for el in subList:
        if el[0] not in vocab_list:
            vocab_list.append(el[0])

# save explanations
with open('bert_explanation_list', 'wb') as f:
    pickle.dump(explanations, f)

# save vocab list
with open('tweets_vocab_list', 'wb') as f:
    pickle.dump(vocab_list, f)

