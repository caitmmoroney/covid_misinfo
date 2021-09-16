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
import time
import joblib
print('loaded imports')

data_file = pd.read_csv('./COVID19_Dataset-text_labels_only.csv')
print('loaded data file')
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
print('fitted embedder to tweets')


embedded_tweets = embedder.transform(data_file['Tweet'])

# load array of embedded tweets
bert_embeddings = np.load('bert_embeddings.npy')
print('loaded numpy bert embeddings')

# if these don't match the prior bert embeddings I'm working with, there's an issue
if not np.array_equal(embedded_tweets, bert_embeddings):
	print('!!!!! WARNING: embedded_tweets do not match bert_embeddings file !!!!!')


# load trained classification algorithm
svc = joblib.load('svm_bert_Fold 4.pkl')
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
for idx in range(np.shape(embedded_tweets)[0]): # FIX THIS
    tweet = data_file['Tweet'][idx]
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
with open('bert_explanations.pkl', 'wb') as f:
    pickle.dump(explanations, f)

# save explanations as lists
with open('bert_explanations_aslist.pkl', 'wb') as f:
	pickle.dump(explanation_tuples, f)

# save vocab list
with open('tweet_vocab_list_unprocessed.pkl', 'wb') as f:
    pickle.dump(vocab_list, f)

# print average LIME explanation time
print(np.mean(lime_time))

