# BERT document embeddings
import pandas as pd
import numpy as np
import torch
import transformers as ppb # pytorch transformers
from sklearn.base import TransformerMixin

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
