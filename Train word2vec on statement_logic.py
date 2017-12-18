"""Training word2vec on statement_logic output
"""

#import sys
#sys.path.append('C:/Users/Sam/Dropbox/Summer Research/Code')

from gensim.models.word2vec import Word2Vec as Word2Vec_gensim
from word2vec_modified import Word2Vec as Word2Vec_train_on_current
# Difference between 'Word2Vec_no_train_on_current' and 'Word2Vec_gensim' is just that reduced_window and sample_int things are commented out (and no cython code so slower)
from word2vec_modified2 import Word2Vec as Word2Vec_no_optimisations # used to be called Word2Vec_no_train_on_current
from word2vec_modified3 import Word2Vec as Word2Vec_no_train_w_operator_input # won't train on current (like standard)

import os
import time
import string
import logging # Import the built-in logging module and configure it so that Word2Vec creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from numpy import save as np_save


class SpacedLines:
    """Creates an object with a __iter__ method that returns a list of strings,
       where each string is the next entire example of logical reasoning.
       The examples are in .txt files, with one statement per line, and each
       example separated by a newline
    """
    def __init__(self, fullpath):
        """Initialisation"""
        self.fullpath = fullpath
    
    def __iter__(self):
        """Iterates through Bible verses, returning a list of words in each verse.
           Called by word2vec when iterating through the verses to train the
           model.
        """
        example = []
        for line in open(self.fullpath):
            if line != '\n':
                example.append(line.rstrip()) # remove newline
            else:
                yield example
                example = []


def test_SpacedLines():
    logic_examples = SpacedLines(corpus_dir)
    i = 0
    for ex in logic_examples:
        print(ex)
        i += 1
        if i > 10:
            break    


if __name__ == '__main__':
    corpus_dir = 'C:/Users/Sam/Dropbox/Summer Research/Code/Informal logic'
    
    models_dict = {} # concern: too big to be stored in ram?
    
    epochs = 150
    dims = 50
    eta = 0.1
    window_size = 1
    minimum_count = 2
    
    for model_name, Word2Vec_model in [('Gensim', Word2Vec_gensim), ('Train_on_current', Word2Vec_train_on_current), ('Not_optimised', Word2Vec_no_optimisations), ('No_train_operator_input', Word2Vec_no_train_w_operator_input)]:
        for filename in os.listdir(corpus_dir):
            if filename.endswith('.txt'):
                fullpath = os.path.join(corpus_dir, filename)
                logic_examples = SpacedLines(fullpath)
                models_dict['{}-{}'.format(model_name, filename)] = Word2Vec_model(logic_examples, sg=1, min_count=minimum_count, window=window_size, size=dims, alpha=eta, iter=epochs)
                print('Model {} file {} complete'.format(model_name, filename))
                
    
    np_save('models_dict2.npy', models_dict)
    print('Done!!')
