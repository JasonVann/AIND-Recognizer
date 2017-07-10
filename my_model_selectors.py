import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        best_model = None
        best_score = None
        N = sum(self.lengths) # number of data points
              
        for n in range(self.min_n_components, self.max_n_components + 1):
            # Use try-catch to eliminate non-viable models from consideration.
            # For example, for "FISH", this fails if there's more than 6 components
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)

                # For number of parameters, Katie_tiwari gave a formular in this post
                # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/15
                d = len(self.X[0]) # num of features
                p = n*n + 2*n*d-1 # p: number of params

                score = -2 * logL + p * math.log(N)
                if best_score is None or score < best_score:
                    # For BIC, smaller value is better
                    best_score = score
                    best_model = model
            except:
                #print('{} components fails'.format(n))
                pass
            
        return best_model
    
class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        
        best_model = None
        best_score = None
        N = sum(self.lengths) # number of data points
        M = len(self.hwords) # total number of classes
        
        lst_logL = []
        models = []
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
                total_score = 0 # acculator for the sum of other classes, excluding the current word
                for k, v in self.hwords.items():
                    if k == self.this_word:
                        continue
                    temp_X, temp_lengths = v
                    temp_logL = model.score(temp_X, temp_lengths)
                    total_score += temp_logL
                
                DIC = logL - total_score/(M-1)
                if best_score is None or DIC > best_score:
                    best_score = DIC
                    best_model = model
 
            except:
                pass

        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        
        best_model = None
        best_score = None
        N = sum(self.lengths) # number of data points
              
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                # TODO
                

                # For number of parameters, Katie_tiwari gave a formular in this post
                # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/15
                d = len(self.X[0]) # num of features
                p = n*n + 2*n*d-1 # p: number of params

                DIC = -2 * logL + p * math.log(N)
                if best_score is None or DIC < best_score:
                    # For DIC, smaller value is better
                    best_score = score
                    best_model = model
            except:
                pass
            
        return best_model
