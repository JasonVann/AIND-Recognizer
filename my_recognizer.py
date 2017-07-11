import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    for key in test_set.get_all_Xlengths().keys():
        word = test_set.wordlist[key]
        X, lengths = test_set.get_item_Xlengths(key)
        
        # Iterate through models and find the highest prob
        best_model = None
        best_word = None
        best_score = float('-inf')
        prob = {}
        for model_word, model in models.items():
            try:
                logL = model.score(X, lengths)
                #if model_word not in prob:
                    #prob[model_word] = logL
                if logL > best_score:
                    prob[model_word] = logL
                    best_score = logL
                    best_word, best_model = model_word, model
            except:
                prob[model_word] = float('-inf')
                pass
        probabilities.append(prob)
        guesses.append(best_word)
        
    return (probabilities, guesses)

        
        
