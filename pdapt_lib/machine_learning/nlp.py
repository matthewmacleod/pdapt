""" NLP

basic natural language processing code


"""
from pdapt_lib.machine_learning.maths import sum_of_squares, dot, factorial
import re

def expand(s):
    """ expand common contractions
    input: string
    output: string with contractions expanded
    >>> expand("It's cold out")
    'it is cold out'
    >>> expand("I've been busy")
    'I have been busy'
    >>> expand("you're going to be amazed")
    'you are going to be amazed'
    """
    s = re.sub(r'[Ii]t\'s', 'it is', s) # this will remove capitalized I
    s = re.sub(r'\'ve', ' have', s)
    s = re.sub(r'n\'t', ' not',s)
    s = re.sub(r'\'ll', ' will',s)
    s = re.sub(r'\'m', ' am', s)
    s = re.sub(r'\'re', ' are', s)
    s = re.sub(r'\'tis', 'it is',s)
    s = re.sub(r'\'twas', 'it was',s)
    s = re.sub(r'let\'s', 'let us', s)
    s = re.sub(r'shan\'t', 'shall not', s)
    # since can be possesive cant make this general
    s = re.sub(r'who\'s', 'who is',s)
    s = re.sub(r'where\'s', 'where is',s)
    s = re.sub(r'what\'s', 'what is',s)
    s = re.sub(r'why\'s', 'why is',s)
    s = re.sub(r'that\'s', 'that is',s)
    s = re.sub(r'there\'s', 'there is', s)
    s = re.sub(r'someone\'s', 'someone is',s)
    s = re.sub(r'somebody\'s', 'somebody is',s)
    s = re.sub(r'something\'s', 'something is',s)
    s = re.sub(r'he\'s', 'he is',s) # this could be dangerous
    s = re.sub(r'o\'clock', 'of the clock',s)
    s = re.sub(r'ain\'t', 'am not', s)
    return s


def n_gram(n, words):
    """ n gram model
    input: list of words
    output: list of n-grams from text
    NB to join sublists, [" ".join(i) for i in ngrams]
    >>> n_gram(2, "the rain in Spain falls mainly on the plain".split(" "))
    [['the', 'rain'], ['rain', 'in'], ['in', 'Spain'], ['Spain', 'falls'], ['falls', 'mainly'], ['mainly', 'on'], ['on', 'the'], ['the', 'plain']]
    >>> n_gram(3, "the rain in Spain falls mainly on the plain".split(" "))
    [['the', 'rain', 'in'], ['rain', 'in', 'Spain'], ['in', 'Spain', 'falls'], ['Spain', 'falls', 'mainly'], ['falls', 'mainly', 'on'], ['mainly', 'on', 'the'], ['on', 'the', 'plain']]
    """
    ngrams = [words[i:i+n] for i in range(len(words)-(n-1))]
    return ngrams


def skip_gram(k, n, words):
    """ skip grams
    input: k (skip), n (as in n-gram), list of words
    output: list of skip grams
    NB this is a bit trickier to do in one line...
    >>> skip_gram(1, 2, "the rain in Spain falls mainly on the plain".split(" "))
    [['the', 'in'], ['rain', 'Spain'], ['in', 'falls'], ['Spain', 'mainly'], ['falls', 'on'], ['mainly', 'the'], ['on', 'plain']]
    >>> skip_gram(1, 3, "the rain in Spain falls mainly on the plain".split(" "))
    [['the', 'in', 'falls'], ['rain', 'Spain', 'mainly'], ['in', 'falls', 'on'], ['Spain', 'mainly', 'the'], ['falls', 'on', 'plain']]
    >>> skip_gram(2, 2, "the rain in Spain falls mainly on the plain".split(" "))
    [['the', 'Spain'], ['rain', 'falls'], ['in', 'mainly'], ['Spain', 'on'], ['falls', 'the'], ['mainly', 'plain']]
    >>> skip_gram(2, 3, "the rain in Spain falls mainly on the plain".split(" "))
    [['the', 'Spain', 'on'], ['rain', 'falls', 'the'], ['in', 'mainly', 'plain']]
    """
    sgrams = [list([words[i]]) + [words[i+k+j] for j in range(1,n+k,k+1)] for i in range(len(words)-((1+k)*(n-1)))]
    return sgrams
