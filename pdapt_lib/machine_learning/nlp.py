""" NLP

basic natural language processing code

"""
from pdapt_lib.machine_learning.maths import sum_of_squares, dot, factorial


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
    >>> skip_gram(1, 2, "the rain in Spain falls mainly on the plain".split(" "))
    [['the', 'in'], ['rain', 'Spain'], ['in', 'falls'], ['Spain', 'mainly'], ['falls', 'on'], ['mainly', 'the'], ['on', 'plain']]
    >>> skip_gram(1, 3, "the rain in Spain falls mainly on the plain".split(" "))
    [['the', 'in', 'falls'], ['rain', 'Spain', 'mainly'], ['in', 'falls', 'on'], ['Spain', 'mainly', 'the'], ['falls', 'on', 'plain']]
    >>> skip_gram(2, 2, "the rain in Spain falls mainly on the plain".split(" "))
    [['the', 'Spain'], ['rain', 'falls'], ['in', 'mainly'], ['Spain', 'on'], ['falls', 'the'], ['mainly', 'plain']]
    """
    sgrams = [list([words[i]]) + [words[i+k+j] for j in range(1,n+1,2)] for i in range(len(words)-(k*2*(n-1)))]
    return sgrams
