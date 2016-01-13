""" NLP
basic natural language processing code


"""
from pdapt_lib.machine_learning.maths import sum_of_squares, dot, factorial
import re

# processing

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
    s = re.sub(r'[Ii]t\'s', 'it is', s) # this will remove capitalized It
    s = re.sub(r'\'ve', ' have', s)
    s = re.sub(r'n\'t', ' not',s)
    s = re.sub(r'\'ll', ' will',s)
    s = re.sub(r'\'m', ' am', s)
    s = re.sub(r'\'re', ' are', s)
    s = re.sub(r'\'tis', 'it is',s)
    s = re.sub(r'\'twas', 'it was',s)
    s = re.sub(r'let\'s', 'let us', s)
    s = re.sub(r'shan\'t', 'shall not', s)
    s = re.sub(r'G\'day', 'Good day', s)
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

def standardize_abbreviations(s):
    """ want to retain abbreviations, but a consistent set
    input: string
    output: string with reformated abbreviated
    NB this is needed to get more accurate counts
       eg  biologists are sloppy with acronyms (see tests)
    NB U.S. will be converted to US and not us (ambiguous)
    >>> standardize_abbreviations("The U.S. or U.S.A. test")
    'The US or USA test'
    >>> standardize_abbreviations("Test for dashes JAK-1")
    'Test for dashes JAK1'
    >>> standardize_abbreviations("Test for caps Jak-1")
    'Test for caps JAK1'
    """
    s = re.sub(r'(?<=[A-Z])\.', '', s) # positive lookbehind assertion
    s = re.sub(r'(?<=[A-Z])\-', '', s)
    s = re.sub(r'(?<=[a-z])\-', '', s)
    s = re.sub(r'[A-Z]*[a-z]*[0-9]+', lambda x: x.group().upper(), s)
    return s

def lowercase(s):
    """ return lowercased text EXCEPT for abbreviations
    input: text string
    output: text but special lowercased version
    NB will leave ALL capitalization, want this since
       associated with emphasis
    >>> lowercase("This is a NLP test.")
    'this is a NLP test.'
    """
    s = re.sub(r'[A-Z][a-z]+', lambda x: x.group().lower(), s)
    return s


def remove_numbers(s):
    """ remove numbers in string
    input: string text
    output: string text without numbers
    NB want to keep numbers associated with letters since
    they can be part of an abbreviation or name
    >>> remove_numbers("A sentence with some1 55 items.")
    'A sentence with some1 items.'
    >>> remove_numbers("123 test.")
    ' test.'
    >>> remove_numbers("A negative int -123 test.")
    'A negative int test.'
    >>> remove_numbers("A negative float -123.0 test.")
    'A negative float test.'
    >>> remove_numbers("A positive1 4float 123.0 test.")
    'A positive1 4float test.'
    >>> remove_numbers("A 1 2 3 test.")
    'A test.'
    >>> remove_numbers("A floating point at the end 123.4.")
    'A floating point at the end '
    >>> remove_numbers("Another floating point at the end 123.4)")
    'Another floating point at the end '
    >>> remove_numbers("Another floating point at the end 123.4]")
    'Another floating point at the end '
    >>> remove_numbers("Another floating point at the end 123.4!")
    'Another floating point at the end '
    >>> remove_numbers("Another floating point at the end 123.4}")
    'Another floating point at the end '
    """
    s = re.sub(r"(\s\d+\s)+", " ",s)
    s = re.sub(r"^\d+\s", " ", s)
    s = re.sub(r"\s[-+]*\d+\s", " ", s) # ints
    s = re.sub(r"\s[-+]*\d+\.\d+[\W]", " ", s) # floats
    return s


def remove_punctuation(s):
    """ remove punctionation
    input: string text
    output: string text with punctiontion removed
    >>> remove_punctuation("A bit of marks # @ ! & * [] {}.")
    'A bit of marks '
    >>> remove_punctuation("Another test-for-you!")
    'Another test for you '
    """
    s = re.sub(r'[^A-Za-z0-9]+', ' ', s)
    return s




# Models

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
