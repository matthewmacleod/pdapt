""" NLP
pdapt basic natural language processing code


"""
from pdapt_lib.machine_learning.maths import sum_of_squares, dot, factorial
import re
from collections import defaultdict
from functools import reduce


# processing

def split_by_n(s, n):
    """ generator to divide a string into n chunks """
    while s:
        yield s[:n]
        s = s[n:]

def is_even(x):
    """ return true if number is even """
    if x % 2 == 1:
        return False
    else:
        return True

def n_split(n, s):
    """ split a string
    input:  total number of chunks n, string s
    output: list of string chunks
    NB: exact chunck sizes may differ slightly but there will be
    no information loss and number of chunks returned should be consistent.
    >>> n_split(2, "its a whole new world!")
    ['its a whole ', 'new world!']
    >>> n_split(2, "it's an odd new world!!")
    ["it's an odd ", 'new world!!']
    >>> n_split(3, "it's a whole new world")
    ["it's a w", 'hole new', ' world']
    >>> n_split(4, "it's a whole new world")
    ["it's a", ' whole', ' new w', 'orld']
    """
    chunks = (len(s)//n) + 1
    return list(split_by_n(s, chunks))


def expand(s):
    """ expand common contractions
    input: string s
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
    input: string s
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


def extract_sentences(s):
    """ extract sentences
    input: string text s
    output: list of sentence strings
    >>> extract_sentences("I'm a sentence. Me too! And me?")
    ["I'm a sentence.", 'Me too!', 'And me?']
    """
    extract = re.compile('[A-Z].+?[.!?]\s*?') # non-greedy
    return extract.findall(s)


def lowercase(s):
    """ return lowercased text EXCEPT for abbreviations
    input: text string s
    output: text but special lowercased version
    NB will leave ALL capitalization, want this since
       associated with emphasis
    >>> lowercase("This is a NLP test.")
    'this is a NLP test.'
    """
    s = re.sub(r'[A-Z][a-z]+', lambda x: x.group().lower(), s)
    s = re.sub(r'^[A-Z]\s', lambda x: x.group().lower(), s)
    return s

def uppercase_i(s):
    """ uppercase single word I
    >>> uppercase_i("i better work or i will have trouble Indeed")
    'I better work or I will have trouble Indeed'
    """
    s = re.sub(r'^i\s+', lambda x: x.group().upper(), s)
    s = re.sub(r'\s+i\s+', lambda x: x.group().upper(), s)
    return s


def remove_numbers(s):
    """ remove most numbers in string
    input: string text s
    output: string text without *most* numbers
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


def remove_stopwords(s):
    """ remove common words
    input: text string s
    output: text string with stop words removed
    >>> remove_stopwords('a sentence with some sans common stopwords')
    'sentence sans common stopwords'
    """
    stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again",
                  "against", "all", "almost", "alone", "along", "already", "also","although",
                  "always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another",
                  "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",
                  "at", "back","be","became", "because","become","becomes", "becoming", "been",
                  "before", "beforehand", "behind", "being", "below", "beside", "besides",
                  "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can",
                  "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe",
                  "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either",
                  "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every",
                  "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
                  "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found",
                  "four", "from", "front", "full", "further", "get", "give", "go", "had", "has",
                  "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein",
                  "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred",
                  "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself",
                  "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may",
                  "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
                  "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless",
                  "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere",
                  "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others",
                  "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps",
                  "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems",
                  "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty",
                  "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere",
                  "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them",
                  "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein",
                  "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though",
                  "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top",
                  "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon",
                  "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence",
                  "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
                  "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole",
                  "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you",
                  "your", "yours", "yourself", "yourselves", "s", "t", "d"]
    uncommon_words = list(filter(lambda x: x not in stopwords, s.split(" ")))
    return " ".join(uncommon_words)


def tokenize(s, n=1, removing_stopwords=False):
    """ lex the text
    input: string of text s, n-gram model with default unigram model
    output: dictionary of ngram token keys and counts (values)
    NB: testing is moved to tests_nlp.py since more complicated tests
    are required.
    NB 2: careful with stopword removal, sometimes they are helpful
    """
    # NB the order of the following processing functions is important!
    s = expand(s)
    s = standardize_abbreviations(s)
    s = remove_numbers(s)
    s = lowercase(s)
    s = uppercase_i(s)
    s = remove_punctuation(s)
    if removing_stopwords:
        s = remove_stopwords(s)
    tokens = {}
    ngrams =  n_gram(n,s)
    for w in ngrams:
        if w in tokens:
            tokens[w] += 1
        else:
            tokens[w] = 1
    return tokens

def merge_tokens(a,b):
    """ combine token sets
    input: tokens a, tokens b
    output: combined dictionary of a and b tokens where values of same keys have been combined
    NB the lengths of each dictionary may differ
    """
    new_tokens = {}
    for k,v in a.items():
        if k in b:
            new_tokens[k] = b[k] + v
        else:
            new_tokens[k] = v
    for k,v in b.items():
        if k in a:
            new_tokens[k] = a[k] + v
        else:
            new_tokens[k] = v
    return new_tokens


# Simple string features

def mean_sentence_length(s):
    """ average sentence size
    input: string text s
    output: avg number of characters in each sentence in string
    >>> mean_sentence_length("I'm a very long sentence. Me too! And me?")
    13.0
    """
    sentences = extract_sentences(s)
    total = float(len(sentences))
    return sum(map(lambda x: len(x), sentences)) / total


def mean_words_in_sentence(s):
    """ average number of words in sentence
    input: string text s
    output: average number of words in each sentence
    NB while similar to mean_sentance_length this is perhaps more comprehensible metric
    >>> mean_words_in_sentence("I'm a very long sentence. Me too! And me?")
    3.0
    """
    sentences = extract_sentences(s)
    total = float(len(sentences))
    return sum(map(lambda x: len(x.split(" ")), sentences)) / total


# Simple token teatures

def mean_token_occurance(tokens):
    """ average occurences of token in vocabulary
    input: dictionary of tokens
    output: average number of time each token in the vocabulary occurs
    >>> mean_token_occurance({'the rain': 2, 'in spain': 2, 'rain in': 2, 'spain falls': 1, 'falls mainly': 1, 'mainly in': 1})
    1.5
    """
    total = float(len(tokens))
    return sum(map(lambda x: x[1], tokens.items())) / total


def mean_vocab_word_length(tokens):
    """ currently expecting unigrams
    input: dictionary of unigram tokens
    output: average length of each vocab word
    NB does not reflect number of times occured in text (see mean_corpus_word_length for this)
    >>> mean_vocab_word_length({'a': 2, 'simple': 1, 'version': 1, 'tokenizer': 1, 'of': 1})
    5.0
    """
    total = float(len(tokens))
    return reduce(lambda acc, x: acc + len(x[0]), tokens.items(), 0) / total


def mean_corpus_word_length(tokens):
    """ currently expecting unigrams
    input: dictionary of unigram tokens
    output: average length each word in the corpus
    >>> mean_corpus_word_length({'a': 2, 'simple': 1, 'version': 1, 'tokenizer': 1, 'of': 2})
    4.0
    """
    total = float(sum(map(lambda x: x[1], tokens.items())))
    return reduce(lambda acc, x: acc + len(x[0])*x[1], tokens.items(), 0) / total


def personal_pronoun_density(tokens):
    """ get ration of personal pronouns to words
    input: tokens
    output: density of person pronouns in corpus (all text)
    NB a supposedly good metric for differentiating gender
    >>> personal_pronoun_density({'a': 2, 'simple': 1, 'version': 1, 'tokenizer': 1, 'of': 2, 'he': 2, 'she': 5})
    0.5
    """
    pp = ['I','me','you','he','him','his','she','her','it','we','they','them','us']
    total = float(sum(map(lambda x: x[1], tokens.items())))
    pros = list(filter(lambda x: x[0] in pp, tokens.items()))
    counts = reduce(lambda acc, x: acc + x[1], pros, 0)
    return counts/total


def anagram(a,b):
    """ condititional test for anagram
    input: a string, b string
    output: true if anagram else false
    >>> anagram('salvador dali','avida dollars')
    True
    """
    return sorted(a) == sorted(b)


def anagrams(words):
    """
    input: list of words
    output: list of anagram tuples
    >>> anagrams(['test','listen','silent','ceiiinosssttuv','zest','uttensiosicvis','hamlet','amleth'])
    [('listen', 'silent'), ('ceiiinosssttuv', 'uttensiosicvis'), ('hamlet', 'amleth')]
    """
    agrams = []
    for x in words:
        for y in words:
            if x != y:
                if anagram(x,y):
                    if (y,x) not in agrams:
                        agrams.append((x,y))
    return agrams


def anagram_vocab_density(tokens):
    """ return the density of anagrams in a vocabulary
    input: tokens
    output: number of anagrams divided by total vocabulary
    >>> anagram_vocab_density({'simple': 1, 'version': 1, 'tokenizer': 1, 'of': 2, 'he': 2, 'she': 5, 'bird': 3, 'brid': 3})
    0.25
    """
    total = float(len(tokens))
    words = tokens.keys()
    agrams = anagrams(words)
    return (len(agrams)*2.0)/total


def anagram_corpus_density(tokens):
    """ return anagram usage density
    >>> anagram_corpus_density({'a': 2, 'simple': 1, 'version': 1, 'tokenizer': 1, 'of': 2, 'he': 2, 'she': 5, 'bird': 3, 'brid': 3})
    0.3
    """
    total = float(sum(map(lambda x: x[1], tokens.items())))
    words = tokens.keys()
    agrams = anagrams(words)
    return sum(map(lambda x: tokens[x[0]] + tokens[x[1]], agrams)) / total



# Models

def n_gram(n, s):
    """ n gram model
    input: string of text s
    output: list of n-grams from text
    NB to join sublists, [" ".join(i) for i in ngrams]
    >>> n_gram(2, "the rain in Spain falls mainly on the plain")
    ['the rain', 'rain in', 'in Spain', 'Spain falls', 'falls mainly', 'mainly on', 'on the', 'the plain']
    >>> n_gram(2, "the rain in Spain falls mainly in Spain")
    ['the rain', 'rain in', 'in Spain', 'Spain falls', 'falls mainly', 'mainly in', 'in Spain']
    >>> n_gram(3, "the rain in Spain falls mainly on the plain")
    ['the rain in', 'rain in Spain', 'in Spain falls', 'Spain falls mainly', 'falls mainly on', 'mainly on the', 'on the plain']
    >>> n_gram(1, "the rain in Spain falls mainly on the plain")
    ['the', 'rain', 'in', 'Spain', 'falls', 'mainly', 'on', 'the', 'plain']
    """
    words = s.split(" ")
    ngrams = [words[i:i+n] for i in range(len(words)-(n-1))]
    return [" ".join(i) for i in ngrams]


def skip_gram(k, n, s):
    """ skip grams
    input: k (skip), n (as in n-gram), s string text
    output: list of skip grams
    NB this is a bit trickier to do in one line...
    >>> skip_gram(1, 2, "the rain in Spain falls mainly on the plain")
    ['the in', 'rain Spain', 'in falls', 'Spain mainly', 'falls on', 'mainly the', 'on plain']
    >>> skip_gram(1, 3, "the rain in Spain falls mainly on the plain")
    ['the in falls', 'rain Spain mainly', 'in falls on', 'Spain mainly the', 'falls on plain']
    >>> skip_gram(2, 2, "the rain in Spain falls mainly on the plain")
    ['the Spain', 'rain falls', 'in mainly', 'Spain on', 'falls the', 'mainly plain']
    >>> skip_gram(2, 3, "the rain in Spain falls mainly on the plain")
    ['the Spain on', 'rain falls the', 'in mainly plain']
    """
    words = s.split(" ")
    sgrams = [list([words[i]]) + [words[i+k+j] for j in range(1,n+k,k+1)] for i in range(len(words)-((1+k)*(n-1)))]
    return [" ".join(i) for i in sgrams]


def bigram_predict(t, s):
    """ predict next word based on tokens
    input: tokens t, string s
    output: ngram+1 (where 1 is highest probable next word based on tokens)
    NB tokens must consist of bigrams or larger, string input can be a unigram
    empty string returned on no match
    >>> bigram_predict({'the rain': 2, 'in spain': 2, 'rain in': 2, 'spain falls': 1, 'falls mainly': 1, 'mainly in': 1}, 'in the')
    'rain'
    >>> bigram_predict({'the rain': 2, 'in spain': 2, 'rain in': 2, 'spain falls': 1, 'falls mainly': 1, 'mainly in': 1}, 'in bed')
    ''
    """
    start_word = (s.split(" "))[-1]
    token_tuples = sorted(t.items(), key=lambda x: -x[1])
    next_word = ''
    for i in token_tuples:
        ngram = i[0].split(" ")
        if start_word == ngram[-2]:
            next_word = ngram[-1]
            break
    return next_word





