# example format: {'review_id': 'LZp4UX5zK3e-c5ZGSeo3kA', 'user_id': 'msQe1u7Z_XuqjGoqhB0J5g', 'business_id': 'jtQARsP6P-LbkyjbO1qNGg', 'stars': 1, 'date': '2014-10-23', 'text': 'Terrible. Dry corn bread. Rib tips were all fat and mushy and had no flavor. If you want bbq in this neighborhood go to john mulls roadkill grill. Trust me.', 'useful': 3, 'funny': 1, 'cool': 1}

import json
import re
import nltk
# nltk.download('averaged_perceptron_tagger') # need to uncomment on first run

posfile = open("positive-words.txt")
negfile = open("negative-words.txt")

positive = [ line.rstrip() for line in posfile.readlines()]
negative = [ line.rstrip() for line in negfile.readlines()]

# list of nltk word types
nountypes = ["NN", "NNS", "NNP", "NNPS"]
verbtypes = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
adjtypes = ["JJ", "JJR", "JJS"]

def get_tags(sentence):
    text = sentence.split(" ")
    return nltk.pos_tag(text)

def select_replace(sentence):
    newsent = ""
    for ch in sentence:
        if ch.isalnum() or ch == '\'' or ch == '/' or ch == '\\' or ch == ' ' or ch == '.':
            newsent += ch
        else:
            newsent += ' '
    return newsent

def get_pos_neg_counts(sentence):
    # print (sentence)
    pos = 0 # num positive words
    neg = 0 # num negative words
    countexclaim = 0 # num ! characters
    numnouns = 0 # num nouns in sentence
    numverbs = 0 # num verbs in sentence
    numadj = 0 # num adjectives in sentence

    countexclaim += sentence.count('!')
    numsentences = sentence.count(". ")

    # sentence = re.sub('[^0-9a-zA-Z]+', ' ', sentence)
    sentence = select_replace(sentence)

    sentence = sentence.replace("...", ". ")
    # re.sub(' +',' ',sentence)

    sentence = " ".join(sentence.split()) # remove excess whitespace
    sentence = sentence.rstrip()

    print ('"' + sentence + '"')

    # sentence.replace("  ", " ")

    print (sentence[:20])

    sentarray = get_tags(sentence)
    # print (sentarray)

    numwords = len(sentarray) # total num words

    for sent in sentarray:
        word = sent[0]
        speech = sent[1] # nltk part of speech
        # word = re.sub('\W+', ' ', word) # remove non-alpha characters

        if word in positive:
            pos += 1
        elif word in negative:
            neg += 1

        if speech in nountypes:
            numnouns += 1
        elif speech in verbtypes:
            numverbs += 1
        elif speech in adjtypes:
            numadj += 1

    return [numnouns, numverbs, numadj, countexclaim, pos, neg, numwords, numsentences]

file = open("yelp_academic_dataset_review.json", 'r')
yelpdata = open("yelpdata.csv", "w")

line = file.readline()
# for i in range(20):
while line:
    jdata = json.loads(line)

    stars = jdata['stars']
    text = jdata['text']
    # print (text)

    if "??" in text:
        line = file.readline()
        continue

    counts = get_pos_neg_counts(text) + [stars]

    newline = ",".join(str(x) for x in counts) + "\n"
    # print (newline)
    yelpdata.write(newline)

    line = file.readline()
