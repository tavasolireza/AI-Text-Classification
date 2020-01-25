from nltk import ngrams
from collections import Counter, defaultdict


def read_train_data():
    file = open("data/HAM-Train.txt", 'r')
    ff = open("data/HAM-Train.txt", 'r').readline()

    train = file.readlines()
    bigram_art_model = defaultdict(lambda: defaultdict(lambda: 0))
    unigram_art_model = defaultdict(lambda: 0)
    art_data = ""
    bigram_politic_model = defaultdict(lambda: defaultdict(lambda: 0))
    unigram_politic_model = defaultdict(lambda: 0)
    politic_data = ""
    bigram_social_model = defaultdict(lambda: defaultdict(lambda: 0))
    unigram_social_model = defaultdict(lambda: 0)
    social_data = ""
    bigram_sport_model = defaultdict(lambda: defaultdict(lambda: 0))
    unigram_sport_model = defaultdict(lambda: 0)
    sport_data = ""
    bigram_economy_model = defaultdict(lambda: defaultdict(lambda: 0))
    unigram_economy_model = defaultdict(lambda: 0)
    economy_data = ""
    for line in train:
        if line.startswith("اقتصاد@@@@@@@@@@"):
            text = line.strip().split("@@@@@@@@@@")[1].strip()
            economy_data += text + " "
        elif line.startswith("سیاسی@@@@@@@@@@"):
            text = line.strip().split("@@@@@@@@@@")[1].strip()
            politic_data += text + " "
        elif line.startswith("اجتماعی@@@@@@@@@@"):
            text = line.strip().split("@@@@@@@@@@")[1].strip()
            social_data += text + " "
        elif line.startswith("ورزش@@@@@@@@@@"):
            text = line.strip().split("@@@@@@@@@@")[1].strip()
            sport_data += text + " "
        elif line.startswith("ادب و هنر@@@@@@@@@@"):
            text = line.strip().split("@@@@@@@@@@")[1].strip()
            art_data += text + " "
    bigram_model(bigram_economy_model, economy_data)
    unigram_model(unigram_economy_model, economy_data)
    bigram_model(bigram_sport_model, sport_data)
    unigram_model(unigram_sport_model, sport_data)
    bigram_model(bigram_art_model, art_data)
    unigram_model(unigram_art_model, art_data)
    bigram_model(bigram_social_model, social_data)
    unigram_model(unigram_social_model, social_data)
    bigram_model(bigram_politic_model, politic_data)
    unigram_model(unigram_politic_model, politic_data)
    # for i in bigram_economy_model:
    #     print(str(i), bigram_economy_model[i].items())

    # for i in unigram_economy_model:
    #     print(str(i), unigram_economy_model[i])


def unigram_model(model, text):
    text = text.split()
    m = len(text)
    for w in ngrams(text, 1):
        model[w[0]] += 1
    for w in model:
        model[w] /= m
        print(w, model[w])


def bigram_model(model, text):
    for w1, w2 in ngrams(text.split(), 2):
        model[w1][w2] += 1
    for w11 in model:
        total_count = float(sum(model[w11].values()))
        for w22 in model[w11]:
            model[w11][w22] /= total_count


def run():
    read_train_data()


if __name__ == '__main__':
    read_train_data()
