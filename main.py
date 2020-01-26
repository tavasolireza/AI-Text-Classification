from nltk import ngrams
from collections import defaultdict
import math

classification = []


def measure(predicted):
    c = (0, 1, 2, 3, 4)
    counter = [0, 0, 0, 0, 0]
    recall_counter = [0, 0, 0, 0, 0]
    precision_counter = [0, 0, 0, 0, 0]
    for cls in c:
        for pair in predicted:
            if pair[0] == cls:
                if pair[1] == cls:
                    counter[cls] += 1
                else:
                    recall_counter[cls] += 1
            elif pair[1] == cls:
                precision_counter[cls] += 1

    for i in range(5):
        p = counter[i] / (counter[i] + precision_counter[i])
        r = counter[i] / (counter[i] + recall_counter[i])
        f = (2 * p * r) / (p + r)
        print(f"Class{i}:", "Precision:", p)
        print("           Recall:", r)
        print("        F-measure:", f)
        print()


def evaluate_test(model, text, class_probability):
    p = [0, 0, 0, 0, 0]
    class_list = ["اقتصاد", "ورزش", "ادب و هنر", "اجتماعی", "سیاسی"]
    original_class = text[0].strip()
    test_data = text[1].strip()

    for m in range(len(model)):
        for word in ngrams(test_data.split(), 2):
            p_data = model[m][word[0]][word[1]]
            if p_data != 0:
                p[m] += math.log10(p_data)
            elif p_data == 0:
                p[m] += math.log10(0.00000000001)
        p[m] += math.log10(class_probability[m])

    classification.append([class_list.index(original_class), p.index(max(p))])


def read_test_data(model, class_probability):
    test_data = open("data/HAM-Test.txt", 'r')
    data = test_data.readlines()

    for line in data:
        line = line.strip().split("@@@@@@@@@@")
        evaluate_test(model, line, class_probability)

    measure(classification)


def read_train_data():
    file = open("data/HAM-Train.txt", 'r')
    # original_class, test_data = read_test_data()

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

    text_length = len(train)
    c_eco, c_sport, c_art, c_social, c_politic = 0, 0, 0, 0, 0

    for line in train:
        if line.startswith("اقتصاد@@@@@@@@@@"):
            c_eco += 1
            text = line.strip().split("@@@@@@@@@@")[1].strip()
            economy_data += text + " "
        elif line.startswith("سیاسی@@@@@@@@@@"):
            c_politic += 1
            text = line.strip().split("@@@@@@@@@@")[1].strip()
            politic_data += text + " "
        elif line.startswith("اجتماعی@@@@@@@@@@"):
            c_social += 1
            text = line.strip().split("@@@@@@@@@@")[1].strip()
            social_data += text + " "
        elif line.startswith("ورزش@@@@@@@@@@"):
            c_sport += 1
            text = line.strip().split("@@@@@@@@@@")[1].strip()
            sport_data += text + " "
        elif line.startswith("ادب و هنر@@@@@@@@@@"):
            c_art += 1
            text = line.strip().split("@@@@@@@@@@")[1].strip()
            art_data += text + " "
    unigram_model(unigram_economy_model, economy_data)
    unigram_model(unigram_sport_model, sport_data)
    unigram_model(unigram_art_model, art_data)
    unigram_model(unigram_social_model, social_data)
    unigram_model(unigram_politic_model, politic_data)

    bigram_model(bigram_economy_model, economy_data, unigram_economy_model)
    bigram_model(bigram_sport_model, sport_data, unigram_sport_model)
    bigram_model(bigram_art_model, art_data, unigram_art_model)
    bigram_model(bigram_social_model, social_data, unigram_social_model)
    bigram_model(bigram_politic_model, politic_data, unigram_politic_model)

    read_test_data(
        (bigram_economy_model, bigram_sport_model, bigram_art_model, bigram_social_model, bigram_politic_model), (
            c_eco / text_length, c_sport / text_length, c_art / text_length, c_social / text_length,
            c_politic / text_length))


def unigram_model(model, text):
    text = text.split()
    m = len(text)
    for w in ngrams(text, 1):
        model[w[0]] += 1
    for w in model:
        model[w] /= m


def bigram_model(model, text, model2):
    l1, l2 = 0.8, 0.2
    for w1, w2 in ngrams(text.split(), 2):
        model[w1][w2] += 1
    for w11 in model:
        total_count = float(sum(model[w11].values()))
        for w22 in model[w11]:
            model[w11][w22] /= total_count
            model[w11][w22] = l1 * model[w11][w22] + l2 * model2[w11]


def run():
    read_train_data()


if __name__ == '__main__':
    read_train_data()
