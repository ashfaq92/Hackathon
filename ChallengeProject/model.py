import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random


# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]


def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1


def get_r0():
    return random.choice([2, 2.1, 4, 4.1, 10, 18])


def get_similar_epidemic(r_0):
    if r_0 == 2:
        return "Ebola"
    if r_0 == 2.1:
        return "Hepatitis C"
    elif r_0 == 4:
        return "HIV"
    elif r_0 == 4.1:
        return "SARS"
    elif r_0 == 10:
        return "Mumps"
    elif r_0 == 18:
        return "Measels"


def check_similarity(new_gene):
    virus3 = pd.read_table(new_gene)
    virus1 = pd.read_table('virus1_data.txt')
    virus2 = pd.read_table('virus2_data.txt')

    virus1['words'] = virus1.apply(lambda x: getKmers(x['sequence']), axis=1)
    virus1 = virus1.drop('sequence', axis=1)
    virus2['words'] = virus2.apply(lambda x: getKmers(x['sequence']), axis=1)
    virus2 = virus2.drop('sequence', axis=1)
    virus3['words'] = virus3.apply(lambda x: getKmers(x['sequence']), axis=1)
    virus3 = virus3.drop('sequence', axis=1)

    virus1_texts = list(virus1['words'])
    for item in range(len(virus1_texts)):
        virus1_texts[item] = ' '.join(virus1_texts[item])
    y_h = virus1.iloc[:, 0].values  # y_h for virus1

    virus2_texts = list(virus2['words'])
    for item in range(len(virus2_texts)):
        virus2_texts[item] = ' '.join(virus2_texts[item])
    y_c = virus2.iloc[:, 0].values                       # y_c for virus2

    virus3_texts = list(virus3['words'])
    for item in range(len(virus3_texts)):
        virus3_texts[item] = ' '.join(virus3_texts[item])
    y_d = virus3.iloc[:, 0].values

    cv = CountVectorizer(ngram_range=(4, 4))
    X = cv.fit_transform(virus1_texts)
    X_virus2 = cv.transform(virus2_texts)
    X_virus3 = cv.transform(virus3_texts)

    X_train, X_test, y_train, y_test = train_test_split(X, y_h, test_size=0.20, random_state=42)

    ### Multinomial Naive Bayes Classifier ###
    # The alpha parameter was determined by grid search previously
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, y_train)

    # y_pred = classifier.predict(X_test)

    # print("Confusion matrix\n")
    # print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))

    # Predicting the sequences
    y_pred_virus3 = classifier.predict(X_virus3)

    accuracy, precision, recall, f1 = get_metrics(y_d, y_pred_virus3)
    # print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
    r0 = get_r0()
    similar_epidemic = get_similar_epidemic(r0)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "r_0": r0, "similar_epidemic": similar_epidemic}


file = 'virus3_data.txt'
res = check_similarity(file)
print(res)
