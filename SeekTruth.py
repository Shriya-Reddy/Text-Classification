# SeekTruth.py : Classify text objects into two categories
#
# PLEASE PUT YOUR NAMES AND USER IDs HERE
# Submitted by : Name : SHRIYA REDDY PULAGAM, Usename : spulagam
#                Name : SRINIVAS YASHVANTH VALAVALA, Usename : svalaval
#                Name : SRI VENKATA SAI ANOOP BULUSU, Usename : srbulusu
# Based on skeleton code by D. Crandall, October 2021
#

import sys
import math
from decimal import Decimal


def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated classlabel for test_data["objects"][i]
#
# Do not change the return type or parameters of this function!
#
def classifier(train_data, test_data):
    #print(train_data["objects"])
    a = 1
    truthful = dict()
    bag = dict()
    deceptive = dict()
    no_of_words_truthful = 0
    no_of_words_deceptive = 0
    test_data_labels = []
    truthful_prob = dict()
    deceptive_prob = dict()
    likelihood_deceptive = 1
    likelihood_truthful = 1
    # This is just dummy code -- put yours here!
    count_deceptive = train_data["labels"].count("deceptive")
    count_truthful = train_data["labels"].count("truthful")
    count_total = count_deceptive + count_truthful

    for i in range(len(train_data["objects"])):
        for word in train_data["objects"][i].split(' '):
            if word not in truthful.keys():
                truthful[word] = 0
            if word not in deceptive.keys():
                deceptive[word] = 0
            if word not in bag.keys():
                bag[word] = 0
            if train_data["labels"][i] == "truthful":
                truthful[word] = truthful[word] + 1
                no_of_words_truthful+=1
            else:
                deceptive[word] = deceptive[word] + 1
                no_of_words_deceptive+=1
            bag[word] = bag[word] + 1
    total_words = no_of_words_deceptive + no_of_words_truthful
    for word in truthful.keys():
        truthful_prob[word] = (truthful[word]+a)/(count_truthful+total_words+2)
    for word in deceptive.keys():
        deceptive_prob[word] = (deceptive[word]+a)/(count_deceptive+total_words+2)

    prior_deceptive = count_deceptive/count_total
    prior_truthful = 1-prior_deceptive
    for i in range(0,len(test_data["objects"])):
        likelihood_deceptive = 1
        likelihood_truthful = 1
        for word in test_data["objects"][i].split(' '):
            likelihood_deceptive = likelihood_deceptive * Decimal(deceptive_prob.get(word,a/(count_deceptive+total_words+2)))   #calculating likelihood probability that a given review is deceptive
            likelihood_truthful= likelihood_truthful * Decimal(truthful_prob.get(word,a/(count_truthful+total_words+2)))     #calculating likelihood probability that a given review is truthful   
        prob_spam_given_a_word = (likelihood_deceptive )
        prob_not_spam_given_a_word = (likelihood_truthful)
        if Decimal(prob_not_spam_given_a_word/prob_spam_given_a_word) > 1 :
            test_data_labels.append("truthful")
        else :
            test_data_labels.append("deceptive")
    test_data["labels"] = test_data_labels
    return test_data_labels


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    #print(train_data["labels"])
    test_data = load_file(test_file)
    if(sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    results= classifier(train_data, test_data_sanitized)

    # calculate accuracy
    correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
