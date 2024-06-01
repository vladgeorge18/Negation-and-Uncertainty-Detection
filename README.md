# NLP_Project

## Introduction

In this project we will implement three different Sequence Tagging techniques
to solve the task of Detection of Negation and Uncertainty:

• A Rule-Based algorithm using basic text processing tools.

• A Machine-Learning system.

• A Deep Learning system.

The goal of this task is to find any words that indicate the start of negated (such
as “no” or “without”) or uncertain (such as “maybe”, “perhaps” or modal verbs)
clauses and finding the scope of text that is affected by them. We will work
on a dataset that contains reports of medical exams written in Catalan and
Spanish

## Data

The dataset we will be working with is a collection of real medical notes that
have been manually labelled to find negation cues and negated sequences of
words. 
The dataset is packaged in two JSON files.

The training set can be found in train_data.json and the test set in test_data.json.

The only information we care about is the text itself and
the labelling of the negation/uncertainty words and scopes. 


## REPORT

In report.pdf can be found the synthesis of all the three parts. Each algorithm is thoroughly explained and the results are being presented along with our conclusions.

## PART 1

For the first part the file rule_based_model.ipynb is needed. It also uses a library, called CUTEXT, comprised of popular Spanish medical terms that we imported through the terms_raw.txt file.

## PART 2

The Machine Learning model is implemented in the CRF_model.ipynb file. As mentioned in the name, this approach highlights a CRF trained with various features such as PoS, lemma, surrounding words and many other.

## PART 3

The Deep Learning model in implemented in the Deep_Learning_approach.ipynb file. This approach contains a neural network architecture based on CNN and BiLSTM layers that receieves as input features as word at character level, word itself, PoS, position of the word in the sentence in order to predict the TAG.   


