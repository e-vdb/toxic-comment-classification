# toxic-comment-classification

## Summary

Multi-label classification of comments

## Dataset

Wikipedia comments rated in 6 categories of *toxicity*.

* toxic
* severe_toxic
* obscene
* threat
* insult
* identity_hate

(dataset available on https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

## Repository content

* model.py
* f1_score_test.py
* dev_tests.py
* user.py

## Jupyter notebook

* Exploratory data analysis
* Data preprocessing
* Natural Language Processing 
  * standard CountVectorizer   
  * Tfidf Vectorizer
* Machine learning methods
  * Multinomial Naive Bayes Classifier
  * Linear Support Vector Machine Classifier
  * Logistic Regression
* Evaluation of the f1 score   
* Parameters tuning with GridSearch CV

