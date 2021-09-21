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
* toxicCommentClassification.ipynb : Jupyter notebook
* ML_model : folder 
   * model.py : Python script that saves a Linear SVC model trained on the dataset
   * f1_score_test.py : Python script that evaluates the f1_score on the test dataset
   * dev_tests.py : Python script with predictions of toxicity
   * user.py : Python script with predictions of toxicity for a console entry


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

