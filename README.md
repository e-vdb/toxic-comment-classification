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
* ML_model : folder of Python scripts
   * model.py : saves a Linear SVC model trained on the dataset with Pickle (finalized_model.sav)
   * f1_score_test.py : loads trained model and evaluates the f1_score on the test dataset
   * dev_tests.py : loads trained model and gives predictions of toxicity
   * user.py : loads trained model and gives predictions of toxicity for a console entry


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

