# toxic-comment-classification

## Summary

Multi-label classification of Wikipedia comments. 
The purpose is to predict the *toxicity* of these comments among 6 categories :

* toxic
* severe_toxic
* obscene
* threat
* insult
* identity_hate

## Dataset

Dataset available on https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

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
  * Standard CountVectorizer   
  * Tfidf Vectorizer
* Multi-label classifier using a MultiOutputClassifier from sklearn.multioutput
* Machine learning algorithms
  * Multinomial Naive Bayes Classifier
  * Linear Support Vector Machine Classifier
  * Logistic Regression
* Evaluation of the f1 score (weighted)   
* Parameters tuning with GridSearch CV

## TASK LIST
- [x] Implement machine learning algorithm using Scikit-learn
- [x] Implement deep learning algorithm using Keras
- [ ] Deploy model
