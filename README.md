# **Twitter Sentiment Analysis Using Machine Learning**

## **Overview**

This project involves building a machine learning model to perform sentiment analysis on tweets. The objective is to classify the sentiments of tweets as positive or negative based on their textual content. The project uses the Sentiment140 dataset from Kaggle and includes preprocessing, feature extraction, and model building to analyze sentiment trends on Twitter.

## **Project Highlights**

**Data Preprocessing**: Performed data cleaning such as removing stopwords, punctuation, and stemming the text using NLP techniques to prepare the data for modeling.

**Feature Engineering**: Applied text vectorization using techniques such as TF-IDF and Bag of Words to convert the text into numerical features.

**Modeling**: Built and evaluated machine learning models such as Logistic Regression, Naive Bayes, and Support Vector Machine (SVM) to classify the sentiment of tweets.

**Evaluation**: Assessed the model performance using accuracy, precision, recall, and F1-score to select the best-performing model.

## **Dataset Information**

**Sentiment140 Dataset**

Source: Kaggle

Contains 1.6 million tweets labeled for sentiment analysis (0 for negative and 4 for positive).

## **Columns:**

|**Column Name**	|   **Description**|

| target	        |   Sentiment of the tweet (0 = Negative, 4 = Positive)|

|  text	          |   The actual tweet content|

## **Insights**

**Tweet Sentiments**: The model successfully classifies tweets into positive or negative sentiments with high accuracy, offering valuable insights into public opinion trends on social media.

**Common Words**: Most frequent words associated with negative sentiment include "bad," "sad," and "hate," while positive sentiment words include "happy," "love," and "great."

**Model Performance**: The Logistic Regression model achieved 79% accuracy on training data and 77% accuracy on test data.

## **Key Steps**

**Dataset Import**: Downloaded the dataset from Kaggle and loaded it into a Pandas DataFrame.
**Data Cleaning:**
Converted text to lowercase.

Removed special characters, stopwords, and performed stemming.

**Text Vectorization**: Used TF-IDF to convert the text into numerical features for model training.

**Model Building**: Implemented various machine learning algorithms including Logistic Regression, Naive Bayes, and SVM for sentiment classification.

**Model Evaluation**: Evaluated the models using accuracy, precision, recall, and F1-score.

## **Results**

**Best Model**: Logistic Regression with 79% training accuracy and 77% test accuracy.
