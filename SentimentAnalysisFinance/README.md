# Boost your NLP classification task with pre-trained embedding and DeepLearning

Henri Bouxin
Publication date: 6th may 2020

## Motivation
**Text information is everywhere**: in social media, user reviews or open forums, news, e-mail, etc. In today's context, many Natural Langage Processing techniques are here to help you **extract the value from all this textual information and can lead to many exciting use cases**:
- Predict a stock price movement with a sentiment index based on twitter data,
- Understand the pain points of customer during their journeys thanks to social media or forums,
- Etc.

The goal was therefore to acquire hands-on experience in implementing an NLP classification task on financial news headlines using sklearn, Keras and Tensorflow-Hub

## Project description
The goal of the project is to classify financial news headlines according to three sentiment levels: negative, neutral, positive. Such a classifier could be very helpful if you want to build a sentiment index based on news for a given sector for example. I focused my work on trying to answer the following questions:
- What are the financial headlines I should pay attention to in terms of their very positive or negative tone?
- To what extent can I trust such a classifier?

## Repo structure
- **Text_classification.ipynb** : jupyter notebook with all the differents steps of the project with the following table of contents:
  - `I. Data Exploration of the financial news headlines, and first feature engineering`
  - `II. Machine learning standard approach for sentiment classification`
  - `III. More advanced approach leveraging on Deep Learning, Embeddings and RNN`

## Package Used
This project requires Python 3.x and the following Python libraries installed:
- NLTK
- sklearn
- Tensorflow
- Keras
- Seaborn

## Conclusion and further improvements
- `Key takeaways`:
  - Learn how to implement a classification task (sentiment analysis, topic classification) using different approaches: standard machine learning vs. deep-learning
  - Learn how to boost your model with pre-trained embeddings either with `Tensorflow-Hub` or with others available embeddings like `Glove`
  - Learn how to implement an `LSTM` model from scratch


- `Further Improvements`:
  - Fine tune the different models in order to better increase the performance vs. the baseline model
  - Strengthen the baseline model using pre-defined sentiment analysis package like the textblob library
  - Implement another kind of best-in-class NLP models like BERT

## Blog Post
You can also find more details on the project by following the link to my blogpost: `https://medium.com/@henri.bouxin/boost-your-nlp-classification-tasks-with-pre-trained-embedding-and-deeplearning-72982c52792d`

## Licensing, Acknowledgements
- This project was completed as part of the `Udacity Data Scientist Nanodegree`. Data were provided by Udacity. The data was originally sourced by Udacity from Starbucks.
