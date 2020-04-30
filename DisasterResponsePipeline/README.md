# Disaster Response Pipeline Project
Author: Henri Bouxin

Start Date: 2020 April 30th

## Project description
When natural disasters happen, people use social media (like twitter) or else to try and get help. However, not all communications are natural disaster tweets and **there isn't a simple way to use key words in order to identify these kinds of texts**.

That is why, the objective is to **deploy a machine learning application in order to be able to classify a communication between many categories** so that to be able to react more quickly.

Such a pipeline works can work with any kind of data as long as there are labelled data concerning your classification needs.

There are three steps:
1. Create an ETL which cleans the Data
2. Create a ML pipeline which performs feature extraction using NLP techniques, and trains a model
3. Take the model and embed it into a Flask webapp

### Instructions to run the code:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py "data/disaster_messages.csv" "data/disaster_categories.csv" "data/DisasterDB.db"`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py "data/DisasterDB.db" "models/model_classifier.pkl"`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Repo Structure
- **app**
  - template
  - master.html  # main page of web app
  - go.html  # classification result page of web app
  - run.py  # Flask file that runs app

- **data**
  - disaster_categories.csv  # data to process
  - disaster_messages.csv  # data to process
  - process_data.py
  - DisasterDB.db   # database to save clean data to

- **models**
  - train_classifier.py
  - model_classifier.pkl  # saved model

## Packed Used :
- sys
- pandas
- sqlalchemy
- joblib
- re
- nltk
- sklearn
- json
- plotly
- flask

## To go further -- Potential improvements
- The **classification task being imbalanced**, it could be important to train to fit a classification less prone to class imbalanced like a GBMClassifier, or try to implement StratifiedKfold for multiple Outputs, or re-weight the classes (class_weight="balanced" for a Random Forest Classification)
- In order to improve the performance of our model (i.e. recall and precision metrics), we could try other techniques such as :
  - **deep learning model relying on a pre-trained embeddings** (`https://www.tensorflow.org/tutorials/keras/text_classification_with_hub`)
  - **RNN model** in order to take into account the context of the sentences (you could try to adapt the LSTM model implemented in the Deep Learning Specialization: `https://github.com/Kulbear/deep-learning-coursera/blob/master/Sequence%20Models/Emojify%20-%20v2.ipynb`)
