import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import joblib

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


def load_data(database_filepath):
    '''
    input:
        database_filepath: File path where sql database was saved.
    output:
        X: Training message List.
        Y: Training target.
        category_names: Categorical name for labeling.
    '''
    ## Load the data from the database, and split label vs. features
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('messages', engine)  
    X = df["message"]
    Y = df.drop(labels=["id","message", "original", "genre"], axis=1)
    
    # Implement corrections in the output data set so that the multioutput classification algorithm can work
    Y_true = Y.drop(labels=["child_alone"], axis=1) # drop since it only has 0's values
    Y_true["related"] = Y["related"].apply(lambda x: 1 if x > 0 else 0) # since our classification algorithm expect binary outputs
    
    category_names = Y_true.columns.values.tolist()
    
    return X, Y_true, category_names


def tokenize(text):
    '''
    input:
        text: Message data for tokenization.
    output:
        clean_tokens: Result list after tokenization.
    '''
        
    # Normalise by setting to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Create tokens 
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    # Lemmatise words
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    return clean_tokens

def build_model(grid_search_is_true=False):
    """
    Creates machine learning pipeline for learning
    :Input:
        grid_search_is_true: False by defautl since it takes on average one hour
    :Returns:
        :pipeline: Machine Learning pipeline with fit/predict methods
    """
    
    basic_model = LogisticRegression(random_state=42)
    basic_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(basic_model, n_jobs=-1))
        ])
    
    if grid_search_is_true ==True:
        parameters = {'clf__estimator__C': np.linspace(1,10,3),
             'clf__estimator__penalty': ['l1', 'l2']}
        gridsearch_pipeline = GridSearchCV(basic_pipeline, param_grid=parameters, scoring='precision_samples', cv = 3)
        basic_pipeline = gridsearch_pipeline
    
    return basic_pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Takes model and evaluates it against X and Y test with the category names
        
    :Input:
        :model: Model trained on X_train and Y_train
        :X_test: Dataframe, validation data for model
        :Y_test: Dataframe, actual labels for the test data in X
        :category_names: List of strings, categories to be evaluated
    :Returns:
        :None: Prints out report to terminal
    """
    
    Y_pred = model.predict(X_test)
    output_classes = category_names
    for class_idx,class_name in enumerate(Y_test):
        print("Column {}: {}".format(class_idx, class_name))

        y_true = Y_test[class_name].values.tolist()
        y_pred = Y_pred[:,class_idx].tolist()
        target_names = ['is_{}'.format(class_name), 'is_not_{}'.format(class_name)]
        print(classification_report(y_true, y_pred, target_names=target_names))

def save_model(model, model_filepath):
    """
    Saves model as a pickle file to model_filepath
    
    :Input:
        :model: pipeline/model, to be pickled for later use
        :model_filepath: String, filepath where model will be saved
    :Returns:
        :None: Pickle file will be created at model_path
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()