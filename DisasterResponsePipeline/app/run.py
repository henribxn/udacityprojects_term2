import json
import plotly
import pandas as pd
import re
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import joblib


app = Flask(__name__)

def tokenize(text):  
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

# load data
engine = create_engine('sqlite:///../data/DisasterDB.db')
df = pd.read_sql_table('messages', engine).drop(labels=["child_alone"], axis=1)

# load model
model = joblib.load("../models/model_classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Show distribution of different category
    category = list(df.columns[4:])
    category_counts = []
    for column_name in category:
        category_counts.append(np.sum(df[column_name]))
        
    # Show to 10 percentage categories
    sorted_queries = df[df.columns[4:]].sum(axis=0).sort_values(ascending=False)
    sorted_queries_pct = round(sorted_queries/sorted_queries.sum()*100,1)
    top_10_classes = sorted_queries_pct[0:10]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
         {
            'data': [
                Bar(
                    x=top_10_classes.index.values.tolist(),
                    y=top_10_classes.values.tolist()
                )
            ],

            'layout': {
                'title': 'Top 10 distribution of Message Categories in percentage',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()