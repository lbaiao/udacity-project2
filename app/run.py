import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, ngrams
import nltk

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

import numpy as np


nltk.download(['stopwords'])


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = list(stopwords.words('english'))  # About 150 stopwords

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok.isalpha() and clean_tok not in stop_words:
            clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/database1.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/model-test.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # get the lengths of all messages
    messages_lens = [len(i.split(' ')) for i in df['message']]
    msgs_lens_counts, msgs_lens = np.histogram(messages_lens)
    # count how many times each token appears
    tokens = []
    for i in df['message']:
        tokens += tokenize(i)
    all_counts = FreqDist(ngrams(tokens, 1))
    lemmas, lemmas_counts = zip(*all_counts.most_common(20))
    lemmas = [i[0] for i in lemmas]

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
        # messages lengths visual
        {
            'data': [
                Bar(
                    x=msgs_lens,
                    y=msgs_lens_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Lengths',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Length"
                }
            }
        },
        # tokens appearances
        {
            'data': [
                Bar(
                    x=lemmas,
                    y=lemmas_counts
                )
            ],

            'layout': {
                'title': 'Top 20 Most Common Tokens',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Token"
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
