import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    Tokenization function to process text data:
        - Apply word_tokenize from nltk.tokenize to tokeninze the words
        - lemmatize the tokens, strip to remove any extra space.
        - return the cleaned tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Categories_Table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    index webpage displays cool visuals and receives user input text for model.
    '''
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #Genre and weather_related counts
    weather_related1 = df[df['weather_related']==1].groupby('genre').count()['message']
    weather_related0 = df[df['weather_related']==0].groupby('genre').count()['message']
    genre_names = list(weather_related1.index)

    # Distribution of categories with 1
    cat_dist1 = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()/len(df)

    #sorting
    cat_dist1 = cat_dist1.sort_values(ascending = False)

    #Values that have 0 in categories
    cat_dist0 = (cat_dist1 -1) * -1
    cat_names = list(cat_dist1.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=weather_related1,
                    name = 'Weather Related',
                    marker = dict(
                            color = 'rgb(0, 100, 100)'
                                )
                ),
                Bar(
                    x=genre_names,
                    y= weather_related0,
                    name = 'Not Weather Related',
                    marker = dict(
                            color = 'rgb(100, 100, 100)'
                                )
                )
            ],

            'layout': {
                'title': 'Distribution of messages by Genre and \'Weather Related\'',
                'yaxis': {
                    'title': "Weather Related Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode' : 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_dist1,
                    name = 'Label 1',
                    marker = dict(
                            color = 'rgb(200, 100, 100)'
                                )
                    
                ),
                Bar(
                    x=cat_names,
                    y=cat_dist0,
                    name = 'Label 0',
                    marker = dict(
                            color = 'rgb(200, 200, 100)'
                                )
                    
                )
            ],

            'layout': {
                'title': 'Distribution of labels within categories',
                'yaxis': {
                    'title': "Distribution Percentage"
                },
                'xaxis': {
                    'title': "Categories",
            
                },
                'barmode' : 'stack'
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
    '''
    Web page that handles user query and displays model results.
    '''
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