import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    '''
    Load the data:
        - Using the database filepath, load the data in a dataframe.
        - Split the data into messages and categories and assign to X and Y.
        - Extract the categories column names and assign to category_names.
        - Function return X, Y, category_names.
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Categories_Table', engine)  
    X = df.message.values
    Y = df[df.columns.difference(['id', 'message','original', 'genre'])].values
    
    category_names = list(df.columns.difference(['id', 'message','original', 'genre']))
    return X, Y, category_names

def tokenize(text):
    '''
    Tokenization function to process text data:
        - Remove all characters which are not alphanumeric. Convert all letters to lowercase.
        - Apply word_tokenize from nltk.tokenize to tokeninze the words
        - lemmatize the tokens, strip to remove any extra space.
        - Remove the stopwords
        - return the cleaned tokens
    '''
    text = re.sub(r"[^a-z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        if clean_tok not in stopwords.words("english"):
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Building the model to train and create a classifier:
        - Create a pipeline with CountVectorizer, TfidfTransformer and MultiOutputClassifier as RandomForestClassifier.
        - To further improve the model, define set of parameters and perform a GridSearchCV to identify the best parameters.
        - Return the model with best parameters.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    model = pipeline
    
    parameters = {
        'vect__analyzer': ['word'],
        'vect__max_features': [5, 50],
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_split': [2, 5]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model created:
        - Using the test data (X_test), predict the categories using the fitted model.
        - Print classification report using classification_report from sklearn.metrics.
    '''
    y_pred = model.predict(X_test)
    
    y_pred_df = pd.DataFrame (y_pred, columns = category_names)
    Y_test_df = pd.DataFrame (Y_test, columns = category_names)
    for col in category_names: 
        print("Category Name:", col)
        print(classification_report(Y_test_df.loc[:,col], y_pred_df.loc[:,col]))

def save_model(model, model_filepath):
    '''
    Save the model under the filepath provided.
    '''
    filename = 'finalized_model.sav'
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    main function uses the functions defined earlier:
        - Splits the data into a training set and a test set. 
        - Creates a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model.
        - Model uses the message column to predict classifications for 36 categories (multi-output classification). 
        - Finally, it exports the model to a pickle file. 
    '''
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