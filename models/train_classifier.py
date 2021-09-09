import sys
from sqlalchemy import create_engine
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from stopwords import get_stopwords
from pickle import dump
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk


nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """Load the data from a SQLite database.

    Parameters
    ---
    database_filepath: str
        SQLite database filepath.

    Returns
    ---
    X: pandas.DataFrame
        Input data for the ML pipeline.
    y: pandas.DataFrame
        Target data for the ML pipeline.
    y.columns: list
        Messages' categories.
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('InsertTableName', engine)
    df = df.dropna()
    X = df['message']
    y = df.drop(columns=['id', 'message', 'genre'])
    return X, y, y.columns


def tokenize(text):
    """Tokenizes the text, by applying NLTK tokenization,
    along with lemmatization. It also removes symbols and
    punctuation, along with stop words.

    Parameters
    ---
    text: str
        Text to be tokenized.

    Returns
    ---
    clean_tokens: list[str]
        Clean tokens extracted from `text`.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()        
    nltk_stop_words = list(stopwords.words('english')) # About 150 stopwords
    stop_words = list(get_stopwords('en'))
    stop_words.extend(nltk_stop_words)
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok.isalpha() and clean_tok not in stop_words:                        
            clean_tokens.append(clean_tok)

    return clean_tokens    

def build_model():
    """Build the ML pipeline and structures the
    grid search process.

    Returns
    ---
    cv: sklearn.model_selection.GridSearchCV
        Grid search object for model training.
    """
    pipeline = Pipeline([            
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),    
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__criterion': ['gini', 'entropy']
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
                      
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the trained model and print a report with
    the accuracy, recall and f1 score metrics.

    Parameters
    ---
    model: sklearn.model_selection.GridSearchCV
        Grid search object with the fitted model.
    X_test: pandas.DataFrame
        Input data for testing.
    Y_test: pandas.DataFrame
        Target data for testing.
    category_names: list[str]
        List with the category names.
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred,
                                target_names=category_names))


def save_model(model, model_filepath):
    """Saves the model to a pickle file.

    Parameters
    ---
    model: sklearn.model_selection.GridSearchCV
        Grid search object with the fitted model.
    model_filepath: str
        Pickle file path.
    """
    with open(model_filepath, 'wb') as f:
        dump(model, f)


def main():
    """ Runs the ML pipeline."""
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
