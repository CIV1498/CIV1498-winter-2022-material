import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from sklearn.metrics import f1_score, plot_confusion_matrix, classification_report

words = ['drug', 'bank', 'prescription', 'memo', 'private',
         'body', 'business', 'html', 'money', 'offer', 'please']


def train(f_name='emails.csv', model_name='forest', save_model=True):
    """"
    train a classifier based on model_name
    model_name: ['log_reg', 'tree', 'forest']
    f_name: reads csv file from
    save_model: if true save the model next to the script
    """
    # Read Data
    email_data = _read_data(f_name)
    #Imputation
    _preprocess_data(email_data)
    # Split Data
    train, test = train_test_split(email_data, test_size=0.1, random_state=0, stratify=email_data['label'])
    # Feature Extraction
    X_train = _word_detector(words, train['email'])
    y_train = train[['label']]
    X_test = _word_detector(words, test['email'])
    y_test = test[['label']]
    # Model Building
    if model_name == 'log_reg':
        log_reg = LogisticRegression()
        model = log_reg.fit(X_train, y_train)
    elif model_name == 'tree':
        tree_ = tree.DecisionTreeClassifier()
        model = tree_.fit(X_train, y_train)
    else:
        forest = RandomForestClassifier(random_state=0)
        model = forest.fit(X_train, y_train)
        
    print('[INFO] Training done!')
    # Visualize Evaluations on Training Data
    _report_results(X_train, y_train, model)
    print('[INFO] Training data stats:')
    # Visualize Evaluations on Test Data
    _report_results(X_test, y_test, model)
    print('[INFO] Test data stats:')
    # Save Model
    # save the model to disk
    if save_model:
        file_name = 'model.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)
        print(f'[INFO] Model saved as {file_name}')
        
    return model


def predict(data, model='', model_fname='model.pkl'):
    """
    data: pd.DataFrame
    """
    # Transform
    # lower case
    data['email'] = data['email'].str.lower()
    X = _word_detector(words, data['email'])
    # Load Model
    if not model:
        with open(model_fname, 'rb') as f:
            model = pickle.load(f)
    # Predict
    y_pred = model.predict(X)
    # Report Label
    return y_pred


# train_helper
def _read_data(f_name='emails.csv'):
    """
    reads the input file in csv and returns a dataframe
    the csv file has to contain 'subject',  'email', 'target' columns
    """
    email_data = pd.read_csv(f_name, index_col=0)
    # lower case
    email_data['subject'] = email_data['subject'].str.lower()
    email_data['email'] = email_data['email'].str.lower()
    
    return email_data


def _preprocess_data(email_data):
    """ fill na 'subject' and 'email' columns with empty"""
    email_data[['subject', 'email']] = email_data[['subject', 'email']].fillna('')
    

def _word_detector(words, texts):
    """
    Returns a DataFrame with detections of words.

    Parameters:
        words (list): A list of words to look for.
        texts (Series): A series of strings to search in.

    Returns:
        (DataFrame): A DataFrame with len(words) columns and texts.shape[0] rows.
    """

    # Write your code here
    detections = 1 * np.array([texts.str.contains(word) for word in words]).T

    return pd.DataFrame(index=texts.index, data=detections, columns=words)


def _report_results(X, y, model):
    """
    Report the scores and confusion matrix for *X*, *y*, and the estimator *model*
    """
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
    plot_confusion_matrix(model, X, y)
    print(f'f_score = {f1_score(y, y_pred)}')

