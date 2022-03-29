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

    #Imputation

    # Split Data

    # Feature Extraction

    # Model Building
        
    print('[INFO] Training done!')
    # Visualize Evaluations on Training Data

    print('[INFO] Training data stats:')
    # Visualize Evaluations on Test Data

    print('[INFO] Test data stats:')
    # Save Model
    # save the model to disk

        
    return 


def predict(data, model='', model_fname='model.pkl'):
    """
    data: pd.DataFrame
    """
    # Transform
    # lower case

    # Load Model

    # Predict

    # Report Label
    return 


# train_helper
def _read_data(f_name='emails.csv'):
    """
    reads the input file in csv and returns a dataframe
    the csv file has to contain 'subject',  'email', 'target' columns
    """
    # lower case

    return 


def _preprocess_data(email_data):
    """ fill na 'subject' and 'email' columns with empty"""
    

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

    return


def _report_results(X, y, model):
    """
    Report the scores and confusion matrix for *X*, *y*, and the estimator *model*
    """


