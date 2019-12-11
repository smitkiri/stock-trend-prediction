from pandas_datareader import data
import pandas as pd
import numpy as np
import datetime
import re

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical


def add_datepart(df, fldname, drop=True):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    if drop: df.drop(fldname, axis=1, inplace=True)


def get_stock_data(start, end, company):
  stock_df = data.DataReader(company, 'yahoo', start, end) # getting numerical stock data like daily open and close
  stock_df = stock_df.reset_index()

  stock_df['Date'] = pd.to_datetime(stock_df['Date'])
  return stock_df



def get_stock_labels(start, end, company, cutoff, n_labels = 3, shift = 0):
  """
    returns labels indicating stock price change
  """ 
  stock_df = get_stock_data(start, end, company)
  
  stock_df.index = stock_df['Date']
  
  stock_df['Close_new'] = stock_df['Close'].shift(-shift) # shifting close values to difference with open
  
  stock_df = stock_df.dropna()
  
  stock_df['diff'] = (stock_df['Close_new'] - stock_df['Open']).div(stock_df['Open'], axis=0)*100

  """
    labels: 1 for increase, 2 for decrease, 0 for no change
  """
  if n_labels == 3:
    stock_df['y'] = np.where(stock_df['diff'] >= cutoff, 1, np.where(stock_df['diff'] <= -cutoff, 2, 0))
  else:
    stock_df['y'] = np.where(stock_df['diff'] >= cutoff, 1, 0)
  stock_df = stock_df['y']
  return stock_df

def get_sequential_stock_data(start, end, company):
  
  stock_df = get_stock_data(start, end, company)

  add_datepart(stock_df, 'Date', drop=False) # Feature engineering
  
  stock_df.index = stock_df['Date']

  stock_df = stock_df.drop(['Date', 'Adj Close'], axis=1)
  
  """
  One-hot encoding categorical features
  """
  stock_df = pd.concat(	
      [stock_df, 
       pd.get_dummies(stock_df['Dayofweek'], prefix = 'dayofweek'), 
       pd.get_dummies(stock_df['Week'], prefix = 'week'), 
       pd.get_dummies(stock_df['Month'], prefix = 'month')], 
       axis=1)
  
  stock_df.drop(['Dayofweek', 'Dayofyear', 'Month', 'Year', 'Day', 'Week'], axis=1, inplace=True)
  
  return stock_df
  

def prepare_sequential_stock_data(stock_df, labels, prev, validation_split = 0.2):
  
  if sorted(stock_df.index) != sorted(labels.index):
  	raise ValueError("Dates are not identical in stock data and labels")

  stock_df = stock_df.sort_index()
  labels = labels.sort_index()

  dataset = stock_df.values
  labels = labels.values

  """
  Scaling features
  """
  scaler = MinMaxScaler(feature_range=(0,1))
  dataset = scaler.fit_transform(dataset)
  
  split = int((1-validation_split)*len(dataset))

  """
  Preparing training data
  """
  X_train, y_train = [], []
  for i in range(prev, split):
    X_train.append(dataset[i-prev:i])
    y_train.append(labels[i-1])

  X_train, y_train = np.array(X_train), np.array(y_train)

  """
  Shuffle training data
  """
  ind = np.arange(len(X_train))
  np.random.shuffle(ind)

  X_train = X_train[ind]
  y_train = y_train[ind]


  """
  Preparing test data
  """
  X_test, y_test = [], []
  for i in range(0, len(dataset[split:])):
  	X_test.append(dataset[split+i-prev:split+i])
  	y_test.append(labels[split+i-1])

  X_test, y_test = np.array(X_test), np.array(y_test)

  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)

  return X_train, y_train, X_test, y_test

def normalize_headline(row):
    row = str(row)
    result = row.lower()
    #Delete useless character strings
    result = result.replace('b\'','')
    result = result.replace('b"','')
    result = result.replace('...', '')
    whitelist = set('abcdefghijklmnopqrstuvwxyz 0123456789.,;\'-:?')
    result = ''.join(filter(whitelist.__contains__, result))
    return result

def prepare_data(data_df, text_col, label_col, top_words = 2000, validation_split = 0.2, max_review_length = 100):
  np.random.seed(0)

  headlines = data_df[text_col].values
  tokenizer = Tokenizer(top_words)
  tokenizer.fit_on_texts(headlines)

  split = int((1-validation_split)*len(headlines))

  X = tokenizer.texts_to_sequences(headlines)
  y = data_df[label_col].values

  idx = np.arange(len(X))
  np.random.shuffle(idx)

  X = np.array(X)[idx]
  y = np.array(y)[idx]

  X_train = X[:split]
  y_train = y[:split]

  X_test = X[split:]
  y_test = y[split:]
 
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)

  X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
  X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

  return X_train, X_test, y_train, y_test

def plot_roc_curve(y_true, y_pred_prob, fname = 'figure'):
  y_true = label_binarize(y_true, classes = [0, 1, 2])
  n_classes = y_true.shape[1]
  
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

  for i in range(n_classes):
      plt.plot(fpr[i], tpr[i],
              label='ROC curve of class {0} (area = {1:0.2f})'
              ''.format(i, roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc="lower right")
  plt.savefig(fname)

def print_metrics(y_true, y_pred, y_pred_prob = None, fname = 'figure'):
  print("Accuracy: {:.3f}".format(accuracy_score(y_true, y_pred)))
  print("Precision: {:.3f}".format(precision_score(y_true, y_pred, average = "micro")))
  print("Recall: {:.3f}".format(recall_score(y_true, y_pred, average = "micro")))
  print("F1 score: {:.3f}".format(f1_score(y_true, y_pred, average = "micro")))
  if y_pred_prob is not None:
    print("\n\n")
    plot_roc_curve(y_true, y_pred_prob, fname)

def data_summary(data):
    print("datasize: ", len(data))
    print("No. of columns : ", len(data.columns))
    print("\nColumn List:",list(data.columns),"\n")
    print("\n data glimpse\n")
    print(data.head())