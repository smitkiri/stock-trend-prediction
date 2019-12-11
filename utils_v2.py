from pandas_datareader import data
import datetime
import pandas as pd
import re
from nltk.corpus import stopwords
import numpy as np

from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import pickle
 
## to get date features 
def add_datepart(df, fldname, drop=True):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)

        
# to get additional stock features        
def get_stock_add_data(stock_df):
    add_datepart(stock_df, 'Date', drop=False) # Feature engineering
    stock_df = stock_df.drop(['Adj Close'], axis=1)
    """
    One-hot encoding categorical features
    """
    stock_df = pd.concat([stock_df, 
                          pd.get_dummies(stock_df['Dayofweek'], prefix = 'dayofweek'), 
                          pd.get_dummies(stock_df['Week'], prefix = 'week'), 
                          pd.get_dummies(stock_df['Month'], prefix = 'month')], 
                         axis=1)
    stock_df.drop(['Dayofweek', 'Dayofyear', 'Month', 'Year', 'Day', 'Week', 'Elapsed'], axis=1, inplace=True)
    return stock_df

# to get stock data
def get_stock_data(start, end, company, additional_features=False):
    stock_df = data.DataReader(company, 'yahoo', start, end) # getting numerical stock data like daily open and close
    stock_df = stock_df.reset_index()
    if additional_features:
        stock_df = get_stock_add_data(stock_df)
    stock_df = stock_df.sort_values('Date').reset_index(drop=True)
    return stock_df

# to get stock labels
def get_stock_change_labels(start, end, company, cutoff, n_labels = 3, shift = 0):
    """
    returns labels indicating stock price change
    """ 
    stock_df = get_stock_data(start, end, company)
    stock_df.set_index('Date', inplace=True) 

    stock_df['Close_new'] = stock_df['Close'].shift(-shift) # shifting close values to difference with open
    stock_df = stock_df.dropna()

    stock_df['diff'] = (stock_df['Close_new'] - stock_df['Open']).div(stock_df['Open'], axis=0)*100 # change

    """
    labels: 1 for increase, 2 for decrease, 0 for no change
    """
    label_name = 'change_' + str(shift)
    if n_labels == 3:
        stock_df[label_name] = np.where(stock_df['diff'] >= cutoff, 1, np.where(stock_df['diff'] <= -cutoff, 2, 0))
    else:
        stock_df[label_name] = np.where(stock_df['diff'] >= cutoff, 1, 0)
    
    stock_df = stock_df.reset_index()   
    stock_df = stock_df[['Date', label_name]]
    stock_df = stock_df.sort_values('Date').reset_index(drop=True)
    
    return stock_df


# to prepare sequential data
def prepare_sequential_stock_data(stock_df,label, prev, scaling =True):
  
    # getting values in array format
    dataset = stock_df[[col for col in stock_df.columns if col != label]].values
    labels = stock_df[label].values

    """
    Scaling features
    """
    if scaling:
        scaler = MinMaxScaler(feature_range=(0,1))
        dataset = scaler.fit_transform(dataset)
   
    # Preparing sequential data
 
    X, y = [], []
    for i in range(prev, len(dataset)+1):
        arr = dataset[i-prev:i,:].copy() # getting prev days data
        #arr[-1,:3] = arr[-2,:3]*(1+(arr[-2,:3] - arr[-3,:3])/arr[-3,:3]) # updating first 3 cols of last row, assuming no data  
        X.append(arr)
        y.append(labels[i-1])

    return np.array(X), np.array(y)



def get_words_contractions():
    # A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    return { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"}


# to clean the text
def get_clean_text(text, modify_contractions = True, remove_stopwords = True):
    '''Remove unwanted characters and format the text to create fewer nulls word embeddings'''
    
    text = str(text)
    text = text.lower()
    
    text = text.replace('b\'','')
    text = text.replace('b"','')
    text = text.replace('...', '')
    
    contractions = get_words_contractions()

    # Replace contractions with their longer forms 
    if modify_contractions:
        text = text.split() # converting sentence to list of words
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text) # converting list of words back to sentence 
    
    # Format words and remove unwanted characters
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'0,0', '00', text) 
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)
    
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


def get_word_embeddings(file):
    # Load GloVe's embeddings
    embeddings_index = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0] # getting word 
            embedding = np.asarray(values[1:], dtype='float32') # getting embeddings
            embeddings_index[word] = embedding
    
    return embeddings_index
    
## prepares vocabulary for text col in dataframe   
def get_corpus_vocab(df, text_col):
    vocab = {}
    for i in range(len(df)):
        for word in df.iloc[i][text_col].split():
            if word not in vocab:
                vocab[word] = 1
            else: 
                vocab[word] += 1
    return vocab

# to print top n dictionary elements
def print_n_dict_items(dict_, n):
    for x in list(dict_)[:n]:
        print ("key: {}, key count: {} ".format(x,  dict_[x]))

# text to sequence conversion based on given word indexing
def convert_text_to_indexes(text, word_indexes):
    text_words_indexes = [] 
    for word in text.split(): # iterating over words in a text
        if word in word_indexes: # if word present in word_indexes
            text_words_indexes.append(word_indexes[word]) 
        else: # if word not in word indexes
            text_words_indexes.append(word_indexes["<UNK>"]) 
    return text_words_indexes


# to get count of total words(repeated) and total unknown words(repeated not in vocab)
def get_word_unk_cnt(text, word_indexes):
    word_cnt = 0 # to count total number of words in text
    unk_cnt = 0 # to count total number of unknown words in text
    for word in text.split(): # iterating over words in a text
        word_cnt += 1 
        if word not in word_indexes: # if word not present in word_indexes
            unk_cnt += 1
    return word_cnt, unk_cnt


# to make fix length word index list
def limiting_text_length(word_indexes, limit_length, pad_index):
    l = len(word_indexes) # no of words
    if l >= limit_length:
        return np.asarray(word_indexes[:limit_length])
    else: 
        n_pads = limit_length - l
        return np.asarray(word_indexes + [pad_index]*n_pads) # returning with addition of pad_index 

    
# to limit test of max length and add pad at end
def add_pad(word_indexes, max_length, pad_index):
    l = len(word_indexes) # no of words
    if l >= max_length:
        return np.asarray(word_indexes[:max_length]+[pad_index])
    else: 
        return np.asarray(word_indexes + [pad_index]) # returning with addition of pad_index 

# preprocessing of text      
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

def get_data_summary(data):
    print("datasize: ", len(data))
    print("No. of columns : ", len(data.columns))
    print("\nColumn List:",list(data.columns),"\n")
    print("\n data glimpse\n")
    print(data.head())
    
    
def save_pickle(filename, variable):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(variable, f)

def open_pickle(filename):
    with open(filename + '.pickle', 'rb') as f:
        return pickle.load(f)


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

