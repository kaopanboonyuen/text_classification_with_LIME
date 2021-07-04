#!/usr/bin/env python
# coding: utf-8

# # Text Classification using Nueral Network and LIME
# 
# Local Interpretable Model-Agnostic Explanations
# 
# ![](https://d33wubrfki0l68.cloudfront.net/b342157befa54829f056658dddcd2e897062417d/46b71/static/afa7e0536886ee7152dfa4c628fe59f0/5040b/text_process_prediction.png)
# 
# Author: Kao Panboonyuen

# In[1]:


import numpy as np
import pandas as pd
import sklearn

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from pythainlp import word_tokenize
from tqdm import tqdm_notebook
from pythainlp.ulmfit import process_thai

import warnings
warnings.filterwarnings("ignore")


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Collection, Callable, Tuple

__all__ = ['top_feats_label', 'top_feats_all', 'plot_top_feats']

def top_feats_label(X: np.ndarray, features: Collection[str], label_idx: Collection[bool] = None,
                    min_val: float = 0.1, agg_func: Callable = np.mean)->pd.DataFrame:
    '''
    original code (Thomas Buhrman)[from https://buhrmann.github.io/tfidf-analysis.html]
    rank features of each label by their encoded values (CountVectorizer, TfidfVectorizer, etc.)
    aggregated with `agg_func`
    :param X np.ndarray: document-value matrix
    :param features Collection[str]: feature names
    :param label_idx Collection[int]: position of rows with specified label
    :param min_val float: minimum value to take into account for each feature
    :param agg_func Callable: how to aggregate features such as `np.mean` or `np.sum`
    :return: a dataframe with `feature`, `score` and `ngram`
    '''
    res = X[label_idx] if label_idx is not None else X
    res[res < min_val] = 0
    res_agg = agg_func(res, axis=0)
    df = pd.DataFrame([(features[i], res_agg[i]) for i in np.argsort(res_agg)[::-1]])
    df.columns = ['feature','score']
    df['ngram'] = df.feature.map(lambda x: len(set(x.split(' '))))
    return df

def top_feats_all(X: np.ndarray, y: np.ndarray, features: Collection[str], min_val: float = 0.1, 
                  agg_func: Callable = np.mean)->Collection[pd.DataFrame]:
    '''
    original code (Thomas Buhrman)[from https://buhrmann.github.io/tfidf-analysis.html]
    for all labels, rank features of each label by their encoded values (CountVectorizer, TfidfVectorizer, etc.)
    aggregated with `agg_func`
    :param X np.ndarray: document-value matrix
    :param y np.ndarray: labels
    :param features Collection[str]: feature names
    :param min_val float: minimum value to take into account for each feature
    :param agg_func Callable: how to aggregate features such as `np.mean` or `np.sum`
    :return: a list of dataframes with `rank` (rank within label), `feature`, `score`, `ngram` and `label`
    '''
    labels = np.unique(y)
    dfs = []
    for l in labels:
        label_idx = (y==l)
        df = top_feats_label(X,features,label_idx,min_val,agg_func).reset_index()
        df['label'] = l
        df.columns = ['rank','feature','score','ngram','label']
        dfs.append(df)
    return dfs

def plot_top_feats(dfs: Collection[pd.DataFrame], top_n: int = 25, ngram_range: Tuple[int,int]=(1,2),)-> None:
    '''
    original code (Thomas Buhrman)[from https://buhrmann.github.io/tfidf-analysis.html]
    plot top features from a collection of `top_feats_all` dataframes
    :param dfs Collection[pd.DataFrame]: `top_feats_all` dataframes
    :param top_n int: number of top features to show
    :param ngram_range Tuple[int,int]: range of ngrams for features to show
    :return: nothing
    '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(top_n)
    for i, df in enumerate(dfs):
        df = df[(df.ngram>=ngram_range[0])&(df.ngram<=ngram_range[1])][:top_n]
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("score", labelpad=16, fontsize=14)
        ax.set_title(f"label = {str(df.label[0])}", fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.score, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        ax.invert_yaxis()
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# In[4]:


data_df = pd.read_csv('CIND1_2.csv')
# Show the top 5 rows
display(data_df.head())
# Summarize the data
data_df.describe()


# # Start (Split Train/Test)

# In[5]:


from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(data_df, test_size=0.15, random_state=2021)
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)


# In[6]:


y = data_df.target


# In[7]:


train_df.head()


# In[8]:


train_df.shape


# In[9]:


train_df.target.value_counts() / train_df.shape[0]


# In[10]:


#dependent variables
y_train = train_df["target"]
y_valid = valid_df["target"]


# In[11]:


#text faetures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer(tokenizer=process_thai, ngram_range=(1,2), min_df=20, sublinear_tf=True)
tfidf_fit = tfidf.fit(data_df["headline"])
# text_train = tfidf_fit.transform(train_df["headline"])
# text_valid = tfidf_fit.transform(valid_df["headline"])
# text_train.shape, text_valid.shape


# # TFIDF (Transform)

# In[12]:


from sklearn.model_selection import train_test_split

text_train, text_test, y_train, y_test = train_test_split(data_df, y,
                                                          random_state=42,
                                                          test_size=0.25,
                                                          stratify=y)


# In[13]:


X_train_tfidf =  tfidf_fit.transform(text_train["headline"])
X_test_tfidf = tfidf_fit.transform(text_test["headline"])

X_train_tfidf.shape, X_test_tfidf.shape


# In[14]:


# #visualize texts
# # from visualize import top_feats_all, plot_top_feats
# features = tfidf_fit.get_feature_names()
# %time ts = top_feats_all(text_train.toarray(), y_train, features)
# print(ts[0].shape)
# ts[0].head()


# In[15]:


# %time plot_top_feats(ts)


# # Train Model (Neural Network Model)
# 
# ![](https://scikit-learn.org/stable/_images/multilayerperceptron_network.png)

# In[16]:


from sklearn.neural_network import MLPClassifier


# In[17]:


print(X_train_tfidf.shape, X_test_tfidf.shape)

rf = lf = MLPClassifier(solver='lbfgs', 
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2), 
                        random_state=1)

rf.fit(X_train_tfidf, y_train)


# In[18]:


# print(X_train_tfidf.shape, X_test_tfidf.shape)

# rf = RandomForestClassifier()

# rf.fit(X_train_tfidf, y_train)


# In[19]:


print("Test  Accuracy : %.2f"%rf.score(X_test_tfidf, y_test))
print("Train Accuracy : %.2f"%rf.score(X_train_tfidf, y_train))
print()
print("Confusion Matrix : ")
print(confusion_matrix(y_test, rf.predict(X_test_tfidf)))
print()
print("Classification Report")
print(classification_report(y_test, rf.predict(X_test_tfidf)))


# In[20]:


def pred_fn(text):
    text_transformed = tfidf_fit.transform(text)
    return rf.predict_proba(text_transformed)

pred_fn(text_test.headline)


# In[21]:


from lime import lime_text

# train_df.target.unique() = (['STABLE', 'UP', 'DOWN'] 

explainer = lime_text.LimeTextExplainer(class_names=(['STABLE', 'UP', 'DOWN']))
explainer


# In[22]:


train_df.target.unique()


# In[24]:


idx = 0

text_test.headline
text_test.headline.values.reshape(-1,1)[idx][0]


# # Explainable AI

# In[30]:


import random 

idx = random.randint(1, len(text_test))

print("Actual Text : ", text_test.headline)

print("Prediction : ", rf.predict(X_test_tfidf[1].reshape(1,-1))[0])
print("Actual :     ", y_test)

for i in range(0,2):
    explanation = explainer.explain_instance(text_test.headline.values.reshape(-1,1)[i][0], classifier_fn=pred_fn)
    explanation.show_in_notebook()


# # Explainable AI - True Type

# In[31]:


preds = rf.predict(X_test_tfidf)

true_preds = np.argwhere((preds == y_test.values)).flatten()


# In[32]:


idx  = random.choice(true_preds)

print("Actual Text : ", text_test.headline)

print("Prediction : ", rf.predict(X_test_tfidf[1].reshape(1,-1))[0])
print("Actual :     ", y_test)

for i in range(0,2):
    explanation = explainer.explain_instance(text_test.headline.values.reshape(-1,1)[i][0], classifier_fn=pred_fn)
    explanation.show_in_notebook()


# # Explainable AI - False Type

# In[33]:


preds = rf.predict(X_test_tfidf)

false_preds = np.argwhere((preds != y_test.values)).flatten()


# In[34]:


idx  = random.choice(false_preds)

print("Actual Text : ", text_test.headline)

print("Prediction : ", rf.predict(X_test_tfidf[1].reshape(1,-1))[0])
print("Actual :     ", y_test)

for i in range(0,2):
    explanation = explainer.explain_instance(text_test.headline.values.reshape(-1,1)[i][0], classifier_fn=pred_fn)
    explanation.show_in_notebook()

