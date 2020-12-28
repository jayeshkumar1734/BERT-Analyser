import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop=set(stopwords.words('english'))
from nltk.util import ngrams
import string
from collections import defaultdict
import re
from sklearn.model_selection import train_test_split

train_df=pd.read_csv("./train.csv")
test_df=pd.read_csv("./test.csv")
df_train=train_df.copy()
df_test=test_df.copy()
# Dropping single null value in train data set
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)
string=r"#$%&'!()*+,-.:;<=>?@[\]^_\"`{|}~"
df_train["text"]=df_train["text"].apply(str)
df_test["text"]=df_test["text"].apply(str)
df_train['text']=df_train['text'].apply(lambda x : x.lower())
df_test['text']=df_test['text'].apply(lambda x : x.lower())

def remove_URL(text):
	  return re.sub(r"https?://\S+|www\.\S+", "", text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_punct(text):
    table=str.maketrans('\t',' ',string)
    return text.translate(table)


def fix_Plan(location):
    letters_only = re.sub("[^a-zA-Z]",  # Search for all non-letters
                          " ",          # Replace all non-letters with spaces
                          str(location))     # Column and row to search    

    words = letters_only.lower().split()     
    stops = set(stopwords.words("english"))      
    meaningful_words = [w for w in words]      
    return (" ".join(meaningful_words))       

num_responses_train = df_train["text"].size    
clean_Plan_responses_train = []
num_responses_test = df_test["text"].size    
clean_Plan_responses_test = []

  
def giveout_words_labels():

    df_test['text']=df_test['text'].apply(lambda x : remove_URL(x))
    df_train['text']=df_train['text'].apply(lambda x : remove_URL(x))
    df_train['text']=df_train['text'].apply(lambda x : remove_html(x))
    df_test['text']=df_test['text'].apply(lambda x : remove_html(x))
    df_train['text']=df_train['text'].apply(lambda x : remove_punct(x))
    df_test['text']=df_test['text'].apply(lambda x : remove_punct(x))
	
	
    for i in range(0,num_responses_train):
        clean_Plan_responses_train.append(fix_Plan(df_train["text"].values[i]))
    for j in range(0,num_responses_tets):
          clean_Plan_responses_test.append(fix_Plan(df_test["text"].values[j]))   
    text_train=[]
    for i in range(len(clean_Plan_responses_train)):
        result = re.sub(r"^\s+", "", clean_Plan_responses_train[i])
        text_train.append(re.sub(r"\s+$", "", result))
    text_test=[]
    for i in range(len(clean_Plan_responses_test)):
        result = re.sub(r"^\s+", "", clean_Plan_responses_test[i])
        text_test.append(re.sub(r"\s+$", "", result))
    for i in range(df_train.shape[0]):
        if df_train["sentiment"].values[i]=="neutral":
            df_train["sentiment"].values[i]=0
        if df_train["sentiment"].values[i]=="positive":
            df_train["sentiment"].values[i]=1
        if df_train["sentiment"].values[i]=="negative":
            df_train["sentiment"].values[i]=2
    labels_train = df_train.sentiment.values

    for i in range(df_test.shape[0]):
        if df_test["sentiment"].values[i]=="neutral":
            df_test["sentiment"].values[i]=0
        if  df_test["sentiment"].values[i]=="positive":
            df_test["sentiment"].values[i]=1
        if df_test["sentiment"].values[i]=="negative":
            df_test["sentiment"].values[i]=2

    labels_test = df_test.sentiment.values

    return text_train,labels_train,text_test,labels_test
