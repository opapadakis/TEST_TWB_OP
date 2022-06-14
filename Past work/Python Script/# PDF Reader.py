# PDF Reader
from PyPDF2 import PdfFileWriter, PdfFileReader
import fitz
from pdfminer.high_level import extract_text
 # Table from PDF
import camelot

# Word manipulation
from pluralizer import Pluralizer
import inflection
#from pattern.en import pluralize, singularize

# Dataframe
import pandas as pd
import numpy as np



# Regex
import re

# NLP
from gensim.models import Word2Vec
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
ps = PorterStemmer()
wnl = WordNetLemmatizer()

# Modelling
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
import pickle

# Generate unique identifier
import uuid

# File manipulation
import os
import io
import glob

# Format manipulation
import operator
import string


df=pd.read_csv('../../../Processing/Out/Explore/paragraphs_all.csv',delimiter=";")
df["verbs"]=""
#df["text_lower"]=[item.lower() for item in df["text_para"]]

for phrase_id in df.index:
    phrase=df.text_lemma_tag[phrase_id]
    verbs = []
    if isinstance(phrase,str):
        word_tokens = word_tokenize(phrase)
        sentence=sent_tokenize(phrase)
        tag=nltk.pos_tag(word_tokens)
        pos_tags = [pos_tag for _,pos_tag in tag]
        fd = nltk.FreqDist(pos_tags)
        
        for word,pos in tag:
            if (pos.find("VB")!=-1):
                verbs.append(word +" "+ pos+",")

        df.verbs[phrase_id]="".join(verbs)
        df.verbs[phrase_id]=df.verbs[phrase_id][:-1]

df=df[["file_name","page","text_para","para_id","title","text_lemma_tag","recommendation","verbs"]]
df=df[df["recommendation"]!="No"]
df=df[df["recommendation"]!="Unlikely"]
df.to_csv("../../../Processing/Out/Language/theme_KP.csv",sep=";",index=False)


