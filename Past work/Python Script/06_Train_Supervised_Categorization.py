# Dataframe
import pandas as pd
import numpy as np

# NLP
from gensim.models import Word2Vec
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import PolynomialFeatures

import pickle
from io import StringIO
from math import sqrt

# Read lemma
para= pd.read_csv('../../../Processing/Out/Explore/paragraphs_for_language.csv',delimiter=";")
text_type="text_lemma"#"text_lemma_tag"
para=para[["para_id",text_type]]

# Read KP qualitative
kp_qual= pd.read_csv('../../../Processing/Out/Language/Themes/KP_Poc_Rol_0503.csv',delimiter=";")
df=para.merge(kp_qual,how="right",on="para_id")
cols=["Poc","Rol","Perf"]

# For 3 themes
for col in cols:
    df["predict_"+col]=""
    df=df.loc[df[text_type].isnull()==False]
    df["kp"]=[1 if item=="yes" else 0 for item in df["KP_"+col]]

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',  ngram_range=(1, 2))
    features = tfidf.fit_transform(df[text_type]).toarray()
    X_train, X_test, y_train, y_test = train_test_split(df[text_type], df['kp'],test_size=0.2, random_state = 0)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    # Test different models
    # models = [
    #     RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    #     LinearSVC(),#Linear Support Vector Classification.
    #     MultinomialNB(),#Naive Bayes classifier for multinomial models
    #     LogisticRegression(random_state=0),
    # ]
    # CV = 10
    # cv_df = pd.DataFrame(index=range(CV * len(models)))
    # entries = []
    # for model in models:
    #     model_name = model.__class__.__name__
    #     accuracies = cross_val_score(model, features, df["kp"], scoring='accuracy', cv=CV)
    #     for fold_idx, accuracy in enumerate(accuracies):
    #         entries.append((model_name, fold_idx, accuracy))
    # cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

    # cv_df.groupby('model_name').accuracy.mean()

    # choose the best one
    model = LinearSVC()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, df["kp"], df.index, test_size=0.5, random_state=0)
    model.fit(X_train, y_train)
    y_v=model.predict(X_train)
    y_pred = model.predict(X_test)
    res = sqrt(mean_squared_error(y_test, y_pred))

    print("Linear SVC Accuracy Score " + col,accuracy_score(y_pred, y_test)*100)

# To use for qualitative results

    comp=0
    for i in y_test.index:
        df.loc[i,"predict_"+col]=y_pred[comp]
        comp+=1

    comp=0
    for i in y_train.index:
        df.loc[i,"predict_"+col]=y_v[comp]
        comp+=1

folder_out="Report_for_Analysis/Explore/"

df.to_csv(folder_out+"Predict_SVC.csv",sep=";", index=False, decimal=",",float_format='%.3f')



