# Dataframe
import pandas as pd
import numpy as np

import random
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import stopwords

import guidedlda
import pyLDAvis.gensim
import pyLDAvis.sklearn
import pyLDAvis

# Read paragraphs
paragraphs= pd.read_csv(r"D:/DPO_Minusca/Processing/Out/Explore/paragraphs_all.csv",delimiter=";")
paragraphs=paragraphs.loc[paragraphs.text_lemma_tag.isnull()==False]

num_topic=17

# Define Seed for initialization
text_type="text_lemma_tag"
liste_text=list(paragraphs[text_type])

tokenized_docs = [word_tokenize(doc.lower()) for doc in liste_text]
dictionary = Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

vectorizer = CountVectorizer(ngram_range=(1, 3))
X = vectorizer.fit_transform(liste_text)
X = X.astype(np.int32)
word2id= Dictionary(tokenized_docs)
#X_array = Y.toarray()

model = guidedlda.GuidedLDA(n_topics=num_topic, n_iter=100, random_state=7, refresh=20)
model.fit(X)
tf_feature_names = vectorizer.get_feature_names()
vocab = vectorizer.get_feature_names()
word2id = dict((v, idx) for idx, v in enumerate(vocab))

# Guided LDA with seed topics.
themes_defined= pd.read_csv(r"D:/DPO_Minusca/Processing/Lists/theme_list.csv",delimiter=";")

seed_words=themes_defined["GuidedLDA"]
seed_words=seed_words[seed_words.notnull()]
seed_words= [item.split(",")  for item in seed_words]
seed_topic_list=[[item.strip().lower() for item in lis] for lis in seed_words]
not_in_voc=[[item for item in lis if item not in vocab] for lis in seed_topic_list]
seed_topic_list=[[item for item in lis if item in vocab] for lis in seed_topic_list]


model = guidedlda.GuidedLDA(n_topics=num_topic, n_iter=100, random_state=7, refresh=20)
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)
# model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)
n_top_words = 15
topic_word = model.topic_word_

df=pd.DataFrame(columns=["Topic_Value","Words"])
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    val=topic_word[i][np.argsort(topic_dist)][:-(n_top_words+1):-1]
 
    print('Values {}: {}'.format(str(i), str(val).replace("\n","")))
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    a=pd.Series([i,str(val).replace("\n","")], index=df.columns )
    df=df.append(a , ignore_index=True)
    a=pd.Series([i,' '.join(topic_words)], index=df.columns )
    df=df.append(a , ignore_index=True)

df.to_csv(r"D:/DPO_Minusca/Processing/Out/Language/Guided_LDA/coefficients_for_viz_last_themes.csv", sep=";",encoding="UTF-8")
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(model, X, vectorizer, mds='tsne')

pyLDAvis.save_html(panel, r"D:/DPO_Minusca/Processing/Out/Language/Guided_LDA/GuidedLDA_last_themes.html")
pyLDAvis.show(panel)
print("there")