# Create Model and dictionary for word embeddings with word2vec

# Dataframe
import pandas as pd
import numpy as np
import numpy.linalg as la

# NLP
from gensim.models import Word2Vec
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords

# save model
import pickle

# Folder where the files will be saved
folder_out="Report_for_Analysis/Explore/"

paragraphs= pd.read_csv(folder_out+"paragraphs_for_language.csv",delimiter=";")
# Realized on real words, could be on lemmatize version, using tags or not
type_text="text_tag"  # 'text_para', 'text_lower', 'text_tag','text_stem', 'text_lemma', 'text_lemma_tag'
paragraphs=paragraphs.loc[paragraphs[type_text].isnull()==False]
liste_text=list(paragraphs[type_text])

# For some columns it is need
documents=[ remove_stopwords(item) for item in liste_text]

tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
dictionary = Dictionary(tokenized_docs)
dictionary.token2id
dictionary.token2id.get("actor")
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

word_tokens = [nltk.word_tokenize(sent) for sent in liste_text]
model = Word2Vec(word_tokens, min_count=1)
model_filename=folder_out+"w2v_ours_"+type_text.replace("text_","")+".pkl"
with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
model.save(folder_out+"vocab_all_"+type_text.replace("text_","")+".bin")
dictionary=list(model.wv.vocab.keys())
dictionary=[num for num in dictionary if num.isdigit()==False]
np.savetxt(folder_out+"dictionary_"+type_text.replace("text_","")+".csv", dictionary,
delimiter=';', fmt='%s', encoding='UTF-8')

