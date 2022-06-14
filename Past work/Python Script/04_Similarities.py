import os
import csv
import glob
import pandas as pd
import numpy as np

from Functions import *

# Folder where the files will be saved
folder_out="Report_for_Analysis/Explore/"
folder_list="Report_for_Analysis/Lists/"

# Previous files
paragraphs= pd.read_csv(folder_out+"paragraphs_all.csv",delimiter=";")
themes= pd.read_csv(folder_out+"themes.csv",delimiter=";")

# List needed
tag_list= pd.read_csv(folder_list+"tags_list_final.csv",delimiter=";")

# 12 Create pairs of recommendation
load_pairs=0
if load_pairs==0:
    paragraphs_simil=paragraphs.merge(themes,on="para_id")
    recommandation_pairs=link_recommandations(paragraphs_simil,list(tag_list["Level 2"]))
    recommandation_pairs.columns=["idx1","idx2"]
    recommandation_pairs.to_csv(folder_out+"recommandation_pairs.csv",sep=";",index=False)
else:
    recommandation_pairs=pd.read_csv(folder_out+"recommandation_pairs.csv",delimiter=";")

# 13 Calculate similarities
load_wmd=0
if load_wmd==0:
    # See 05_Vocabulary
    model_filename=folder_list+"w2v_ours.pkl"
    flat_wmd=lemma_wmd(paragraphs,recommandation_pairs,model_filename)
    flat_wmd.to_csv(folder_out+"flat_wmd_percent.csv",sep=";",index=False)
