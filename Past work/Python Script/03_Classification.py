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
paragraphs= pd.read_csv(folder_out+"paragraphs_for_language.csv",delimiter=";")
phenotype_third= pd.read_csv(folder_out+"phenotype_third.csv",delimiter=";")
actors= pd.read_csv(folder_out+"actors.csv",delimiter=";")
dates= pd.read_csv(folder_out+"dates.csv",delimiter=";")
locations= pd.read_csv(folder_out+"locations.csv",delimiter=";")
references= pd.read_csv(folder_out+"references.csv",delimiter=";")
themes= pd.read_csv(folder_out+"themes.csv",delimiter=";")
tags= pd.read_csv(folder_out+"tags.csv",delimiter=";")


# List needed
actors_list= pd.read_csv(folder_list+"actors_list_final.csv",delimiter=";")
qualitative_file=pd.read_csv(folder_list+"qualitative_recommendation_KP_2021_028_manually.csv",delimiter=";")

# 8 Extract Recommandations elements
load_elements=0
if load_elements==0:
    recommendations_elements =extract_recommendations_elements(paragraphs[['para_id','text_lower', \
        "full_structure","title","text_tag","file_name"]],phenotype_third,actors_list)
    recommendations_elements.to_csv(folder_out+"recommendations_elements.csv",sep=";",index=False)
else:
    recommendations_elements= pd.read_csv(folder_out+"recommendations_elements.csv",delimiter=";")

# 9 Create Model
load_model=0
if load_model==0:
    recommendations_model,col=calculate_recommendations(recommendations_elements,qualitative_file,folder_out)
else: 
    recommendations_model = folder_out+"linear_model.pkl"
    col=["title","modal","must_has_to","may_might", \
     "can_could","will","would", "ought_to", "should_shall", "expression","stronger_words", \
         "weaker_words","last_sentence","title_word"]

# 10 Apply model
load_recommendation=0
if load_recommendation==0:
    recommendations=apply_model_recommendation(recommendations_elements,col,recommendations_model)
    recommendations.to_csv(folder_out+"recommendations.csv",sep=";",index=False)
else:
    recommendations= pd.read_csv(folder_out+"recommendations.csv",delimiter=";")
    print("load recommendations")


# 11 Merge all
merge_all=0
if merge_all==0:
    paragraphs_all=paragraphs[['file_name', 'page', 'block_num',  
      'full_structure', 'text_para', 'para_id', 'title', 'text_lemma_tag',"text_tag"]]
    paragraphs_all=paragraphs_all.merge(actors[["para_id","actors"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(dates[["para_id","all_full_dates"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(locations[["para_id","Admin1_Code","Admin1_Name"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(references[["para_id","ref"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(themes[["para_id","themes"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(tags[["para_id","tags"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(recommendations[["para_id","recommendation","score"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(recommendations_elements[["para_id","explicit"]],on="para_id")
    paragraphs_all.reset_index()
    paragraphs_all.index.name="index"
    paragraphs_all.to_csv(folder_out+"paragraphs_all.csv",sep=";")
    # Lighter files for pbi
    col_pbi=["file_name","page","text_para","para_id","actors","all_full_dates","Admin1_Code","Admin1_Name","ref","themes",	"tags","recommendation","explicit"]
    paragraphs_pbi=paragraphs_all[col_pbi]
    paragraphs_pbi.to_csv(folder_out+"paragraphs_pbi.csv",sep=";")
else:
    paragraphs_all=pd.read_csv(folder_out+"paragraphs_all.csv",delimiter=";")
    paragraphs_all.set_index("index")
