import os
import csv
import glob
import pandas as pd
import numpy as np
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import io
from gensim.models import Word2Vec
from Functions_add import *
#from temp_test import *
from pdfminer.high_level import extract_text
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()

# This need to be adapted
# 1) Take list of files not in paragraph_all
# 2) Run Separation
# 3) Run Extract
# 4) ML?


cwd = os.getcwd()

#pdf_files=glob.glob("../../../Data/Reports/Review_reports/*.pdf")
pdf_files=glob.glob("../../../Data/Reports/Review_reports/Reports_to_add/*.pdf")


# 0 : do, 1 : load from file
# 1 Extract Metadata
load_metadata=1
# 2 Extract images/pages
load_images_pages=1
# 3 Extract paragraphes
b=0
load_paragraphes=b
load_first_phenotype=b
load_restructure=b
load_second_phenotype=b
load_redivise=b
load_arrange_structure=b
# 6 Extract actors
load_actors=0
# 10 Prepare Language

prepare_language=0
load_third_phenotype=0
# 4 Extract dates
load_dates=1
# 5 Extract locations
load_locations=1

# 7 Extract references
load_references=1
# 8 Extract themes
load_themes=1


# 9 Extract Recommandations

load_recommendation=1


# 1 Extract Metadata
if load_metadata == 0:
    metadata_prior= pd.read_csv('../../../Processing/Out/Explore/metadata.csv',delimiter=";")
    metadata=add_metadata(pdf_files,metadata_prior)
    metadata.to_csv('../../../Processing/Out/Systematize/metadata.csv',sep=';',index=False, encoding='UTF-8') 
else:
    metadata= pd.read_csv('../../../Processing/Out/Systematize/metadata.csv',delimiter=";")
    print("loaded metadata")

# 2 Extract images/pages
if load_images_pages == 0:
    images_pages_tables_prior= pd.read_csv('../../../Processing/Out/Explore/images_pages.csv',delimiter=";")
    images_pages_tables=add_images_pages(pdf_files,images_pages_tables_prior)
    images_pages_tables.to_csv('../../../Processing/Out/Systematize/images_pages.csv',sep=';',index=False, encoding='UTF-8') 
else:
    images_pages_tables= pd.read_csv('../../../Processing/Out/Systematize/images_pages.csv',delimiter=";")
    print("loaded images_pages")

# 3 Extract paragraphes
if load_paragraphes == 0:
    paragraphs_prior= pd.read_csv('../../../Processing/Out/Explore/paragraphs.csv',delimiter=";")
    paragraphs= add_structure(pdf_files,images_pages_tables,paragraphs_prior)
    paragraphs.to_csv('../../../Processing/Out/Systematize/paragraphs.csv',sep=';',index=False, encoding='UTF-8') 
else:
    paragraphs= pd.read_csv('../../../Processing/Out/Systematize/paragraphs.csv',delimiter=";")
    print("load structure")


# Eventually reorganize for bullet point, merging by section and we would need the useless lines
# Define phenotype
if load_first_phenotype == 0:
    phenotype_prior= pd.read_csv("../../../Processing/Out/Explore/phenotype_first.csv",delimiter=";")
    phenotype= add_phenotype(paragraphs,5,phenotype_prior)
    phenotype.to_csv("../../../Processing/Out/Systematize/phenotype_first.csv",sep=";",index=False,encoding="UTF-8")
else:
    phenotype= pd.read_csv("../../../Processing/Out/Systematize/phenotype_first.csv",delimiter=";")
    print("load phenotype")



# Define phenotype
if load_restructure == 0:
    paragraphs_restructured_prior= pd.read_csv("../../../Processing/Out/Explore/paragraphs_restructured.csv",delimiter=";",encoding="UTF-8")
    paragraphs_restructured= add_restructure(paragraphs,phenotype,paragraphs_restructured_prior)
    paragraphs_restructured.to_csv("../../../Processing/Out/Systematize/paragraphs_restructured.csv",sep=";",index=False,encoding="UTF-8")
else:
    paragraphs_restructured= pd.read_csv("../../../Processing/Out/Systematize/paragraphs_restructured.csv",delimiter=";",encoding="UTF-8")
    print("load paragraphs restructured")  


if load_second_phenotype == 0:
    phenotype_second_prior= pd.read_csv("../../../Processing/Out/Explore/phenotype_second.csv",delimiter=";")
    phenotype_second= add_phenotype(paragraphs_restructured,-1,phenotype_second_prior)
    phenotype_second.to_csv("../../../Processing/Out/Systematize/phenotype_second.csv",sep=";",index=False,encoding="UTF-8")
else:
    phenotype_second= pd.read_csv("../../../Processing/Out/Systematize/phenotype_second.csv",delimiter=";")
    print("load phenotype second")

# Define phenotype
if load_redivise == 0:
    paragraphs_fused_smart_prior=pd.read_csv("../../../Processing/Out/Explore/paragraphs_fused_smart.csv",delimiter=";")
    paragraphs_fused_smart= add_fused_smart(paragraphs_restructured,phenotype_second,paragraphs_fused_smart_prior)
    paragraphs_fused_smart.to_csv("../../../Processing/Out/Systematize/paragraphs_fused_smart.csv",sep=";",index=False,encoding="UTF-8")
    paragraphs_redivised_prior= pd.read_csv("../../../Processing/Out/Explore/paragraphs_redivised.csv",delimiter=";")
    paragraphs_redivised= add_redivise(paragraphs_fused_smart,phenotype_second,paragraphs_redivised_prior)
    # Adhoc - Clean Santos Cruz
    paragraphs_redivised["text_para"]=paragraphs_redivised.text_para[paragraphs_redivised.text_para!="SHORT TERM:"]
    paragraphs_redivised["text_para"]=paragraphs_redivised.text_para[paragraphs_redivised.text_para!="MID- LONG TERM"]
    paragraphs_redivised.to_csv("../../../Processing/Out/Systematize/paragraphs_redivised.csv",sep=";",index=False,encoding="UTF-8")
else:
    paragraphs_redivised= pd.read_csv("../../../Processing/Out/Systematize/paragraphs_redivised.csv",delimiter=";")
    print("load paragraphs redivised")  


# Define phenotype
if load_arrange_structure == 0:
    paragraphs_restructured2_prior= pd.read_csv("../../../Processing/Out/Explore/paragraphs_restructured2.csv",delimiter=";")
    paragraphs_restructured2=add_restructure2(paragraphs_redivised,phenotype_second,paragraphs_restructured2_prior)
    paragraphs_restructured2.to_csv("../../../Processing/Out/Systematize/paragraphs_restructured2.csv",sep=";",index=False,encoding="UTF-8")
    paragraphs_rearranged_prior= pd.read_csv("../../../Processing/Out/Explore/paragraphs_rearranged.csv",delimiter=";",encoding="UTF-8")
    paragraphs_rearranged= add_arrange_structure(paragraphs_restructured2,phenotype_second,paragraphs_rearranged_prior)
    paragraphs_rearranged.to_csv("../../../Processing/Out/Systematize/paragraphs_rearranged.csv",sep=";",index=False)
else:
    paragraphs_rearranged= pd.read_csv("../../../Processing/Out/Systematize/paragraphs_rearranged_police_added.csv",delimiter=";",encoding="UTF-8")
    print("load arrange structure")  


# 4 Remove useless for now  
# Remove numbers and short lines
paragraphs=paragraphs_rearranged
#paragraphs=paragraphs.loc[paragraphs.text_para.str.len()>3]



# 6 Extract actors
actors_list= pd.read_csv("../../../Processing/Lists/actors_list_sage_cat_210208.csv",delimiter=";")
if load_actors == 0:
    actors,paragraphs,tags_actors=add_extract_actors(paragraphs,actors_list)
    #actors,paragraphs=addactors(paragraphs)
    tags_actors.to_csv("../../../Processing/Out/Systematize/tags_actors.csv",sep=";",index=False)
    actors.to_csv("../../../Processing/Out/Systematize/actors.csv",sep=";",index=False)
    paragraphs.to_csv("../../../Processing/Out/Systematize/paragraphs_with_tag.csv",sep=";",index=False)
   
else:
    actors= pd.read_csv('../../../Processing/Out/Systematize/actors.csv',delimiter=";")
    paragraphs= pd.read_csv('../../../Processing/Out/Systematize/paragraphs_with_tag.csv',delimiter=";")
    print("loaded actors and tagged paragraphs")


# 10 Prepare Language
if prepare_language==0:
    paragraphs_for_language =add_prepare_language_tags(paragraphs)
    paragraphs_for_language.to_csv("../../../Processing/Out/Systematize/paragraphs_for_language.csv",sep=";",index=False)

else:
    paragraphs_for_language= pd.read_csv('../../../Processing/Out/Systematize/paragraphs_for_language.csv',delimiter=";")
    print("load paragraphs_for_language")

if load_third_phenotype == 0:
    phenotype_third_prior= pd.read_csv("../../../Processing/Out/Explore/phenotype_second.csv",delimiter=";")
    phenotype_third= add_phenotype(paragraphs_for_language,5,phenotype_third_prior)
    phenotype_third.to_csv("../../../Processing/Out/Systematize/phenotype_third.csv",sep=";",index=False,encoding="UTF-8")
else:
    phenotype_third= pd.read_csv("../../../Processing/Out/Systematize/phenotype_third.csv",delimiter=";")
    print("load phenotype third")

paragraphs=paragraphs_for_language

# 4 Extract dates
if load_dates == 0:
    dates=add_extract_dates(paragraphs[['para_id','text_lower']])
    dates.to_csv("../../../Processing/Out/Systematize/dates.csv",sep=";",index=False)
else:
    dates= pd.read_csv('../../../Processing/Out/Systematize/dates.csv',delimiter=";")
    print("loaded dates")

# 5 Extract locations
if load_locations == 0:
    locations=add_extract_locations(paragraphs[['para_id','text_lower']])
    locations.to_csv("../../../Processing/Out/Systematize/locations.csv",sep=";",index=False)
else:
    locations= pd.read_csv('../../../Processing/Out/Systematize/locations.csv',delimiter=";")
    print("loaded locations")

# 7 Extract references
if load_references == 0:
    references=add_extract_references(paragraphs[['para_id','text_lower']])
    references.to_csv("../../../Processing/Out/Systematize/references.csv",sep=";",index=False)
else:
    references= pd.read_csv('../../../Processing/Out/Systematize/references.csv',delimiter=";")
    print("loaded references")


# 8 Extract themes and tags
if load_themes == 0:
    make_dico=0
    if make_dico==1:
        word_tokens = [word_tokenize(doc.lower())  for doc in paragraphs["text_lemma_tag"] if isinstance(doc,str)]
        #### Vocab word tokens ... vocab2200?
        model = Word2Vec(word_tokens, min_count=1)
        model.save("../../../Processing/Out/Language/vocab_all_lemma_tag_200122.bin")
        dictionary=list(model.wv.vocab.keys())
        dictionary=[num.strip() for num in dictionary if num.isdigit()==False]
        #dictionary=[num for num in dictionary if num.isdigit()==False]
        pd.DataFrame(dictionary).to_csv("../../../Processing/Out/Language/dictionary_200122.csv",sep=";",index=False)
    else:
        dictionary=pd.read_csv("../../../Processing/Out/Language/dictionary_200122.csv",delimiter=";")

    thematic_list= pd.read_csv('../../../Processing/Lists/theme_list.csv',delimiter=";")
    themes=add_extract_themes(paragraphs[['para_id','text_lemma_tag']],thematic_list)
    themes.to_csv("../../../Processing/Out/Systematize/themes.csv",sep=";",index=False)

    tag_list= pd.read_csv('../../../Processing/Lists/tags_1712.csv',delimiter=";")
    tags=add_extract_tags(paragraphs[['para_id','text_lower']],tag_list)
    tags.columns.values[2]="tags"
    tags.to_csv("../../../Processing/Out/Systematize/tags.csv",sep=";",index=False)
else:
    themes= pd.read_csv('../../../Processing/Out/Systematize/themes.csv',delimiter=";")
    tags= pd.read_csv('../../../Processing/Out/Systematize/tags.csv',delimiter=";")
    tags.columns.values[2]="tags"
    #tags.column

    print("loaded themes and tags")





load_recommendation=0
# 9 Extract Recommandations
if load_recommendation==0:
    load_elements=0
    if load_elements==0:
        recommendations_elements_prior= pd.read_csv('../../../Processing/Out/Explore/recommendations_elements.csv',delimiter=";")
        recommendations_elements =add_extract_recommendations_elements(paragraphs_for_language[['para_id','text_lower', \
            "full_structure","title","text_lemma_tag","file_name"]],phenotype_third,recommendations_elements_prior,actors_list)
        recommendations_elements.to_csv("../../../Processing/Out/Systematize/recommendations_elements.csv",sep=";",index=False)
    else:
        recommendations_elements= pd.read_csv('../../../Processing/Out/Systematize/recommendations_elements.csv',delimiter=";")
    
    #qualitative_file=pd.read_csv("../../../Processing/Out/Language/Recommendations/Review_for_kristen/equiv_KP_endnovember.csv",delimiter=",")
    qualitative_file=pd.read_csv("../../../Processing/Out/Language/Recommendations/Review_for_kristen/equiv_KP_2021_01_ended.csv",delimiter=";")
    
    recommendations_model,col=add_calculate_recommendations(recommendations_elements,qualitative_file)
    col=["modal0","must_has_to","may_might", \
    "can_could","will","would", "ought_to", "should_shall", "expression","stronger_words",  \
        "weaker_words","last_sentence","title","explicit","toverb"]
    recommendations_model = "../../../Processing/Out/Language/Recommendations/add_linear_model.pkl"

    recommendations=add_apply_model_recommendation(recommendations_elements,col,recommendations_model)
    recommendations.to_csv("../../../Processing/Out/Systematize/recommendations.csv",sep=";",index=False)
else:
    #recommendations_elements= pd.read_csv('../../../Processing/Out/Explore/recommendations_elements.csv',delimiter=";")
    recommendations= pd.read_csv('../../../Processing/Out/Systematize/recommendations.csv',delimiter=";")
    recommendations_elements= pd.read_csv('../../../Processing/Out/Systematize/recommendations_elements.csv',delimiter=";")
    print("load recommendations")





# 11 Merge all
merge_all=0
if merge_all==0:
    paragraphs_all=paragraphs[['file_name', 'page', 'block_num',  
      'full_structure', 'text_para', 'para_id', 'title', 'text_lemma_tag']]
    paragraphs_all=paragraphs_all.merge(actors[["para_id","actors"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(dates[["para_id","all_full_dates"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(locations[["para_id","Admin1_Code","Admin1_Name"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(references[["para_id","ref"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(themes[["para_id",'first_theme', 'first_value',
       'second_theme', 'second_value', 'third_theme', 'third_value']],on="para_id")
    paragraphs_all=paragraphs_all.merge(tags[["para_id","tags"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(recommendations[["para_id","recommendation","score"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(recommendations_elements[["para_id","explicit"]],on="para_id")
    paragraphs_all.to_csv("../../../Processing/Out/Systematize/paragraphs_all.csv",sep=";",index=False)
else:
    paragraphs_all=pd.read_csv('../../../Processing/Out/Systematize/paragraphs_all.csv',delimiter=";")


cosim, keep_id,flat=lemma_cosine_similarity(paragraphs_all)
cosim.to_csv("../../../Processing/Out/Explore/Cosim.csv",sep=";",index=False)

count=0
flat_cosim=pd.DataFrame(columns=["cosim","idx_para1","idx_para2"])
#flat_cosim["cosim"]=flat
for i in range(cosim.shape[0]-1):
    for j in range(i+1,cosim.shape[0]):
        if cosim[i][j]!=0 and cosim[i][j]!=1:
            # flat_cosim[count,"cosim"]=cosim[i][j]
            # flat_cosim[count,"idx_para1"]=i
            # flat_cosim[count,"idx_para2"]=j
            # count+=1
            a=pd.Series([cosim[i][j],i,j],index=flat_cosim.columns)
            flat_cosim=flat_cosim.append(a,ignore_index = True) 


flat_cosim.to_csv("../../../Processing/Out/Explore/flat_cosim.csv",sep=";",index=False)            
keep_id.to_csv("../../../Processing/Out/Explore/keep_id.csv",sep=";")
