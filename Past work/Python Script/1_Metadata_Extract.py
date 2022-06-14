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
import math
from gensim.models import Word2Vec
from Functions import *
#from temp_test import *
from pdfminer.high_level import extract_text
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()


cwd = os.getcwd()

#pdf_files=glob.glob("../../../Data/Reports/Review_reports/*.pdf")
pdf_files=list(glob.glob("../../../Data/Reports/Review_reports/**/*.pdf", recursive=True))
#pdf_files=pdf_files[0]
pdf_remove=list(glob.glob("../../../Data/Reports/Review_reports/Reports_to_add/*.pdf"))
pdf_files=[item for item in pdf_files if item.find("Reports_to_add") ==-1]
all_text=""

# 0 : do, 1 : load from file
# 1 Extract Metadata
load_metadata=1
# 2 Extract images/pages
load_images_pages=1
# 3 Extract paragraphes
b=1
load_paragraphes=b
load_first_phenotype=b
load_restructure=b
load_second_phenotype=b
load_redivise=b
load_arrange_structure=b

c=1
load_cleaner = 1
# 6 Extract actors
load_actors=c
# 4 Extract dates
load_dates=c
# 5 Extract locations
load_locations=c

# 7 Extract references
load_references=c

# 10 Prepare Language
prepare_language=c

# 8 Extract thematics
load_thematics=c


# 9 Extract Recommandations
load_third_phenotype=c
load_recommendation=c


# 1 Extract Metadata
if load_metadata == 0:
    metadata=extract_metadata(pdf_files)
    metadata.to_csv('../../../Processing/Out/Explore/metadata.csv',sep=';',index=False, encoding='UTF-8') 
else:
    metadata= pd.read_csv('../../../Processing/Out/Explore/metadata.csv',delimiter=";")
    print("loaded metadata")

# 2 Extract images/pages
if load_images_pages == 0:
    images_pages_tables=extract_images_pages(pdf_files)
    images_pages_tables.to_csv('../../../Processing/Out/Explore/images_pages.csv',sep=';',index=False, encoding='UTF-8') 
else:
    images_pages_tables= pd.read_csv('../../../Processing/Out/Explore/images_pages.csv',delimiter=";")
    print("loaded images_pages")

# 3 Extract paragraphes
if load_paragraphes == 0:
    paragraphs= extract_structure(pdf_files,images_pages_tables)
    
    # Adhoc - Clean italic based on actors names, can be automatized from actors list
    # idx_a=paragraphs[paragraphs.text_para.notnull()]
    # idx_a= idx_a.loc[((idx_a.text_para.str.find("Union pour la paix")!=-1) | (idx_a.text_para.str.find("Sangaris")!=-1) | \
    #     (idx_a.text_para.str.find("Front populaire pour la renaissance de la Centrafrique")!=-1)) & \
    #     (idx_a.font.str.find("Italic")!=-1)].index
    # paragraphs.tag[idx_a]="<p>"
    # paragraphs.text_para[idx_a]=[item[:-1] if item[-1]=="|" else item for item in paragraphs.text_para[idx_a]]
    paragraphs.to_csv('../../../Processing/Out/Explore/paragraphs.csv',sep=';',index=False, encoding='UTF-8') 
else:
    paragraphs= pd.read_csv('../../../Processing/Out/Explore/paragraphs.csv',delimiter=";")
    print("load structure")


# Eventually reorganize for bullet point, merging by section and we would need the useless lines
# Define phenotype
if load_first_phenotype == 0:
    phenotype= extract_phenotype(paragraphs,5)
    phenotype.to_csv("../../../Processing/Out/Explore/phenotype_first.csv",sep=";",index=False,encoding="UTF-8")
else:
    phenotype= pd.read_csv("../../../Processing/Out/Explore/phenotype_first.csv",delimiter=";")
    print("load phenotype")



# Define phenotype
if load_restructure == 0:
    paragraphs_restructured= restructure(paragraphs,phenotype)
    paragraphs_restructured.to_csv("../../../Processing/Out/Explore/paragraphs_restructured.csv",sep=";",index=False,encoding="UTF-8")
else:
    paragraphs_restructured= pd.read_csv("../../../Processing/Out/Explore/paragraphs_restructured.csv",delimiter=";",encoding="UTF-8")
    print("load paragraphs restructured")  


if load_second_phenotype == 0:
    phenotype_second= extract_phenotype(paragraphs_restructured,-1)
    phenotype_second.to_csv("../../../Processing/Out/Explore/phenotype_second.csv",sep=";",index=False,encoding="UTF-8")
else:
    phenotype_second= pd.read_csv("../../../Processing/Out/Explore/phenotype_second.csv",delimiter=";")
    print("load phenotype second")

# Define phenotype
if load_redivise == 0:
    paragraphs_fused_smart= fused_smart(paragraphs_restructured,phenotype_second)
    paragraphs_fused_smart.to_csv("../../../Processing/Out/Explore/paragraphs_fused_smart.csv",sep=";",index=False,encoding="UTF-8")
    paragraphs_redivised= redivise(paragraphs_fused_smart,phenotype_second)
    # Adhoc - Clean Santos Cruz
    paragraphs_redivised["text_para"]=paragraphs_redivised.text_para[paragraphs_redivised.text_para!="SHORT TERM:"]
    paragraphs_redivised["text_para"]=paragraphs_redivised.text_para[paragraphs_redivised.text_para!="MID- LONG TERM"]
    paragraphs_redivised.to_csv("../../../Processing/Out/Explore/paragraphs_redivised.csv",sep=";",index=False,encoding="UTF-8")
else:
    paragraphs_redivised= pd.read_csv("../../../Processing/Out/Explore/paragraphs_redivised.csv",delimiter=";")
    print("load paragraphs redivised")  


# Define phenotype
if load_arrange_structure == 0:
    paragraphs_restructured2=restructure2(paragraphs_redivised,phenotype_second)
    paragraphs_restructured2.to_csv("../../../Processing/Out/Explore/paragraphs_restructured2.csv",sep=";",index=False,encoding="UTF-8")
    paragraphs_rearranged= arrange_structure(paragraphs_restructured2,phenotype_second)
    paragraphs_rearranged.to_csv("../../../Processing/Out/Explore/paragraphs_rearranged.csv",sep=";",index=False)
else:
    paragraphs_rearranged= pd.read_csv("../../../Processing/Out/Explore/paragraphs_rearranged.csv",delimiter=";",encoding="UTF-8")
    paragraphs_rearranged= pd.read_csv("../../../Processing/Out/Explore/paragraphs_rearranged_for_pbi.csv",delimiter=";",encoding="UTF-8")
    print("load arrange structure")  


# 4 Remove punctuation  

if load_cleaner == 0:
    paragraphs=clean_text(paragraphs_rearranged)
    paragraphs.to_csv("../../../Processing/Out/Explore/paragraphs_cleaner.csv",sep=";",index=False)
else:
    paragraphs= pd.read_csv("../../../Processing/Out/Explore/paragraphs_cleaner.csv",delimiter=";",encoding="UTF-8")
#paragraphs=paragraphs.loc[paragraphs.text_para.str.len()>3]

# 6 Extract actors
actors_list= pd.read_csv("../../../Processing/Lists/actors_list_final.csv",delimiter=";")
if load_actors == 0:
    actors,paragraphs,tags_actors=extract_actors(paragraphs,actors_list)
    #actors,paragraphs=extract_actors(paragraphs)
    tags_actors.to_csv("../../../Processing/Out/Explore/tags_actors.csv",sep=";",index=False)
    actors.to_csv("../../../Processing/Out/Explore/actors.csv",sep=";",index=False)
    paragraphs.to_csv("../../../Processing/Out/Explore/paragraphs_with_tag.csv",sep=";",index=False)
    actors_pbi=actors[["para_id","actors"]]
    actors_pbi.to_csv("../../../Processing/Out/Explore/actors_pbi.csv",sep=";",index=False)
else:
    actors= pd.read_csv('../../../Processing/Out/Explore/actors.csv',delimiter=";")
    paragraphs= pd.read_csv('../../../Processing/Out/Explore/paragraphs_with_tag.csv',delimiter=";")
    print("loaded actors and tagged paragraphs")


# 4 Extract dates
if load_dates == 0:
    dates=extract_dates(paragraphs[['para_id','text_lower']])
    dates.to_csv("../../../Processing/Out/Explore/dates.csv",sep=";",index=False)
    dates_pbi=dates[["para_id","all_full_dates"]]
    dates_pbi.to_csv("../../../Processing/Out/Explore/dates_pbi.csv",sep=";",index=False)
else:
    dates= pd.read_csv('../../../Processing/Out/Explore/dates.csv',delimiter=";")
    print("loaded dates")

# 5 Extract locations
if load_locations == 0:
    admins = pd.read_csv('../../../Data/External_Data/GIS/caf_admbnda_adm3_200k_sigcaf_reach_itos_v2.csv',delimiter=";")
    locations=extract_locations(paragraphs[['para_id','text_tag']],admins)
    locations.to_csv("../../../Processing/Out/Explore/locations.csv",sep=";",index=False)
    locations_pbi=locations[["para_id","Admin1_Code","Admin1_Name"]]
    locations_pbi.to_csv("../../../Processing/Out/Explore/locations_pbi.csv",sep=";",index=False)
else:
    locations= pd.read_csv('../../../Processing/Out/Explore/locations.csv',delimiter=";")
    print("loaded locations")

# 7 Extract references
if load_references == 0:
    references=extract_references(paragraphs[['para_id','text_lower']])
    references.to_csv("../../../Processing/Out/Explore/references.csv",sep=";",index=False)
    references_pbi=references[["para_id","ref"]]
    references_pbi.to_csv("../../../Processing/Out/Explore/references_pbi.csv",sep=";",index=False)
else:
    references= pd.read_csv('../../../Processing/Out/Explore/references.csv',delimiter=";")
    print("loaded references")


# 10 Prepare Language
if prepare_language==0:
    paragraphs_for_language =prepare_language_tags(paragraphs)
    paragraphs_for_language.to_csv("../../../Processing/Out/Explore/paragraphs_for_language.csv",sep=";",index=False)

else:
    paragraphs_for_language= pd.read_csv('../../../Processing/Out/Explore/paragraphs_for_language.csv',delimiter=";")
    print("load paragraphs_for_language")


if load_third_phenotype == 0:
    phenotype_third= extract_phenotype(paragraphs_for_language,5)
    phenotype_third.to_csv("../../../Processing/Out/Explore/phenotype_third.csv",sep=";",index=False,encoding="UTF-8")
else:
    phenotype_third= pd.read_csv("../../../Processing/Out/Explore/phenotype_third.csv",delimiter=";")
    print("load phenotype third")

paragraphs=paragraphs_for_language.merge(actors[["para_id","actors"]],on="para_id")


# 8 Extract thematics and tags
if load_thematics == 0:
    theme_list= pd.read_csv('../../../Processing/Lists/theme_list.csv',delimiter=";")
    themes,themes_flat=extract_themes(paragraphs[['para_id','text_lemma',"actors"]],theme_list,actors_list)
    themes.to_csv("../../../Processing/Out/Explore/themes.csv",sep=";",index=False,encoding="UTF-8")
    themes_flat.to_csv("../../../Processing/Out/Explore/themes_flat.csv",sep=";",index=False)
    tag_list= pd.read_csv('../../../Processing/Lists/tags_list_final.csv',delimiter=";")
    tags=extract_tags(paragraphs[['para_id',"text_lemma","actors"]],tag_list, actors_list)
    tags.to_csv("../../../Processing/Out/Explore/tags.csv",sep=";",index=False,encoding="UTF-8")
else:
    themes= pd.read_csv('../../../Processing/Out/Explore/themes.csv',delimiter=";")
    tags= pd.read_csv('../../../Processing/Out/Explore/tags.csv',delimiter=";")
    tag_list= pd.read_csv('../../../Processing/Lists/tags_list_final.csv',delimiter=";")
    print("loaded themes and tags")


# 9 Extract Recommandations
if load_recommendation==0:
    load_elements=1
    if load_elements==0:
        recommendations_elements =extract_recommendations_elements(paragraphs[['para_id','text_lower', \
            "full_structure","title","text_tag","file_name"]],phenotype_third,actors_list)
        recommendations_elements.to_csv("../../../Processing/Out/Explore/recommendations_elements.csv",sep=";",index=False)
    else:
        recommendations_elements= pd.read_csv('../../../Processing/Out/Explore/recommendations_elements.csv',delimiter=";")
    
    qualitative_file=pd.read_csv("../../../Processing/Out/Language/Recommendations/qualitative_recommendation_KP_2021_028_manually.csv",delimiter=";")
    #qualitative_file=pd.read_csv("../../../Processing/Out/Language/Recommendations/Review_for_kristen/equiv_KP_2021_01_ended.csv",delimiter=";")
    
    recommendations_model,col=calculate_recommendations(recommendations_elements,qualitative_file)
    # col=["title","modal","must_has_to","may_might", \
    #     "can_could","will","would", "ought_to", "should_shall", "expression","stronger_words", \
    #         "weaker_words","last_sentence","title_word"]
    recommendations_model = "../../../Processing/Out/Language/Recommendations/linear_model.pkl"

    recommendations=apply_model_recommendation(recommendations_elements,col,recommendations_model)
    recommendations.to_csv("../../../Processing/Out/Explore/recommendations.csv",sep=";",index=False)
else:
    #recommendations_elements= pd.read_csv('../../../Processing/Out/Explore/recommendations_elements.csv',delimiter=";")
    recommendations= pd.read_csv('../../../Processing/Out/Explore/recommendations.csv',delimiter=";")
    recommendations_elements= pd.read_csv('../../../Processing/Out/Explore/recommendations_elements.csv',delimiter=";")
    print("load recommendations")





# 11 Merge all
merge_all=1
if merge_all==0:
    paragraphs_all=paragraphs_for_language[['file_name', 'page', 'block_num',  
      'full_structure', 'text_para', 'para_id', 'title', 'text_lemma_tag',"text_tag"]]
    paragraphs_all=paragraphs_all.merge(actors[["para_id","actors"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(dates[["para_id","all_full_dates"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(locations[["para_id","Admin1_Code","Admin1_Name"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(references[["para_id","ref"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(themes[["para_id","themes"]],on="para_id")
    # paragraphs_all=paragraphs_all.merge(themes[["para_id",'first_theme', 'first_value',
    #    'second_theme', 'second_value', 'third_theme', 'third_value']],on="para_id")
    # For KP Review
    #paragraphs_all=paragraphs_all.merge(themes[["para_id",'first_theme', 'first_value',
    #   'second_theme', 'second_value', 'third_theme', 'third_value',"Protection of civilians","Rule of Law and Justice","Performance and accountability"]],on="para_id")
    
    paragraphs_all=paragraphs_all.merge(tags[["para_id","tags"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(recommendations[["para_id","recommendation","score"]],on="para_id")
    paragraphs_all=paragraphs_all.merge(recommendations_elements[["para_id","explicit"]],on="para_id")
    #paragraphs_all.to_csv("../../../Processing/Out/Explore/paragraphs_all.csv",sep=";")
    paragraphs_all.reset_index()
    paragraphs_all.index.name="index"
    paragraphs_all.to_csv("../../../Processing/Out/Explore/paragraphs_all.csv",sep=";")
    col_pbi=["file_name","page","text_para","para_id","actors","all_full_dates","Admin1_Code","Admin1_Name","ref","themes",	"tags","recommendation","explicit"]
    paragraphs_pbi=paragraphs_all[col_pbi]
    paragraphs_pbi.to_csv("../../../Processing/Out/Explore/paragraphs_pbi.csv",sep=";")
    
else:
    paragraphs_all=pd.read_csv('../../../Processing/Out/Explore/paragraphs_all.csv',delimiter=";")
    paragraphs_all.set_index("index")

load_pairs=1
if load_pairs==0:
    paragraphs_simil=paragraphs_all.merge(themes,on="para_id")
    recommandation_pairs=link_recommandations(paragraphs_simil,list(tag_list["Level 2"]))
    recommandation_pairs.columns=["idx1","idx2"]
    recommandation_pairs.to_csv("../../../Processing/Out/Explore/recommandation_pairs.csv",sep=";",index=False)
else:
    recommandation_pairs=pd.read_csv("../../../Processing/Out/Explore/recommandation_pairs.csv",delimiter=";")


load_wmd=1
if load_wmd==0:
    model_filename="../../../Processing/Out/Language/Similarities/WMD/w2v_ours.pkl"
    flat_wmd=lemma_wmd(paragraphs_all,recommandation_pairs,model_filename)
    flat_wmd.to_csv("../../../Processing/Out/Explore/flat_wmd.csv",sep=";",index=False)

flat_wmd=pd.read_csv("../../../Processing/Out/Explore/flat_wmd.csv",delimiter=";")
flat_wmd=flat_wmd[flat_wmd["wmd"]<8]
flat_wmd["Similarity"]=0
for i in flat_wmd.index:
    flat_wmd.loc[i,"Similarity"]=math.floor(1000*math.exp(-flat_wmd.loc[i,"wmd"]/3))/1000
    
flat_wmd=flat_wmd[["idx1","idx2","Similarity"]]
flat_wmd.to_csv("../../../Processing/Out/Explore/flat_wmd_percent.csv",sep=";",index=False)

flat_wmd=pd.read_csv("../../../Processing/Out/Explore/flat_wmd_percent.csv",delimiter=";")

#flat_wmd["Similarity"]=

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
