import os
import csv
import glob
import pandas as pd
import numpy as np

from Functions import *



# Folder where the files will be saved
folder_out="../../../Processing/Out/Explore/"

# Table with all extraction information from reports
# columns : ['file_name', 'page', 'block_id', 'block_num', 'text', 'font_size',
#       'font', 'tag', 'verb', 'structure_title', 'full_structure',
#       'section_id', 'title', 'text_para', 'text_lower', 'para_id']
paragraphs= pd.read_csv(folder_out+"paragraphs_rearranged.csv",delimiter=";",encoding="UTF-8")
# 4 Remove punctuation  

if load_cleaner == 0:
    paragraphs=clean_text(paragraphs)
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
