import os
import csv
import glob
import pandas as pd
import numpy as np

from Functions import *



# Folder where the files will be saved
folder_out="Report_for_Analysis/Explore/"
folder_list="Report_for_Analysis/Lists/"

# Table with all extraction information from reports
# columns : ['file_name', 'page', 'block_id', 'block_num', 'text', 'font_size',
#       'font', 'tag', 'verb', 'structure_title', 'full_structure',
#       'section_id', 'title', 'text_para', 'text_lower', 'para_id']
paragraphs= pd.read_csv(folder_out+"paragraphs_rearranged.csv",delimiter=";",encoding="UTF-8")
# Lower case & Remove punctuation  
paragraphs=clean_text(paragraphs)

#if __name__ == '__main__':

# 4 Extract actors
# The list of actors contains acronyms, full names and words to be detected
# It needs to be before other steps as actors are also link to themes and tags
actors_list= pd.read_csv(folder_list+"actors_list_final.csv",delimiter=";")
load_actors=1
if load_actors == 0:
    actors,paragraphs,tags_actors=extract_actors(paragraphs,actors_list)
    # For counting actors detected (not needed)
    tags_actors.to_csv(folder_out+"tags_actors.csv",sep=";",index=False)
    # List of actors detected for each row keeping the paragraph id
    actors.to_csv(folder_out+"actors.csv",sep=";",index=False)
    # Updated table with actors detected changed to specific "tag" as DPO_actor
    paragraphs.to_csv(folder_out+"paragraphs_with_tag.csv",sep=";",index=False)
    actors_pbi=actors[["para_id","actors"]]
    actors_pbi.to_csv(folder_out+"actors_pbi.csv",sep=";",index=False)
else:
    actors= pd.read_csv(folder_out+"actors.csv",delimiter=";")
    paragraphs= pd.read_csv(folder_out+"paragraphs_with_tag.csv",delimiter=";")
    print("loaded actors and tagged paragraphs")


# 5 Extract dates, location and references based on specific format, for NER see other file
load_others = 0
if load_others == 0:
    dates=extract_dates(paragraphs[['para_id','text_lower']])
    dates.to_csv(folder_out+"dates.csv",sep=";",index=False)
    admins = pd.read_csv(folder_list+"caf_admbnda_adm3_200k_sigcaf_reach_itos_v2.csv",delimiter=";")
    locations=extract_locations(paragraphs[['para_id','text_tag']],admins)
    locations.to_csv(folder_out+"locations.csv",sep=";",index=False)
    references=extract_references(paragraphs[['para_id','text_lower']])
    references.to_csv(folder_out+"references.csv",sep=";",index=False)
else:
    dates= pd.read_csv(folder_out+"dates.csv",delimiter=";")
    print("loaded dates")
    locations= pd.read_csv(folder_out+"locations.csv",delimiter=";")
    print("loaded locations")
    references= pd.read_csv(folder_out+"references.csv",delimiter=";")
    print("loaded references")


# 6 Lemma and Stem
prepare_language=0
if prepare_language==0:
    paragraphs_for_language =prepare_language_tags(paragraphs)
    paragraphs_for_language.to_csv(folder_out+"paragraphs_for_language.csv",sep=";",index=False)
    # in case the last changes from actors create changes
    phenotype_third= extract_phenotype(paragraphs,5)
    phenotype_third.to_csv(folder_out+"phenotype_third.csv",sep=";",index=False,encoding="UTF-8")
else:
    phenotype_third= pd.read_csv(folder_out+"phenotype_third.csv",delimiter=";")
    print("load phenotype third")
    paragraphs_for_language= pd.read_csv(folder_out+"paragraphs_for_language.csv",delimiter=";")
    print("load paragraphs_for_language")

paragraphs=paragraphs_for_language.merge(actors[["para_id","actors"]],on="para_id")


# 7 Extract thematics and tags
load_themes_tags = 0
if load_themes_tags == 0:
    theme_list= pd.read_csv(folder_list+"theme_list.csv",delimiter=";")
    themes,themes_flat=extract_themes(paragraphs[['para_id','text_lemma',"actors"]],theme_list,actors_list)
    themes.to_csv(folder_out+"themes.csv",sep=";",index=False,encoding="UTF-8")
    themes_flat.to_csv(folder_out+"themes_flat.csv",sep=";",index=False)
    tag_list= pd.read_csv(folder_list+"tags_list_final.csv",delimiter=";")
    tags=extract_tags(paragraphs[['para_id',"text_lemma","actors"]],tag_list, actors_list)
    tags.to_csv(folder_out+"tags.csv",sep=";",index=False,encoding="UTF-8")
else:
    themes= pd.read_csv(folder_out+"themes.csv",delimiter=";")
    tags= pd.read_csv(folder_out+"tags.csv",delimiter=";")
    print("loaded themes and tags")
