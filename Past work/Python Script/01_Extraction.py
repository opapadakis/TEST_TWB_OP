import os
import csv
import glob
import pandas as pd
import numpy as np

from Functions import *


cwd = os.getcwd()

# Reports are classified in subfolders
pdf_files=list(glob.glob("../../../Data/Reports_MINSIGHT/**/*.pdf", recursive=True))
# Folder where the files will be saved
folder_out="Report_for_Analysis/Explore/"
folder_images="Report_for_Analysis/Images/"

# 1 Extract Metadata
load_metadata=1
if load_metadata == 0:
    metadata=extract_metadata(pdf_files)
    metadata.to_csv(folder_out+"metadata.csv",sep=';',index=False, encoding='UTF-8') 
else:
    metadata= pd.read_csv(folder_out+"metadata.csv",delimiter=";")
    print("loaded metadata")

# 2 Extract Images/pages
load_images_pages=1
if load_images_pages == 0:
    images_pages_tables=extract_images_pages(pdf_files,folder_images)
    images_pages_tables.to_csv(folder_out+"images_pages.csv",sep=';',index=False, encoding='UTF-8') 
else:
    images_pages_tables= pd.read_csv(folder_out+"images_pages.csv",delimiter=";")
    print("loaded images_pages")

# 3 Extract Paragraphs (most complex coding part)
load_paragraphes =1
if load_paragraphes == 0:
    # First Reconstruction, define titles
    paragraphs= extract_structure(pdf_files,images_pages_tables)
    # Define report type, titles with Roman number, bullet points, ...
    phenotype= extract_phenotype(paragraphs,5)
    # Clean unique characters and take phenotype information to merge or not the lines
    paragraphs_restructured= restructure(paragraphs,phenotype)
    # Report type, titles with Roman number, bullet points, ...
    phenotype_second= extract_phenotype(paragraphs_restructured,-1)
    # Fuse consecutive rows that had a new line character under conditions
    paragraphs_fused_smart= fused_smart(paragraphs_restructured,phenotype_second)
    # Split rows depending on phenotype information
    paragraphs_redivised= redivise(paragraphs_fused_smart,phenotype_second)
    # Adhoc - Clean Santos Cruz tables
    paragraphs_redivised["text_para"]=paragraphs_redivised.text_para[paragraphs_redivised.text_para!="SHORT TERM:"]
    paragraphs_redivised["text_para"]=paragraphs_redivised.text_para[paragraphs_redivised.text_para!="MID- LONG TERM"]
    # Clean unique characters and take phenotype information to merge or not the lines
    paragraphs_restructured2=restructure2(paragraphs_redivised,phenotype_second)
    # Arrange the structure, don't change text in lines
    paragraphs= arrange_structure(paragraphs_restructured2,phenotype_second)
    paragraphs.to_csv(folder_out+"paragraphs_rearranged.csv",sep=";",index=False)
else:
    paragraphs= pd.read_csv(folder_out+"paragraphs_rearranged.csv",delimiter=";",encoding="UTF-8")

