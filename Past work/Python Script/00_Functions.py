# PDF Reader
from PyPDF2 import PdfFileWriter, PdfFileReader
import fitz
from pdfminer.high_level import extract_text
# Table from PDF
import camelot

from textblob import TextBlob

# Word manipulation
from pluralizer import Pluralizer
import inflection
pluralizer = Pluralizer()
#from pattern.en import pluralize, singularize

# Dataframe
import pandas as pd
import numpy as np

# Images processing
from PIL import Image
import imagehash

# System message
import sys

# Time cleaning
from datetime import datetime
from datetime import date

# Regex
import re

# NLP
from gensim.models import Word2Vec
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.remove("against")
from nltk.stem import PorterStemmer,WordNetLemmatizer
ps = PorterStemmer()
wnl = WordNetLemmatizer()
from textblob import Word
from textblob import TextBlob

# Modelling
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt

# Save model
import pickle

# Generate unique identifier
import uuid

# File manipulation
import os
import io
import glob

# Format manipulation
import operator
import string

# Mathematic expressions
import math

# For 01_Extraction
def extract_metadata(list):
    df=pd.DataFrame(list,columns=["file"])
    df["file_name"],df["title"],df["author"],df["creator"],df["creation_date"],df["modif_date"],df["producer"],df["subject"],df["nb_pages"]=["","","","","","","","",""]
    
    for doc in list:
        # Read file
        with open(doc, 'rb') as f:
            pdf = PdfFileReader(f)
            info = pdf.getDocumentInfo()
            nb_pages = pdf.getNumPages()

            df.loc[df["file"]==doc,"file_name"]=reduce_path(doc)
            # Collect information if present
            if info!=None:
                df.loc[df["file"]==doc,"title"]=info.title
                df.loc[df["file"]==doc,"author"]=info.author
                df.loc[df["file"]==doc,"creator"]=info.creator
                df.loc[df["file"]==doc,"creation_date"]=pdf.getDocumentInfo()['/CreationDate']
                df.loc[df["file"]==doc,"modif_date"]=pdf.getDocumentInfo()['/ModDate']
                df.loc[df["file"]==doc,"producer"]=info.producer
                df.loc[df["file"]==doc,"subject"]=info.subject
                df.loc[df["file"]==doc,"nb_pages"]=nb_pages
                
    return df

def extract_images_pages(list_files,folder_out):
    
    # 1 Save each page from report as Image in subfolder Pages
    # 2 Extract all images present in pages in subfolder Extracted
    # 3 Extract all tables present in pages in folder Table
    # 4 List pages with tables or images 

    df=pd.DataFrame(columns=["file_name","page","type_page","image_adress","xb_adress"])
    
    # Read all images that can be a reference
    image_ref=glob.glob(folder_out+"References/*.png")
    for doc in list_files:  
        # need to open with fitz for pictures, and with pdffilereader for tables
        doc_pdf = fitz.open(doc)
        doc_name_short=doc.rsplit('\\', 1)[-1]
        doc_pdf2 = PdfFileReader(open(doc, "rb"))
        for i in range(len(doc_pdf)):
           # 1 Save each page from report as Image in subfolder Pages
            page = doc_pdf.loadPage(i)  
            pix = page.getPixmap()
            output = folder_out+"Pages/"+doc_name_short+"_"+str(i+1)+".png"
            # unused now
            web_adress="https://unitednations-my.sharepoint.com/:i:/r/personal/olivier_papadakis_un_org/Documents/Images/Pages/"+ \
             doc_name_short.replace(" ","%20")+"_"+str(i+1)+".png"
            # Good direction
            xb_adress="https://unitednations.sharepoint.com/:i:/r/sites/DPPA_DPO-PROJ-CARIOT-MINUSCAperformance/Shared%20Documents/Reports_for_Analysis/Images/Pages/"+ \
             doc_name_short.replace(" ","%20")+"_"+str(i+1)+".png"
            pix.writePNG(output)

            all_words = page.getText("text")
            type_page= "annex" if all_words.lower().find("annex")!=-1 else "normal"

            #type_page="normal"
            # classify the page as with a picture
            if doc_pdf.getPageImageList(i)!=[]:
                type_page=type_page+"_picture"

            # 2 Extract all images present in pages in subfolder Extracted
            for img in doc_pdf.getPageImageList(i):
                xref = img[0]
                pixa = fitz.Pixmap(doc_pdf, xref)
                # if GRAY or RGB
                if pixa.n < 5:       
                    pixa.writePNG(folder_out+"Extracted/%s-p%s-%s.png" % (doc_name_short,i, xref))
                # if CMYK: convert to RGB first
                else:               
                    pix1 = fitz.Pixmap(fitz.csRGB, pixa)
                    pix1.writePNG(folder_out+"Extracted/%s-p%s-%s.png" % (doc_name_short,i, xref))
                    pix1 = None
                pixa = None

                # Check if picture size is enough or problem with pdf (for now less than 10ko =10240 we remove)
                if os.stat(folder_out+"Extracted/"+doc_name_short+"-p"+str(i)+"-"+str(xref)+".png").st_size < 10240:
                    os.remove(folder_out+"/Extracted/"+doc_name_short+"-p"+str(i)+"-"+str(xref)+".png")
                else : 
                    # Detect if page has UN logo then it is declared as first page
                    hash0 = imagehash.average_hash(Image.open(folder_out+"Extracted/"+doc_name_short+"-p"+str(i)+"-"+str(xref)+".png")) 
                    for logo_ref in image_ref:
                        hash1 = imagehash.average_hash(Image.open(logo_ref)) 
                        cutoff = 5
                        # if found one ref file stop the loop
                        if hash0 - hash1 < cutoff:
                            # Logo title page
                            if logo_ref.find("gray")!=-1:
                                type_page="first_page"
                                break
                            # logo first page with content for sg report
                            else:
                                #type_page="normal"
                                break
                    del(hash0)
                    del(hash1)
            pix=None
            pixa=None
            # if no more Images after removing in folder extracted then page is without pictures
            if len(glob.glob(folder_out+"/Images/Extracted/"+doc_name_short+"-p"+str(i)+"*"))==0:
                type_page= "annex" if all_words.lower().find("annex")!=-1 else "normal"

            # 3 Extract all tables present in pages in folder Table                  
            output = PdfFileWriter()
            output.addPage(doc_pdf2.getPage(i))
            # Create pdf  of 1 page
            with open(folder_out+"Tables/"+doc_name_short+"_"+str(i+1)+".pdf", "wb") as outputStream:
                output.write(outputStream)
            outputStream.close()
            del(output)
            # If a table, declare the page type as containing it. If not, remove the 1 page pdf file
            try:
                tables = camelot.read_pdf(folder_out+"Tables/"+doc_name_short+"_"+str(i+1)+".pdf")
                if tables.n > 0:
                    type_page=type_page+"_table"
                else: 
                    os.remove(folder_out+"Tables/"+doc_name_short+"_"+str(i+1)+".pdf")
            except:
                type_page=type_page+"_undetect_table"

            # 4 List pages with tables or images
            a=pd.Series([reduce_path(doc), i, type_page, web_adress, xb_adress], index=df.columns )
            df=df.append(a , ignore_index=True)

    return df

def extract_structure(list_doc, page_list):
    # First part of extracting structure, reconstruct lines 
    df=pd.DataFrame(columns=["file_name","page","block_id","block_num","text","font_size","font", \
        "tag","verb","structure_title","full_structure","section_id","title","text_para"])

    for doc in list_doc:
        doc_open = fitz.open(doc) 
    
        # Read the pages to remove title page and annexes
        pages_keep=page_list.loc[page_list["file_name"]==reduce_path(doc)]
        pages_keep=list(pages_keep.loc[pages_keep["type_page"].str.contains("normal")]["page"])

        font_counts, styles = fonts_bold(doc_open, granularity=False)
        # Check if equilibrated file (not a patchwork of different ones), so create title
        tot=0
        tot_color=0
        for f,c in font_counts:
            tot+=c
            # Count number of words in color
            if f.find("color")!=-1:
                tot_color+=c
        # we make the title if main font represents at least half if not different logic inside the report
        make_title= 1 if font_counts[0][1] >= 0.52*tot else 0


        # Give a size tag (header, paragraph, or side note and extracts)
        size_tag=font_tags_bold(font_counts, styles)
        # Color Filter (can be only links, in this case (SG reports), size should be the same as p)
        # find the paragraph "<p>"
        p_font=[k for k,v in size_tag.items() if v == "<p>"][0]
        if tot_color>0 and p_font!=[]: #tot_color<=10 and
            # Put <p> to the color one too
            size_tag[p_font.replace("black","color")]="<p>"

        # Reconstruct lines
        clean=headers_para_page_bold(doc_open, size_tag, pages_keep)

        # ONLY FOR SANTOS CRUZ remove fonts
        if doc.find("santos_cruz")!=-1:
            # F1 for visuals
            clean=clean[clean.font!="F1"]
            # F11 for inserted text in "" and 2 other small sentence without consequences
            clean=clean[clean.font!="F11"]
            # F11 for inserted text in "" twice
            clean=clean[clean.font!="F8"]
        

        # Pb with ; for reading csv after but we keep the version with | for understanding the steps
        clean["text"]=[item.replace(";",",") for item in clean["text"]]
        clean["text"]=[item.replace("\uf0b7","!-!") for item in clean["text"]]
        clean["text"]=[item.replace("\uf0d8","!-!") for item in clean["text"]]

        # Remove numbers for page footer so ("page 1" and "page 2" become "page " and are found most of the time)
        clean["nonum"]=[''.join([i for i in s if not i.isdigit()]) for s in clean.text]
        # Many errors are a ":" in other font and put problem for structure
        clean=clean.loc[clean.nonum!=":|"]
        if len(pages_keep)>5:
            find_footers_repeated=pd.DataFrame(clean.nonum.value_counts()>0.4*len(pages_keep))
            find_footers_repeated=find_footers_repeated.index[find_footers_repeated.nonum==True]
            find_footers_repeated=[word.replace("|","").strip() for word in find_footers_repeated]
            
            # Remove "Recommendation(s)" from the list of footers/headers (if it begins by the word) and bullet points
            find_footers_repeated=[word for word in find_footers_repeated if word.lower()[:14]!="recommendation"]
            find_footers_repeated=[word for word in find_footers_repeated if word.lower()!='•']
            find_footers_repeated=[word for word in find_footers_repeated if word.lower()!='!-!']
        else:
            find_footers_repeated=[]

        clean["verb"],clean["structure_title"],clean["full_structure"],clean["section_id"],clean["title"]=["",np.nan,np.nan,np.nan,""]

        if make_title==1:
            # Keep titles and index
            title_list=pd.DataFrame(clean[clean.tag.str.contains("<h") & clean.text.str.contains("\|")] )
            # Ad - hoc Remove if confidential in one
            title_list=title_list[title_list.text.str.contains("CONFIDENTIAL")==False] 
            title_list=title_list[title_list.text!="Security Council"]
            previous_i=title_list.index[0]
            title_list.text[previous_i]=title_list.text[previous_i].replace("|","").replace("  "," ").strip()
            # Remove headers / footers
            for i in title_list.index:
                if title_list.nonum[i].replace("|","").strip() in find_footers_repeated:
                    title_list=title_list.drop(i)

            previous_i=title_list.index[-1]
            title_list.text[previous_i]=title_list.text[previous_i].replace("|","").strip()
            list_remove=list()
            # Go backward
            for i in title_list.index[::-1][1:]:
                if i==previous_i-1 and title_list.tag[i]==title_list.tag[previous_i]:
                    title_list.text[i]=title_list.text[i]+" "+title_list.text[previous_i]
                    title_list.text[previous_i]=np.nan
                    list_remove.append(previous_i)

                previous_i=i

            # Remove the line with titles on 2 lines
            for i in list_remove:
                title_list=title_list.drop(i)
            
            clean=clean.loc[[item for item in clean.index if item not in list_remove],:]
            # list of tags
            list_tag=list(set(title_list.tag))
            list_tag.sort()
            # Find back the structure from titles
            for idx in title_list.index:
                if len(clean.loc[idx,"text"].replace("|","").strip())>1:
                    clean.loc[idx,"structure_title"]=title_list.text[idx]
                    clean.loc[idx,"title"]="yes"
                    tag=title_list.tag[title_list.index==idx]
                    tags=list(title_list.tag[title_list.index<=idx])
                    tag_idx=list_tag.index(tag[idx])

                    clean.full_structure[idx]=clean.structure_title[idx]
                    clean.loc[idx,"section_id"]=uuid.uuid4()
                    while tag_idx>0 :
                        if list_tag[tag_idx-1] in tags:
                            # Take higher level indexes
                            tag_to_consider=list_tag[0:tag_idx]
                            # First value appearing
                            first_higher=next(item for item in tags[::-1] if item in tag_to_consider)
                            new_tag_idx = len(tags) - 1 - tags[::-1].index(first_higher)
                            # Add higher level title to the structure
                            clean.full_structure[idx]=str(title_list.text[title_list.text.index[new_tag_idx]])+ \
                                "_/_"+clean.full_structure[idx]
                            
                        tag_idx=tag_idx-1
                else:
                    clean.loc[idx,"tag"]= "<p>"
        else: 
            title_list=clean.loc[clean.index[0:1],:]

        
        # Initialize for first
        clean["clean_paragraph"]=np.nan
        clean.loc[0,"clean_paragraph"]=""

        for line_num in clean.index[1:]:
            if line_num in title_list.index:
                clean.loc[line_num,"clean_paragraph"]=title_list.loc[line_num,"text"]  
            else:
                if clean.loc[line_num,"nonum"].replace("|","").strip() not in find_footers_repeated:
                    phrase=clean.loc[line_num,"text"]
                    # if found title, new group easy
                    if line_num-1 in clean.index:
                        if isinstance(clean.loc[line_num-1,"full_structure"],str):
                            clean.loc[line_num,"clean_paragraph"]=phrase
                        else:
                            # in case the first detection would begin with a bullet point
                            if clean["clean_paragraph"].last_valid_index()==None:
                                idx=line_num
                                clean.loc[idx,"clean_paragraph"]=""
                            else:
                                idx=clean["clean_paragraph"].last_valid_index()
                            # If bullet point / list fuse in one block (naturally it's a new one)
                            if re.search(r"^(•|-|●|!-!)",phrase.strip())!=None:#"^(•|-|●)\|"
                                # if only the bullet is followed by something else
                                clean.loc[line_num,"clean_paragraph"]=phrase
                            # if it begins by a lower letter and not a footnote
                            elif re.match("^[a-z,()]",phrase)!=None and clean.loc[idx,"tag"]!="<s>":
                                clean.loc[idx,"clean_paragraph"]= clean.loc[idx,"clean_paragraph"]+" "+phrase
                            # if it begins by capital letter but that was not following a point and no line break and same block
                            elif (re.match("^[A-Z]",phrase)!=None and re.match(". |$",clean.loc[idx,"text"])==None \
                                and clean.loc[idx,"text"][-2:]!="||" \
                                and line_num==idx+1 and clean.loc[idx,"block_id"]==clean.loc[line_num,"block_id"] \
                                    and clean.loc[idx,"tag"]!="<s>" and clean.loc[idx,"tag"]==clean.loc[line_num,"tag"]):
                                clean.loc[idx,"clean_paragraph"]= clean.loc[idx,"clean_paragraph"]+" "+phrase
                            # if it begins by a number but not something like 12. as a number of paragraph and no line break and same block
                            elif (re.match("^[1-9]+",phrase)!=None and re.match("^[1-9]+\.",phrase)==None \
                                and line_num==idx+1 and clean.loc[idx,"block_id"]==clean.loc[line_num,"block_id"] \
                                    and clean.loc[idx,"tag"]!="<s>" and clean.loc[idx,"tag"]==clean.loc[line_num,"tag"]):
                                clean.loc[idx,"clean_paragraph"]= clean.loc[idx,"clean_paragraph"]+" "+phrase

                            else:
                                clean.loc[line_num,"clean_paragraph"]=phrase
                    else:
                        clean.loc[line_num,"clean_paragraph"]=phrase

        # Ad-hoc remove header
        clean["text"]=[item if item.find("CONFIDENTIAL")==-1 else np.nan for item  in clean["text"]]

        clean=clean.loc[clean.text.notnull()]
        clean["text_para"],clean["para_id"]=["",""]
  
        clean["text_para"]=clean["clean_paragraph"]
        clean["text_para"]=[item.replace(";",",") if isinstance(item,str) else item for item in clean["text_para"]]
        clean["text_para"]=[item.replace("\uf0b7","!-!") if isinstance(item,str) else item for item in clean["text_para"]]
        clean["text_para"]=[item.replace("  "," ") if isinstance(item,str) else item for item in clean["text_para"]]
        clean["text_para"]=[item.strip() if isinstance(item,str) else item for item in clean["text_para"]]

        # Unused in the latest version - No verb in the block could mean it is a title
        # for par in clean[clean.text_para.notnull()].index:
        #     # Check if there are verbs used in the block (title detection)
        #     word_tokens = word_tokenize(clean.loc[par,"text_para"])
        #     tag=nltk.pos_tag(word_tokens)
        #     pos_tags = [pos_tag for _,pos_tag in tag]
        #     fd = nltk.FreqDist(pos_tags)
        #     # only 
        #     if fd["VB"]+fd["VBP"]+fd["VBZ"]+fd["VBD"]==0:
        #         clean.loc[par,"verb"]=0
        #     elif fd["VBD"]>0 and fd["VB"]+fd["VBP"]+fd["VBZ"]==0:
        #         clean.loc[par,"verb"]=2
        #     else:
        #         clean.loc[par,"verb"]=1
            

        clean["text"]=[item.replace("|","") for item in clean["text"]]

        # if not found the first title
        if not isinstance(clean.structure_title[clean.index[0]],str):
            clean.structure_title[clean.index[0]]="first_part_no_title"
            clean.full_structure[clean.index[0]]="first_part_no_title"
            clean.title[clean.index[0]]="yes"
            clean.section_id[clean.index[0]]=uuid.uuid4()

        df=df.append(clean[['file_name', 'page', 'block_id','block_num', 'text', 'font_size', 'font', 'tag', 'verb', 'structure_title','full_structure','section_id','title','text_para']], ignore_index = True) 

    return df

def fonts_bold(doc, granularity=False):
    # Extracts fonts and their usage in PDF documents.
    # Distinction is made on bold, italic and color
    styles = {}
    font_counts = {}

    for page in doc:
        blocks = page.getText("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # block contains text
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if granularity:
                            identifier = "{0}_{1}_{2}_{3}".format(s['size'], s['flags'], s['font'], s['color'])
                            styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                  'color': s['color']}
                        else:
                            fo="bolditalic" if ((re.search("Bold",s["font"])!=None) and (re.search("Italic",s["font"])!=None)) \
                                else "bold" if (re.search("Bold",s["font"])!=None) else "italic" if re.search("Italic",s["font"])!=None else "normal"
                            color="black" if s['color']==0 else "color"
                            identifier = "{0}_{1}_{2}".format(round(float(s['size']),0),fo,color)
                            # add for merging close found by the system
                            styles[identifier] = {'size': s['size'],  'font': fo , 'color':color}

                        font_counts[identifier] = font_counts.get(identifier, 0) + 1  # count the fonts usage

    font_counts = sorted(font_counts.items(), key=operator.itemgetter(1), reverse=True)


    if len(font_counts) < 1:
        raise ValueError("Zero discriminating fonts found!")

    return font_counts, styles

def font_tags_bold(font_counts, styles):
    #Returns dictionary with font sizes as keys and tags as value.

    p_style = styles[font_counts[0][0]]  # get style for most used font by count (paragraph)
    p_size = round(p_style['size'],0)  # get the paragraph's size
    p_font = p_style['font']
    p_color = p_style['color']
    # sorting the font sizes high to low, so that we can append the right integer to each tag 
    font_sizes = []
    for (font_size, count) in font_counts:
        font_sizes.append(float(font_size.split("_")[0]))
    font_sizes=list(set(font_sizes))
    font_sizes.sort(reverse=True)

    # aggregating the tags for each font size
    # The purpose is for everything in bold/color to be as a header if it is at least the size of the common paragraph
    # as "bold" is before "normal" we ensure that the bold characters would be seen as a higher header
    idx = 0
    size_tag = {}
    for size in font_sizes:
        fonts = dict(filter(lambda item: item[0].startswith(str(size)),  styles.items()))
        fonts=sorted(fonts)
        for font in fonts:
            idx += 1
            # it is the same size, we look for boldness and color
            if round(size,0) == p_size:
                # if both are the same, it is a paragraph
                if font.split("_")[1]==p_font and font.split("_")[2]==p_color:
                    size_tag[font] = '<p>'
                # if one of the 3 is higher (bold, italic>normal, color>normal), classify it as a header
                elif ((font.split("_")[1].find("bold")!=-1 or font.split("_")[1].find("italic")!=-1)  and p_font=="normal") or \
                    (font.split("_")[1]=="italic" and p_font=="normal") or (font.split("_")[2]=="color" and p_color=="black"):
                    size_tag[font] = "<h"+str(idx).zfill(2)+">"
                
                # else we put it as a note (it could be as a paragraph) 
                else:
                    size_tag[font] = "<s>"
            if size > p_size:
                size_tag[font] = "<h"+str(idx).zfill(2)+">"
            elif size < p_size:
                size_tag[font] = "<s>"

    return size_tag

def headers_para_page_bold(doc, size_tag,pages_keep):
    #Scrapes headers & paragraphs from PDF and return texts with element tags.

    first = True  # boolean operator for first header
    previous_s = {}  # previous span
    df=pd.DataFrame(columns=["file_name","page","block_id","block_num","text","font_size","font","tag"])
    count_blocks=0

    for page in doc:
        # If the page is not indicated as annexes 
        if page.number>= pages_keep[0] and page.number<=pages_keep[-1]:
            blocks = page.getText("dict")["blocks"]

            for b in blocks:  # iterate through the text blocks
                if b['type'] == 0:  # this block contains text
                    count_blocks=count_blocks+1

                    # REMEMBER: multiple fonts, colors and sizes are possible IN one block
                    id_blocks=uuid.uuid4()
                    block_string = ""  # save text
                    for l in b["lines"]:  # iterate through the text lines
                        for s in l["spans"]:  # iterate through the text spans
                            if s['text'].strip():  # removing whitespaces:
                                if first:
                                    previous_s = s
                                    first = False
                                    siz=round(s['size'],0)
                                    fo="bolditalic" if ((re.search("Bold",s["font"])!=None) and (re.search("Italic",s["font"])!=None)) \
                                        else "bold" if (re.search("Bold",s["font"])!=None) else "italic" \
                                            if re.search("Italic",s["font"])!=None else "normal"

                                    color="black" if s['color']==0 else "color"
                                    a=pd.Series([reduce_path(doc.name), page.number+1, id_blocks,count_blocks, \
                                        s['text'], siz,s['font'],size_tag[str(siz)+"_"+fo+"_"+color]], index=df.columns )
                                    # count block will be later a unique identifier, but easier to review with increasing numbers
                                    df=df.append(a , ignore_index=True)
                                else:
                                    if s['size'] == previous_s['size'] and s['font'] == previous_s['font'] and s['color'] == previous_s['color'] :
                                        if block_string and all((c == "|") for c in block_string):
                                            block_string = s['text']
                                            
                                        elif block_string == "":
                                            # new block has started, so append size tag
                                            block_string = s['text']
                                            
                                        else:  # in the same block, but "||"" so two different pararaphs
                                            
                                            if  block_string[-2:]=="||":
                                                siz=round(previous_s['size'],0)
                                                fo="bolditalic" if ((re.search("Bold",previous_s["font"])!=None) and (re.search("Italic",previous_s["font"])!=None)) \
                                                    else "bold" if (re.search("Bold",previous_s["font"])!=None) else "italic" \
                                                    if re.search("Italic",previous_s["font"])!=None else "normal"
                                                color="black" if previous_s['color']==0 else "color"
                                                a=pd.Series([reduce_path(doc.name), page.number+1, id_blocks,count_blocks,
                                                block_string,siz,previous_s['font'],size_tag[str(siz)+"_"+fo+"_"+color]], index=df.columns )
                                                # count block should be later uuid.uuid4()
                                                df=df.append(a , ignore_index=True)
                                                block_string = s['text']
                                                previous_s = s
                                            else: # in the same block, concatenate paragraphs
                                                block_string += " " + s['text']

                                    else: # in the same block, so concatenate strings
                                        # if new string add previous one to the file
                                        if (block_string!="" and not (all((c == "|") for c in block_string))):
                                            siz=round(previous_s['size'],0)
                                            fo="bolditalic" if ((re.search("Bold",previous_s["font"])!=None) and (re.search("Italic",previous_s["font"])!=None)) \
                                                else "bold" if (re.search("Bold",previous_s["font"])!=None) else "italic" \
                                                if re.search("Italic",previous_s["font"])!=None else "normal"
                                            color="black" if previous_s['color']==0 else "color"
                                            a=pd.Series([reduce_path(doc.name), page.number+1, id_blocks, \
                                            count_blocks,block_string,siz,previous_s['font'],size_tag[str(siz)+"_"+fo+"_"+color]], index=df.columns )
                                            df=df.append(a , ignore_index=True)

                                            
                                        # now update
                                        block_string =  s['text']

                                    previous_s = s

                        # new block started, indicating with a pipe
                        block_string += "|"

                    # if not first line
                    if len(df.index)!=0:#s["text"]!=" ":
                        # if text found is different from previous one + "|"
                        if block_string!=list(df.text)[-1]+"|":
                            # if current text is empty, take saved previous value
                            if s['text'].strip()=="":
                                siz=round(previous_s['size'],0)
                                fo="bolditalic" if ((re.search("Bold",previous_s["font"])!=None) and (re.search("Italic",previous_s["font"])!=None)) \
                                    else "bold" if (re.search("Bold",previous_s["font"])!=None) else "italic" \
                                    if re.search("Italic",previous_s["font"])!=None else "normal"
                                color="black" if previous_s['color']==0 else "color"
                                a=pd.Series([reduce_path(doc.name), page.number+1, id_blocks,count_blocks,
                                block_string,siz,previous_s['font'],size_tag[str(siz)+"_"+fo+"_"+color]], index=df.columns )
                                
                            # save current value
                            else:
                                siz=round(s['size'],0)
                                fo="bolditalic" if ((re.search("Bold",s["font"])!=None) and (re.search("Italic",s["font"])!=None)) \
                                        else "bold" if (re.search("Bold",s["font"])!=None) else "italic" \
                                        if re.search("Italic",s["font"])!=None else "normal"
                                color="black" if s['color']==0 else "color"
                                a=pd.Series([reduce_path(doc.name), page.number+1, id_blocks,count_blocks,
                                block_string,siz,s['font'],size_tag[str(siz)+"_"+fo+"_"+color]], index=df.columns )
                            df=df.append(a , ignore_index=True)

    df2=pd.DataFrame(columns=df.columns)
    for block in df["block_id"].unique():
        # get all and fuse if same font (1st, 2nd, etc..)
        line=df.loc[df["block_id"]==block,:]
        if len(line["tag"].unique())==1:
            bl=" ".join(line["text"])
            keep=line.iloc[0,:]
            keep.loc["text"]=bl
            a=pd.Series(keep, index=df2.columns )
            df2=df2.append(a , ignore_index=True)
        else:
            df2=df2.append(line , ignore_index=True)

    return df2

def extract_phenotype(df2,length):
    # List elements characteristic of the report
    df=pd.DataFrame(columns=["file_name","num_pt","num_par","letter_low_notitle","letter_low_title","letter_up","letter_par","letter_end",\
         "roman", "bullet", "big_bullet", "unfound", "tiret","lot_s_tag","length","type"])
    
    list_doc=set(df2.file_name)

    for doc in list_doc:
        para=df2.loc[df2.file_name==doc]
        para= para[para.text_para.notnull()]
        para["beginning"],para["end"]=["",""]
        nb_para=len(para)
        # Take 5 first characters of the paragraph in the first call, then all
        if length==-1:
            para["beginning"] = [item.strip()  for item in para.text_para]
        else:
            para["beginning"] = [item.strip()[:length] for item in para.text_para]
        para["end"] = [item.strip()[-5:] for item in para.text]
        count_all={"num_pt":0,"num_par":0,"letter_low_notitle":0,"letter_low_title":0,"letter_up":0, "letter_par":0, "letter_end":0,"roman":0, "bullet":0, \
             "big_bullet":0, "unfound":0, "tiret":0, "lot_s_tag":0}
        
        for item in para.index:
        # Count the number of paragraphs beginning by "1." but no "1.0"
            if re.search("^[0-9]{1,3}\.\D?",para.beginning[item].strip())!=None:
                count_all["num_pt"]+=1
        # Count the number of paragraphs beginning by "1)"
            if re.search("^[0-9]{1,3}\)",para.beginning[item].strip())!=None:
                count_all["num_par"]+=1
        # Count the number of paragraphs beginning by "a."
            if re.search("^[a-z]{1}\.",para.beginning[item].strip())!=None and para.title[item]!="yes":
                count_all["letter_low_notitle"]+=1
        # Count the number of paragraphs beginning by "a."
            if re.search("^[a-z]{1}\.",para.beginning[item].strip())!=None and para.title[item]=="yes":
                count_all["letter_low_title"]+=1
        # Count the number of paragraphs beginning by "A."
            if re.search("^[A-Z]{1}\.",para.beginning[item].strip())!=None:
                count_all["letter_up"]+=1
        # Count the number of paragraphs beginning by "(a)"
            if re.search("^\([a-z]\)",para.beginning[item].strip())!=None:
                count_all["letter_par"]+=1
        # Count the number of paragraphs beginning by "I."
            if re.search("^[I,V,X]{1,4}\.",para.beginning[item].strip())!=None:
                count_all["roman"]+=1
        # Count the number of paragraphs beginning by "-"
            if re.search("^-",para.beginning[item].strip())!=None:
                count_all["tiret"]+=1
        # Count the number of paragraphs beginning by "!-!"
            if re.search("^!-!",para.beginning[item].strip())!=None:
                count_all["unfound"]+=1
        # Count the number of paragraphs beginning by "•"
            if re.search("•",para.beginning[item].strip())!=None:
                count_all["bullet"]+=1
        # Count the number of paragraphs beginning by "●"
            if re.search("●",para.beginning[item].strip())!=None:
                count_all["big_bullet"]+=1
            if re.search("\s[a-z]{1}\.$",para.end[item].strip())!=None:
                count_all["letter_end"]+=1
        # Count the number of paragraphs begin flagged as footers (for weird files)
            if para.tag[item]=="<s>":
                count_all["lot_s_tag"]+=1

        max_val=max(count_all, key=count_all.get)
        # Ensure mote than 20% of the paragraphs
        type_file=""
        # Paragraph indicator, so "bullet need to be a %"
        if count_all["num_pt"]*100/nb_para>10:
            type_file=type_file+"num_pt_"
        if count_all["unfound"]*100/nb_para>10 :
            type_file=type_file+"unfound_"
        if count_all["bullet"]*100/nb_para>5 :
            type_file=type_file+"bullet_"
        #try to find hierarchy (at least 3 titles pb SEA.)
        if count_all["roman"]>2 :
            type_file=type_file+"roman_" 
        if count_all["letter_up"]>2 :
            type_file=type_file+"letter_up_" 
        if count_all["letter_low_notitle"]>2 and count_all["letter_low_title"]==0:
            type_file=type_file+"letter_low_notitle"  
        if count_all["letter_low_title"]>2 :
            type_file=type_file+"letter_low_title"  
        if count_all["letter_par"]>2 :
            type_file=type_file+"letter_par_"   
        if count_all["num_par"]>2 :
            type_file=type_file+"num_par_"  
        if count_all["letter_end"]>0:
            type_file=type_file+"letter_end_"  
        # Footnote size problem
        if count_all["lot_s_tag"]*100/nb_para>30:
            type_file=type_file+"s_tag_" 
        # no specific features found
        if type_file=="":
            type_file="undetermined_"

        type_file=type_file[:-1]
        a=pd.Series([doc, count_all["num_pt"], count_all["num_par"], count_all["letter_low_notitle"],count_all["letter_low_title"], \
            count_all["letter_up"], count_all["letter_par"], count_all["letter_end"],count_all["roman"], count_all["bullet"], \
            count_all["big_bullet"], count_all["unfound"], count_all["tiret"],count_all["lot_s_tag"], nb_para, type_file], index=df.columns )
        df=df.append(a , ignore_index=True)

    return df

def restructure(df,phenotype):
    # Use phenotype
    df2=pd.DataFrame(columns=df.columns)

    list_doc=set(df.file_name)
    phenotyp={}
    keep_pheno={}
    title_defined={}
    first_title={}
    # Prepare regex depending on phenotype
    for doc in list_doc:
        pheno=phenotype.loc[phenotype.file_name==doc,"type"]
        pheno=list(pheno)[0]
        regex=""
        if pheno.find("num_pt")!=-1:
            regex="[0-9]{1,3}\.\D"
        if pheno.find("unfound")!=-1:
            regex="!-!"
        if pheno.find("roman")!=-1:
            regex=regex+"|^[I,V,X]{1,4}\."
        if pheno.find("letter_up")!=-1:
            regex=regex+"|^[A-Z]{1}\."
        if pheno.find("letter_low")!=-1:
            regex=regex+"|^[a-z]{1}\."
        if pheno.find("letter_par")!=-1:
            regex=regex+"|^\([a-z]\)"
        if pheno.find("num_par")!=-1:
            regex=regex+"|^[0-9]{1,3}\)"
        regex="("+regex+")"
        phenotyp[doc] =regex
        keep_pheno[doc] =pheno
        title_defined[doc]=len(set(df.loc[df.file_name==doc,"structure_title"]))
        a=df.loc[df.file_name==doc,"structure_title"]
        first_title[doc]=a[a.index[0]]

    df=df[df.text_para.notnull()]
    # For each line look for specific structures and adjust to the phenotype
    # Lots of conditions but it is repeated
    for idx in df.index:
        line=df.loc[idx,:]
        regex=phenotyp[df.loc[idx,"file_name"]]
        # only used for unfound, we then consider that empty lines mean something for paragraph division
        # and as such don't fuse if the line before is empty.
        pheno=keep_pheno[df.loc[idx,"file_name"]]
        # If reconstruction found a single character on the line that indicates the beginning of a new line fuse to the next one
        if df.loc[idx,"text_para"].strip()=="(":
            df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + "("
        elif df.loc[idx,"text_para"].replace("|","").strip()=="•" or df.loc[idx,"text_para"].replace("|","").strip()=="●":
                df.loc[idx+1,"text_para"]="•"+df.loc[idx+1,"text_para"]
        elif df.loc[idx,"text_para"].replace("|","").strip()=="!-!" :
                df.loc[idx+1,"text_para"]="!-! "+df.loc[idx+1,"text_para"]    
        else:
            if pheno=="unfound":
                if idx-1 in df.index:
                    if line.title=="yes":
                        a=pd.Series(line, index=df2.columns )
                        df2=df2.append(a , ignore_index=True)
                    # for undetected title at the beginning of reports
                    elif first_title[df.loc[idx,"file_name"]]=="first_part_no_title":  
                        # if "-" or the expression define for the file is found at the beginning, with same tag, and last sentence has not || (2 empty lines)
                        if re.search("^-",df.loc[idx,"text_para"].strip()[:4])!=None and df2.tag.iloc[-1]==df.loc[idx,"tag"] \
                            and df2.text_para.iloc[-1][-2:]!="||":# \
                            df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"]  
                        # if same font and none of the specific features of the reports are detected
                        elif re.search(regex,df.loc[idx,"text_para"].strip()[:4])==None and df2.tag.iloc[-1]==df.loc[idx,"tag"] \
                            and df2.text_para.iloc[-1].strip()[-2:]!="||":
                            df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"] 
                        # if same block, same tag, and the first one doesn't have a new line character
                        elif df2.block_id.iloc[-1]==df.loc[idx,"block_id"] and df2.tag.iloc[-1]==df.loc[idx,"tag"] \
                            and df2.text_para.iloc[-1][-1:]!="|":
                                df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"]  
                        else:
                            a=pd.Series(line, index=df2.columns )
                            df2=df2.append(a , ignore_index=True)
                    else:
                        # If not a footnote
                        if df.tag[idx]!="<s>":
                            # if same font and none of the specific features of the reports are detected and not a new paragraph
                            if re.search(regex,df.loc[idx,"text_para"].strip()[:4])==None and df2.tag.iloc[-1]==df.loc[idx,"tag"] \
                                and df2.text_para.iloc[-1][-2:]!="||":
                                    df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"]
                            # if same font and "-" at the beginning and not a new paragraph
                            elif re.search("^-",df.loc[idx,"text_para"].strip()[:4])!=None and df2.tag.iloc[-1]==df.loc[idx,"tag"] \
                                and df2.text_para.iloc[-1][-2:]!="||":
                                    df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"]
                            # if same block, same tag, and the first one doesn't have a new line character
                            elif df2.block_id.iloc[-1]==df.loc[idx,"block_id"] and df2.tag.iloc[-1]==df.loc[idx,"tag"] \
                            and df2.text_para.iloc[-1][-1:]!="|":
                                df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"]        
                            else:
                                a=pd.Series(line, index=df2.columns )
                                df2=df2.append(a , ignore_index=True)
                else:
                                a=pd.Series(line, index=df2.columns )
                                df2=df2.append(a , ignore_index=True)

            else:
                # if structure title and text are the same it is a title so don't change
                if line.title=="yes":
                    a=pd.Series(line, index=df2.columns )
                    df2=df2.append(a , ignore_index=True)
                # if no title found at the beginning
                elif first_title[df.loc[idx,"file_name"]]=="first_part_no_title":  
                    # if "-" or the expression define for the file is found at the beginning, with same tag, and last sentence has not || (2 empty lines)
                    if re.search("^-",df.loc[idx,"text_para"].strip()[:4])!=None and df2.tag.iloc[-1]==df.loc[idx,"tag"] \
                        and df2.text_para.iloc[-1][-2:]!="||":# \
                        df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"]   
                    # if same font and none of the specific features of the reports are detected and not a new paragraph
                    elif re.search(regex,df.loc[idx,"text_para"].strip()[:4])==None and df2.tag.iloc[-1]==df.loc[idx,"tag"] \
                        and df2.text_para.iloc[-1].strip()[-2:]!="||":
                        df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"]  
                    # if same block, same tag, and the first one doesn't have a new line character
                    elif df2.block_id.iloc[-1]==df.loc[idx,"block_id"] and df2.tag.iloc[-1]==df.loc[idx,"tag"] \
                            and df2.text_para.iloc[-1][-1:]!="|":
                                df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"] 
                    else:
                            a=pd.Series(line, index=df2.columns )
                            df2=df2.append(a , ignore_index=True)
                else:
                    if df.tag[idx]!="<s>":
                        # if same font and none of the specific features of the reports are detected and not a new paragraph
                        if re.search(regex,df.loc[idx,"text_para"].strip()[:4])==None and df2.tag.iloc[-1]==df.loc[idx,"tag"] \
                            and df2.text_para.iloc[-1][-2:]!="||":
                                df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"]
                        # if "-" or the expression define for the file is found at the beginning, with same tag, and last sentence has not || (2 empty lines)
                        elif re.search("^-",df.loc[idx,"text_para"].strip()[:4])!=None and df2.tag.iloc[-1]==df.loc[idx,"tag"] \
                            and df2.text_para.iloc[-1][-2:]!="||":
                                df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"] 
                        # if same block, same tag, and the first one doesn't have a new line character
                        elif df2.block_id.iloc[-1]==df.loc[idx,"block_id"] and df2.tag.iloc[-1]==df.loc[idx,"tag"] \
                            and df2.text_para.iloc[-1][-1:]!="|":
                                df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"] 

                        else:
                            a=pd.Series(line, index=df2.columns )
                            df2=df2.append(a , ignore_index=True)
            
                    else:
                        a=pd.Series(line, index=df2.columns )
                        df2=df2.append(a , ignore_index=True)
        
    return df2

def fused_smart(df,phenotype):
    # Look for "|" at end of text in consecutive rows
    # Only try to fuse rows that were consecutive since the first extraction
    df2=pd.DataFrame(columns=df.columns)
    df=df[df.text_para.notnull()]
    df=df[df.text_para!=""]
    df=df.reset_index()

    phenotyp={}
    keep_pheno={}
    for doc in list(set(df.file_name)):
        pheno=phenotype.loc[phenotype.file_name==doc,"type"]
        pheno=list(pheno)[0]
     
        regex=""
        if pheno.find("num_pt")!=-1:
            regex="[0-9]{1,3}\.\D"
        if pheno.find("unfound")!=-1:
            regex="!-!"
        if pheno.find("roman")!=-1:
            regex=regex+"|^[I,V,X]{1,4}\."
        if pheno.find("letter_up")!=-1:
            regex=regex+"|^[A-Z]{1}\."
        if pheno.find("letter_low_titl")!=-1:
            regex=regex+"|^[a-z]{1}\."
        if pheno.find("letter_par")!=-1:
            regex=regex+"|^\([a-z]\)"
        if pheno.find("num_par")!=-1:
            regex=regex+"|^[0-9]{1,3}\)"
        regex="("+regex+")"
        phenotyp[doc] =regex
        keep_pheno[doc] =pheno


    df=df[df.text_para.notnull()]
    df=df.reset_index()
    df["end_of_line"]=[0 if item[-1]=="|" else 1 for item in df["text_para"]]
    df["successive"]=0
    for idx in df.index[1:]:
        if df.loc[idx,"end_of_line"]==1 or df.loc[idx-1,"end_of_line"]==1:
            df.loc[idx,"successive"]=1

    index_successive=df[df.successive==1].index
    index_remove=[]
    for idx in df.index:
        
        # if successive 
        if idx in index_successive:
            # if not already removed
            if idx not in index_remove:
            # if first one is <p>
                if df.loc[idx,"tag"]=="<p>":
                    a=1
                    regex =phenotyp[df.loc[idx,"file_name"]]
                    pheno=keep_pheno[df.loc[idx,"file_name"]]
                    # find last one keeping alternance <p> and <h> without regex
                    while ((idx+a) in index_successive) and (df.loc[idx+a,"file_name"]==df.loc[idx,"file_name"]) and \
                        ((a%2==1 and df.loc[idx+a,"tag"].find("<h")!=-1) or (a%2==0 and df.loc[idx+a,"tag"]=="<p>")) and \
                            (re.search(regex,df.loc[idx+a,"text_para"])==None):
                        index_remove.append(idx+a)
                        a=a+1
        
                    # fuse
                    for i in range(1,a):
                        df.loc[idx,"text_para"]=df.loc[idx,"text_para"]+ " " + df.loc[idx+i,"text_para"]

                line=df.loc[idx,:]
                a=pd.Series(line, index=df2.columns )
                df2=df2.append(a , ignore_index=True)
        else:
            line=df.loc[idx,:]
            a=pd.Series(line, index=df2.columns )
            df2=df2.append(a , ignore_index=True)

         
    return df2

def redivise(df,phenotype):
    # Split lines where specific report features are detected
    df=df[df.text_para.notnull()]      
    df2=pd.DataFrame(columns=df.columns)
    list_doc=list(set(df.file_name))
    phenotyp={}
    keep_pheno={}
    title_defined={}
    for doc in list_doc:
     
        pheno=phenotype.loc[phenotype.file_name==doc,"type"]
        pheno=list(pheno)[0]
        regex=""
        # we add \s because we're looking for the ones that are paste inside an other sentence
        if pheno.find("num_pt")!=-1:
            regex="|\s[0-9]{1,3}\.\D"
        if pheno.find("unfound")!=-1:
            regex="|!-!"
        if pheno.find("letter_end")!=-1:
            regex=regex+"|\s[a-z]{1}\."
        if pheno.find("roman")!=-1:
            regex=regex+"|\s[I,V,X]{1,4}\."
        if pheno.find("letter_up")!=-1:
            regex=regex+"|\s[A-Z]{1}\."
        if pheno.find("letter_low")!=-1:
            regex=regex+"|\s[a-z]{1}\."
        if pheno.find("letter_par")!=-1:
            regex=regex+"|\([a-z]\)"
        if pheno.find("num_par")!=-1:
            regex=regex+"|\s[0-9]{1,3}\)"
        if pheno.find("bullet")!=-1 and pheno.find("tag")==-1:
            regex=regex+"|•"
        # Add double line extract 
        regex=regex+"|\\|\\|"
        # remove first | if needed (and taking undetermined into account)
        regex =regex if regex=="" else regex if regex[0]!="|" else regex[1:]
        
        regex="(?<!^)(?="+regex+")"
        phenotyp[doc] =regex
        keep_pheno[doc]=pheno
        title_defined[doc]=len(set(df.loc[df.file_name==doc,"structure_title"]))

    keep_title=pd.DataFrame(columns=["old","new","file"])
    for idx in df.index:
        # Remove only number lines
        if df.loc[idx,"text_para"].isnumeric()==False and len(df.loc[idx,"text_para"])>1:

            regex =phenotyp[df.loc[idx,"file_name"]]
            pheno=keep_pheno[df.loc[idx,"file_name"]]
            # if footer don't take, unless it is a file with many footers (in case it is important)
            if ((df.loc[idx,"tag"]!="<s>" and pheno.find("s_tag")==-1) or pheno.find("s_tag")!=-1):
                line=df.loc[idx,:]
                new_lines=re.compile(regex).split(df.loc[idx,"text_para"])
                if len(new_lines)>1:
                    new_lines=[item for item in new_lines if item!=None]
                    # to keep only the first one in cleaning titles
                    first=True
                    for para in new_lines:
                        # remove lines with less than 3 characters
                        if len(para.strip())>3:
                            line.text_para=para.strip()
                            line.para_id=uuid.uuid4()
                            # remove title from the second line if it was one
                            line.title="no" if new_lines.index(para)!=0 else line.title
                            
                            # if title present, mark it as such
                            if isinstance(line.structure_title,str):
                                line.structure_title=para.strip()
                                line.title="yes"
                            a=pd.Series(line, index=df2.columns )
                            df2=df2.append(a , ignore_index=True)
                else:
                    a=pd.Series(line, index=df2.columns )
                    df2=df2.append(a , ignore_index=True)

    # clean structure titles
    for idx in df2[df2.structure_title.notnull()].index:
        df2.loc[idx,"structure_title"]=df2.loc[idx,"structure_title"].replace("|","").strip()

    # for each file create full list 
    for doc in list_doc:

        list_title=list(df2.structure_title[(df2["file_name"]==doc) &  df2.structure_title.notnull()])
        # remove the one not in list
        for idx in df2[(df2["file_name"]==doc) &  (df2.structure_title.notnull())].index:
            if df2.loc[idx,"structure_title"].find("_/_")!=-1 :
                tmp=re.split("\|\||",df2.loc[idx,"full_structure"])
            else:
                tmp=re.split("\|\||_\/_",df2.loc[idx,"full_structure"])
            tmp=[item.replace("|","").strip() for item in tmp]
            tmp=[item for item in tmp if item!=""]
            z=[item for item in tmp if item in list_title ]
            lookup = set()  # a temporary lookup set
            z = [x for x in z if x not in lookup and lookup.add(x) is None]
            df2.loc[idx,"full_structure"]="_/_".join(z)


    df2["text_para"]=[item.replace("|","") for item in df2["text_para"]]
    df2["text_para"]=[item.replace("  "," ").strip() for item in df2["text_para"]]                
    return df2

def restructure2(df,phenotype):
    # Similar to restructure, fuse lines with only one character and a. depending on the report phenotype
    df2=pd.DataFrame(columns=df.columns)

    list_doc=set(df.file_name)
    phenotyp={}
    keep_pheno={}
    title_defined={}
    first_title={}
    # Prepare regex depending on phenotype
    for doc in list_doc:
        pheno=phenotype.loc[phenotype.file_name==doc,"type"]
        pheno=list(pheno)[0]
        regex=""
        if pheno.find("num_pt")!=-1:
            regex="[0-9]{1,3}\.\D?"
        if pheno.find("unfound")!=-1:
            regex="!-!"
        if pheno.find("roman")!=-1:
            regex=regex+"|^[I,V,X]{1,4}\."
        if pheno.find("letter_up")!=-1:
            regex=regex+"|^[A-Z]{1}\."
        if pheno.find("letter_low")!=-1:
            regex=regex+"|^[a-z]{1}\."
        if pheno.find("letter_par")!=-1:
            regex=regex+"|^\([a-z]\)"
        if pheno.find("num_par")!=-1:
            regex=regex+"|^[0-9]{1,3}\)"
        regex="("+regex+")"
        phenotyp[doc] =regex
        keep_pheno[doc] =pheno
        title_defined[doc]=len(set(df.loc[df.file_name==doc,"structure_title"]))
        a=df.loc[df.file_name==doc,"structure_title"]
        first_title[doc]=a[a.index[0]]

    df=df[df.text_para.notnull()]
    for idx in df.index:
        line=df.loc[idx,:]
        regex=phenotyp[df.loc[idx,"file_name"]]
        pheno=keep_pheno[df.loc[idx,"file_name"]]
      
        # if begin by small letter not a sign of title
        if (re.search("^[a-z,]",df.loc[idx,"text_para"].strip()[:1])!=None) and \
            (re.search(regex,df.loc[idx,"text_para"].strip()[:2])==None):

            df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"] 
        # if the last one end by "(" copy
        elif len(df2)!=0:
            if df2.text_para.iloc[-1]=="(":
                df2.text_para.iloc[-1]=df2.text_para.iloc[-1] + " " + df.loc[idx,"text_para"] 
            else:
                a=pd.Series(line, index=df2.columns )
                df2=df2.append(a , ignore_index=True)
        else:
            a=pd.Series(line, index=df2.columns )
            df2=df2.append(a , ignore_index=True)
         
    return df2

def arrange_structure(df,phenotype):
    # Arrange the structure and complete if empty
    phenotyp={}
    keep_pheno={}

    docs=set(df.file_name)
    keep=[]
    df=df[df.text_para.notnull()] 
    df["text_lower"]=[item.lower() if isinstance(item,str) else item for item in df["text_para"]] 
    for doc in docs:
        pheno=phenotype.loc[phenotype.file_name==doc,"type"]
        pheno=list(pheno)[0]
        regex=""
        # Only for titles, 1. is usually for paragraphs
        if pheno.find("roman")!=-1:
            regex=regex+"|[I,V,X]{1,4}\."
        if pheno.find("letter_up")!=-1:
            regex=regex+"|[A-Z]{1}\."
        if pheno.find("letter_low_titl")!=-1:
            regex=regex+"|[a-z]{1}\."
        regex="("+regex+")"
        regex =regex if regex!="()" else "(^itcantbethis$)" 
        phenotyp[doc] =regex
        keep_pheno[doc] =pheno
        # in case the title get removed / fuse / divide before
        verif=df.loc[df.file_name==doc,"full_structure"].index[0]
        if not isinstance(df.loc[verif,"full_structure"],str):
            df.loc[verif,"full_structure"]="first_part_no_title"
            df.loc[verif,"structure_title"]="first_part_no_title"
            df.loc[verif,"title"]="yes"
            df.loc[verif,"section_id"]=uuid.uuid4()
        # If title in all structure name
        tmp=df.loc[(df.file_name==doc) & (df.full_structure.isnull()==False),"full_structure"]
        # value to check
        ti=tmp[tmp.index[0]]
        check=[1 if s.find(ti)!=-1 else 0 for s in tmp]
        if all(check) & (ti!="first_part_no_title"):
            for idx in df.loc[(df.file_name==doc) & (df.full_structure.isnull()==False)].index:
                df.loc[idx,"full_structure"]=df.loc[idx,"full_structure"].replace(ti+"/","")
        # for file without structure
        first=True
        for idx in df[df.file_name==doc].index:
            if first==True:
                if df.loc[idx,"text_para"].find("I.")==0 or df.loc[idx,"text_para"].find("V.")==0:
                    df.loc[idx,"structure_title"]=df.loc[idx,"text_para"]
                    df.loc[idx,"full_structure"]=df.loc[idx,"text_para"]
                    df.loc[idx,"section_id"]=uuid.uuid4()
                first=False
            else:
                if (re.search(regex,df.loc[idx,"text_para"].strip())!=None and re.search(regex,df.loc[idx,"text_para"].strip()).regs[0][0]>=0 and \
                    re.search(regex,df.loc[idx,"text_para"].strip()).regs[0][0]<=4) or \
                    df.loc[idx,"text_lower"].strip().find("recommendation")==0 or df.loc[idx,"text_lower"].strip().find("executive summary")==0:
                    # When redivided, I.a. may have been one

                    if df.loc[idx,"structure_title"]!="" and isinstance(df.loc[idx,"structure_title"],str) and \
                        (df.loc[idx,"text_para"]!=df.loc[idx,"structure_title"]):
                        
                        df.loc[idx,"structure_title"]=df.loc[idx,"text_para"]
                        df.loc[idx,"full_structure"]=df.loc[idx,"text_para"]
                        df.loc[idx,"section_id"]=uuid.uuid4()
                        #if only title not declared but already the section
                        if isinstance(df.loc[idx,"title"],str)==False:
                            df.loc[idx,"title"]="yes"
                     
                    else:
                        df.loc[idx,"structure_title"]=df.loc[idx,"text_para"]
                        # only if different if not it was good before
                        if df.loc[idx,"full_structure"]!=df.loc[idx,"text_para"]:
                            if isinstance(df.loc[idx,"full_structure"],str):
                                df.loc[idx,"full_structure"]=(df.loc[idx,"full_structure"].rsplit('/', 1)[0]+"/"+df.loc[idx,"text_para"])
                            else:
                                df.loc[idx,"full_structure"]=df.loc[idx,"text_para"]
                            
                        df.loc[idx,"section_id"]=uuid.uuid4()
                        df.loc[idx,"title"]="yes"

                else:
                    # if not a title before, empty
                    if df.loc[idx,"title"]=="":
                        df.loc[idx,"structure_title"]=np.nan

        # Recheck from phenotype. if roman, add the previous roman before if not already
        if list(phenotype.type[phenotype.file_name==doc])[0].find("roman")!=-1:
            tmp=df.loc[(df.file_name==doc) & (df.full_structure.isnull()==False)]
            for idx in df.loc[(df.file_name==doc) & (df.full_structure.isnull()==False)].index:
                if df.loc[idx,"full_structure"].find("I.")==-1 and df.loc[idx,"full_structure"].find("V.")==-1 and df.loc[idx,"full_structure"].find("X.")==-1 and \
                    df.loc[idx,"full_structure"].find("EXECUTIVE SUMMARY")==-1 & df.loc[idx,"full_structure"].find("first_part_no_title")==-1:
                    roman=tmp.full_structure[tmp.index<idx]
                    if [item for item in roman[::-1] if (item.find("_/_")==-1 and (item.find("I.")!=-1 or item.find("V.")!=-1))]!=[]:
                        rom=next(item for item in roman[::-1] if (item.find("_/_")==-1 and (item.find("I.")!=-1 or item.find("V.")!=-1 or item.find("X.")!=-1)))
                        df.loc[idx,"full_structure"]=rom+"_/_"+df.loc[idx,"full_structure"]
        # Classify each paragraphe with a section
        idx_a=df.loc[(df.file_name==doc) & (df.full_structure.isnull())].index
        for idx in idx_a:
            df.loc[idx,"full_structure"]= \
                df.full_structure[df.structure_title[df.index<=idx].last_valid_index()]
            df.loc[idx,"section_id"]= \
                df.section_id[df.structure_title[df.index<=idx].last_valid_index()]
            df.loc[idx,"structure_title"]= \
                df.structure_title[df.structure_title[df.index<=idx].last_valid_index()]
            df.loc[idx,"title"]= "no"

    for idx in df.index:
        df.loc[idx,"structure_title"]=df.loc[idx,"structure_title"].replace("  "," ")
        df.loc[idx,"structure_title"]=df.loc[idx,"structure_title"].replace("Security Council/","")
        
    #df["text_lower"]=[item.lower() if isinstance(item,str) else item for item in df["text_para"]]
    df["para_id"]=[uuid.uuid4() for item in df.index]
    return df

#For 02_Categorization

def clean_text(df):
    # Add lower case and remove punctuation. Keep all for further tests
    df["text_lower"]=[item.lower() if isinstance(item,str) else item for item in df["text_para"]]
    df["text_tag"]=""
    # Keeping "_" for tag system "DPO_actor"
    exclude = set(string.punctuation)
    exclude.remove("_")
    exclude.remove("/")
    exclude.remove("-")
    exclude.add("—")
    exclude.add("–")
    exclude.add("●")
    exclude.add("”")
    exclude.add("“")
    exclude.add("’")
    exclude.add("‘")
    exclude.add("·")
    exclude.add("•")
    exclude.add("●")
    exclude.add("…")
    exclude.add("\uf0d8")

    for phrase_id in df.index:
        phrase_tag=df.text_lower[phrase_id]
        if isinstance(phrase_tag,str):
            word_tokens = word_tokenize(phrase_tag)
            filtered_sentence=[''.join(ch if ch not in exclude else " " for ch in x)  for x in word_tokens]
            df.loc[phrase_id,"text_tag"]=" "+(" ".join(filtered_sentence)).strip()+" "

    return df

def extract_actors(df_base,actors):
    # 
    df=df_base[['para_id','text_tag']]
    df["actors"]=""
    
    # Check if 2 same acronyms, list must be unique
    if len(actors["#org+acronym"].unique())!=len(actors) or len(actors["#org+name"].unique())!=len(actors):
        sys.exit("Errors Actors list, double acronym or names, please check")

    actor_writing={}
    actor_exception={}
    # Create table for counting elements
    df_tag=pd.DataFrame(columns=[["acteur","tag"]])
    
    for actor in actors.index:
        # Find all acronyms and names for the organisation
        names=actors.loc[actor,"Word_detection"].split(",")
        names=[x.strip() for x in names]
        names=[x for x in names if not x== ""]
        
        #Load acronym and remove it from the list to be singularized if ended by an s like "DOS", "UNMAS"...
        acronym=actors.loc[actor,"#org+acronym"].replace("_invented","")
        if re.search("S$",acronym)!=None:
            singular=[inflection.singularize(x) for x in names if x!=acronym]
            plural=[inflection.pluralize(x) for x in names if x!=acronym]
        else:
            singular=[inflection.singularize(x) for x in names]
            plural=[inflection.pluralize(x) for x in names]

        names=list(set(names+plural+singular))
        names.sort()
        names=all_writings(names,"actor")

        names_exc=actors.loc[actor,"Word_exception"]
        
        # Each word is between with spaces to ensure it is not inside a word) (but not just a word as it can be a combination)
        names = [" "+x.lower()+" " if isinstance(x, str) else x for x in names]
        # filter by length as the longer one found is more secure
        names.sort(key=lambda s: len(s))
        # Table to resue in the next loop
        actor_writing[actor]=names
        actor_exception[actor]=names_exc
          
    for phrase_id in df.index:
        phrase=df.text_tag[phrase_id]
        for actor in actors.index: 
            names=actor_writing[actor]
            names_exc=actor_exception[actor]
            
            if any(ext in phrase for ext in names):
                word_tokens = word_tokenize(phrase)
                tag=nltk.pos_tag(word_tokens)
                
                if isinstance(names_exc,str):
                    names_exc=names_exc.split(",")
                    names_exc=[x.strip() for x in names_exc]
                    if not any(ext2 in phrase for ext2 in names_exc):
                        df.loc[phrase_id,"actors"]=df.loc[phrase_id,"actors"]+","+actors.loc[actor,"#org+name"]
                        for ext in names:
                            if re.search(ext,phrase)!=None:
                                # for counting actors detected
                                tagged=[item for item in tag if item[0] == ext.strip()]
                                if tagged!=[]:
                                    a=pd.Series([ext,tagged[0][1]], index=df_tag.columns )
                                    df_tag=df_tag.append(a , ignore_index=True)
                                else:
                                    a=pd.Series([ext,tag], index=df_tag.columns )
                                    df_tag=df_tag.append(a , ignore_index=True)
                                # Update the column for later analysis by switching the word detected by a "tag"
                                df_base.loc[phrase_id,"text_tag"]=" "+re.sub(ext," "+actors.loc[actor,"#org+acronym"].strip()+"_actor ",df_base.loc[phrase_id,"text_tag"]).strip()+" "

                else:
                    for ext in names:
                        if re.search(ext,phrase)!=None:
                            # for counting actors detected
                            tagged=[item for item in tag if item[0] == ext.strip()]
                            if tagged!=[]:
                                a=pd.Series([ext,tagged[0][1]], index=df_tag.columns )
                                df_tag=df_tag.append(a , ignore_index=True)
                            else:
                                a=pd.Series([ext,tag], index=df_tag.columns )
                                df_tag=df_tag.append(a , ignore_index=True)
                            # Update the column for later analysis by switching the word detected by a "tag"
                            df_base.loc[phrase_id,"text_tag"]=" "+re.sub(ext," "+actors.loc[actor,"#org+acronym"].strip()+"_actor ",df_base.loc[phrase_id,"text_tag"]).strip()+" "
                    # Uptade the list of actors found
                    df.loc[phrase_id,"actors"]=df.loc[phrase_id,"actors"]+","+actors.loc[actor,"#org+name"]

        if (phrase_id % 100)==0:
            print("Loop "+str(round(phrase_id/len(df.index)*100))+" %")

    #Remove first "," 
    df.loc[df["actors"]!="","actors"]=(df.loc[df["actors"]!="","actors"]).str[1:]
 
    return df, df_base, df_tag

def all_writings(x,when_called):
    # Create different writings possibilities depending on the text
    # For now not all possibilities, just the clean one because it is going to be clean in paragraph
    from removeaccents import removeaccents
    new_list = []
    ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    for name in range(len(x)):
        x[name]=removeaccents.remove_accents(x[name])
        x[name]=x[name].replace("ç","c")
        x[name]=x[name].replace("'","")
        if "-" in x[name]:
            new_version=x[name].replace("-","")
            new_list.append(new_version)
            new_version=x[name].replace("-"," ")
            new_list.append(new_version)
        
        if "/" in x[name]:
            new_version=x[name].replace("/","-")
            new_list.append(new_version)
            new_version=x[name].replace("/","")
            new_list.append(new_version)
            new_version=x[name].replace("/"," ")
            new_list.append(new_version)
        if "–" in x[name]:
            new_version=x[name].replace("–","-")
            new_list.append(new_version)
            new_version=x[name].replace("–","")
            new_list.append(new_version)
            new_version=x[name].replace("–"," ")
            new_list.append(new_version)
            new_version=x[name].replace("–","/")
            new_list.append(new_version)
        # If it is a theme, we try to remove "of" or "to" from list
        if when_called=="theme":
            # "Protection of civilians"-> "Civilians protection"
            tmp_words=[x[name]]
            if " of " in x[name]:
                new_version=x[name].split("of")
                new_version=" ".join(new_version[::-1])
                new_version=new_version.strip()
                tmp_words.append(new_version)
                new_list.append(new_version)
            if " to " in x[name]:
                new_version=x[name].split("to")
                new_version=" ".join(new_version[::-1])
                new_version=new_version.strip()
                tmp_words.append(new_version)
                new_list.append(new_version) 

            # slow down a lot the algorithm, but improved performance. ML will be better
            # for mots in tmp_words:
            #     mots_token=word_tokenize(mots)
                
            #     for m in mots_token:
            #         word = Word(m)
            #         for syn in word.synsets[:10]:
            #             # if verb, add "ing"/"ed" as possibilities
            #             if syn._name.split(".")[1]=="v":
            #                 a=mots.replace(m,syn._name.split(".")[0])
            #                 new_list.append(a[-1]+"ing" if a[-1]=="e" else a+"ing")
            #                 new_list.append(a+"d" if a[-1]=="e" else a+"ed")
            #             # if noun, look if stem is a verb then add it and "ing"/"ed"
            #             if syn._name.split(".")[1]=="n":
            #                 u=Word(ps.stem(syn._name.split(".")[0])).synsets[:10]
            #                 if u!=[]:
            #                     tmp=[str(item) for item in list(u)]
            #                     tmp= [item for item in tmp if item.find(".v.")!=-1]
            #                     if tmp!=[]:
            #                         tmp= [item.split("'")[1] for item in tmp]
            #                         tmp= [item.split(".v.")[0] for item in tmp]
            #                         # Remove doubles
            #                         tmp=list(set(tmp))[0]
            #                         a=mots.replace(m,tmp)
            #                         new_list.append(a)
            #                         new_list.append(a[-1]+"ing" if a[-1]=="e" else a+"ing")
            #                         new_list.append(a+"d" if a[-1]=="e" else a+"ed")
            
        else:
            # Writing without spaces (not for theme as it is for proper words)
            if " " in x[name]:
                new_version=x[name].replace(" ","-")
                new_list.append(new_version)
                new_version=x[name].replace(" ","")
                new_list.append(new_version)

    new_list=x+new_list
    return new_list

def extract_dates(df):
    # Look for different formats of dates inside a paragraph
    # COULD BE ENHANCED by using NER see the test folder
   
    df["year"], df["month"], df["day"], df["date_para"], df["all_full_dates"]=["","","","",""]
    months=["january","february","march","april","may","june","july","august","september","october","november","december"]
    formats=['%d %B %Y', "%d %B, %Y", "%B %d, %Y","%d %B of %Y",'%B %Y', "%d %B", "%B %d", "%Y", "%B" ]
    for phrase_id in df.index:
        phrase=df.text_lower[phrase_id]
        struct_time=""
        # Look for all format until a date is found, then stop the loop
        for form in formats:
            for month in months:
                sub=re.search(form.replace("%d","\d{1,2}").replace("%Y","\d{4}").replace("%B",month),phrase)
                if sub:
                    struct_time = datetime.strptime(sub.group(0), form).date()
                    # if number in 4 digits is bigger than the current year (resolutions have bigger numbers)
                    # resolution will be found from another function
                    if struct_time.year > int(date.today().strftime("%Y")) or struct_time.year < 1990:
                        print(struct_time.year)
                    elif struct_time.year==1900 :
                        if struct_time.month==5 and struct_time.day==1:
                            # Only "may" found, so nothing
                            print(struct_time.year)
                        # no days found
                        elif struct_time.month!=5 and struct_time.day==1:
                            df.loc[phrase_id,"month"]=struct_time.month
                            df.loc[phrase_id,"date_para"]=str(struct_time.year)+"/"+str(struct_time.month)+"/"+str(struct_time.day)
                        else:
                        # month found but no year
                            df.loc[phrase_id,"month"]=struct_time.month
                            df.loc[phrase_id,"day"]=struct_time.day
                            df.loc[phrase_id,"date_para"]=str(struct_time.year)+"/"+str(struct_time.month)+"/"+str(struct_time.day)#
                    else:
                        # Even if day not found
                        df.loc[phrase_id,"year"]=struct_time.year
                        df.loc[phrase_id,"month"]=struct_time.month
                        df.loc[phrase_id,"day"]=struct_time.day
                        df.loc[phrase_id,"date_para"]=str(struct_time.year)+"/"+str(struct_time.month)+"/"+str(struct_time.day)
                        if form in ['%d %B %Y', "%d %B, %Y", "%B %d, %Y","%d %B of %Y"]:
                            df.loc[phrase_id,"all_full_dates"]=df.loc[phrase_id,"all_full_dates"]+","+str(struct_time.year)+"/"+str(struct_time.month)+"/"+str(struct_time.day)

    df.loc[df["all_full_dates"]!="","all_full_dates"]=(df.loc[df["all_full_dates"]!="","all_full_dates"]).str[1:]         
    return df

def extract_references(df):
    # Extract references found inside the files depending on various formats
    # format "S/2018/922""scr 2387"#"A/69/779", SC resolution, code cable 0467,st/ai/2010/4, security council resolution 2301
    df["ref"],df["ref_date"]=["",""]
    regex=["(s/\d{4}/\d{3})","(a/\d{2}/\d{3})","(a/res/\d{2}/\d{3})","(scr ?\d{4})","(sc resolution \d{4})","(s/res/\d{4})",
    "(security council resolution \d{4})","(code cable \d{4})","(st/ai/\d{4}/\d{1})"]

    for phrase_id in df.index:
        phrase=df.text_lower[phrase_id]
        for form in regex:
            p=re.search(form,phrase)
            if p!=None:
                ref=p.string[p.regs[0][0]:p.regs[0][1]]
                ref=ref.replace("security council resolution","scr")
                ref=ref.replace("sc resolution","scr")
                ref=ref.replace("s/res/","scr")
                ref=ref.replace("scr","scr ")
                ref=ref.replace("  "," ")
                df.loc[phrase_id,"ref"]=df.loc[phrase_id,"ref"]+","+ref

    df.loc[df["ref"]!="","ref"]=(df.loc[df["ref"]!="","ref"]).str[1:]
    return df

def extract_locations(df,admins):
    # Read all locations from reference file (source HDX)
    # Could add POI
    # Training NER would be more effective
    # Location = "Fo","Lim","Mom" are problematic

    # Extract only names / codes for adm 1,2,3
    admins = admins[["admin1Name,C,50","admin1Pcod,C,50","admin2Name,C,50","admin2Pcod,C,50","admin3Name,C,50","admin3Pcod,C,50","admin3RefN,C,50"]]
    region= (admins["admin1Pcod,C,50"]).unique()
    region.sort()
   
    df["Admin1_Code"],df["Admin1_Name"]=["",""]

    admin1={}
    admin1_name={}
    # Preparing list of names to check
    for adm in region:    
        names=admins.loc[(admins['admin1Pcod,C,50'] == adm)]
        adm_name=names["admin1Name,C,50"].unique()
        names = (names["admin1Name,C,50"].append(names["admin2Name,C,50"]).append(names["admin3Name,C,50"]).append(names["admin3RefN,C,50"])).unique()
        # Remove the one with short names like "Fo","Lim","Mom"
        names=[item for item in names if len(item)>3]
        names = all_writings(names,"location")
        names =[" "+x.lower()+" " if isinstance(x, str) else x for x in names]
        admin1[adm]=names
        admin1_name[adm]=adm_name[0]


    for phrase_id in df.index:
        phrase=df.text_tag[phrase_id]
        # Look for names and select admin 1 for now
        for adm in region:
            if any(ext in phrase for ext in admin1[adm]):
                word_tokens = word_tokenize(phrase)
                tag=nltk.pos_tag(word_tokens)
                df.loc[phrase_id,"Admin1_Name"]=df.loc[phrase_id,"Admin1_Name"]+","+admin1_name[adm]
                df.loc[phrase_id,"Admin1_Code"]=df.loc[phrase_id,"Admin1_Code"]+","+adm
                for ext in admin1[adm]:
                    if re.search(ext,phrase)!=None:
                        df.loc[phrase_id,"text_tag"]=" "+re.sub(ext," "+admin1_name[adm].strip()+"_loc ",df.loc[phrase_id,"text_tag"]).strip()+" "
    # Remove first ","
    df.loc[df.iloc[:,0]!="","Admin1_Name"]=(df.loc[df.iloc[:,0]!="","Admin1_Name"]).str[1:]
    df.loc[df.iloc[:,0]!="","Admin1_Code"]=(df.loc[df.iloc[:,0]!="","Admin1_Code"]).str[1:]
    return df

def prepare_language_tags(df):
    df['text_stem'],df['text_lemma'],df['text_lemma_tag']=["","",""]
    # Stemmatization and lemmatization for both plain text and tagged text

    for phrase_id in df.index:
        phrase=df.text_lower[phrase_id]
        phrase_tag=df.text_tag[phrase_id]
        if isinstance(phrase,str):
            word_tokens = word_tokenize(phrase)
            # Remove stopwords
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            # Remove numbers
            filtered_sentence=[w for w in filtered_sentence if w.isdigit()==False]
            # Remove title numerotation
            filtered_sentence=[w for w in filtered_sentence if re.search("^[i,v,x]{1,4}$",w)==None]
            filtered_sentence=[w for w in filtered_sentence if len(w)!=1]
            
            stem_sentence= [ps.stem(w) for w in filtered_sentence]
            df.loc[phrase_id,"text_stem"]=" "+(" ".join(stem_sentence)).strip()+" "
            lemma_sentence=[wnl.lemmatize(w) for w in filtered_sentence]
            df.loc[phrase_id,"text_lemma"]=" "+(" ".join(lemma_sentence)).strip()+" "
            word_tokens = word_tokenize(phrase_tag)

            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            filtered_sentence=[w for w in filtered_sentence if w.isdigit()==False]
            filtered_sentence=[w for w in filtered_sentence if re.search("^[i,v,x]{1,4}$",w)==None]
            filtered_sentence=[w for w in filtered_sentence if len(w)!=1]
            lemma_tag_sentence= [wnl.lemmatize(w) for w in filtered_sentence]
            df.loc[phrase_id,"text_lemma_tag"]=" "+(" ".join(lemma_tag_sentence)).strip()+" "

        else:
            df.loc[phrase_id,"text_stem"]=""
            df.loc[phrase_id,"text_lemma"]=""
            df.loc[phrase_id,"text_lemma_tag"]=""
    return df

def extract_themes(paragraphs,themes, actors_list):
    # Extract keywords from list give different weights.
    # first, second and third highest value and themes are extracted for qualitative analysis
    acronyms=set(actors_list["#org+name"])
    words={}
    paragraphs["themes"]=""

    paragraphs=paragraphs.loc[paragraphs.text_lemma.isnull()==False]
    paragraphs=paragraphs.reset_index()
    before=len(paragraphs.columns)
    for idx in themes.index:
        if isinstance(themes.loc[idx,"Actors_word"],str):
            absent=[w for w in themes.loc[idx,"Actors_word"].split(",") if w.strip() not in acronyms]
            if absent!=[]:
                print("Missing actor from theme list in actors list" + absent[0])        

        # lemmatize words from list
        for priority in ["Primary","Secondary","Tertiary"]:
            if isinstance(themes.loc[idx,priority+"_word"],str):
                keywords=themes.loc[idx,priority+"_word"].split(",")
                keywords=[w.strip() for w in keywords if w.strip()!=""]
                # Review if actors are all in actors_list
                keywords=all_writings(keywords,"theme")
                tmp=[item.strip().split(" ") for item in keywords]
                # Combinations to be made remove of
                tmp=[[wnl.lemmatize(w) for w in item] for item in tmp]
                tmp=[[w.lower() for w in item if not w in stop_words] for item in tmp]
                keywords=[" ".join(item) for item in tmp]
                keywords=[wnl.lemmatize(w.strip()) for w in keywords]
                keywords=[x for x in keywords if not x== ""]
                plural=[pluralizer.pluralize(x) for x in keywords]#
                plurals=[[pluralizer.pluralize(w) for w in item] for item in tmp]#
                plurals=[" ".join(item) for item in plurals]
                singular=[inflection.singularize(x) for x in keywords]
                singulars=[[inflection.singularize(w) for w in item] for item in tmp]
                singulars=[" ".join(item) for item in singulars]
                keywords=list(set(keywords+plural+singular+plurals+singulars))
                keywords.sort()
                keywords=[" "+x+" " if isinstance(x, str) else x for x in keywords]
                words[themes.loc[idx,"theme_id"]+"_"+priority]=keywords


    df=pd.DataFrame(columns=["para_id","theme_id","priority","number"])
    themes_flat=pd.DataFrame(columns=["para_id","theme_id","score"])
    #add columns for all theme
    for th in themes["Theme"]: 
        paragraphs[th]=0

    col=range(before,len(paragraphs.columns))
    paragraphs["first_theme"],paragraphs["first_value"],paragraphs["second_theme"],paragraphs["second_value"]=["","","",""]
    paragraphs["third_theme"],paragraphs["third_value"]=["",""]
    
    # 3 columns (1,2,3 for each theme)
    for idx in paragraphs.index:
    # for each sentence, look for number of primary words, secondary and other
        for th in themes.index:
            score=0
            # Look for actors
            if (isinstance(themes.loc[th,"Actors_word"],str)) and (isinstance(paragraphs.loc[idx,"actors"],str)) :
                aw=themes.loc[th,"Actors_word"].split(",")
                # check if "," in the end of list
                aw=[item.strip() for item in aw if item!=" "]
                # if actor is in the list
                if any(ext in paragraphs.loc[idx,"actors"].split(",") for ext in aw):
                    num=[1 if ext in paragraphs.loc[idx,"actors"].split(",") else 0 for ext in aw]
                    coeff=0.25
                    score=score+sum(num)*coeff
            # Look for actors category
            if (isinstance(themes.loc[th,"Actor_subcategory_word"],str)) and (isinstance(paragraphs.loc[idx,"actors"],str)) :
                # find back the actor category
                ac=paragraphs.loc[idx,"actors"].split(",")
                tac=actors_list[actors_list["#org+name"].isin(ac)]
                tac=list(set(tac["Sub Category"]))
                tac=[item for item in tac if isinstance(item,str)]
                tac=[item.strip() for item in tac if item!=" "]
                # if actor category is in the list
                if any(ext in tac for ext in themes.loc[th,"Actor_subcategory_word"].split(",")):
                    num=[1 if ext in tac else 0 for ext in themes.loc[th,"Actor_subcategory_word"].split(",")]
                    coeff=0.05
                    score=score+sum(num)*coeff
            for priority in ["Primary","Secondary","Tertiary"]:#,"Actors","Actor_subcategory"]:
                # if words were associated before
                if themes.loc[th,"theme_id"]+"_"+priority in words.keys():
                    coeff= 0.37 if priority=="Primary" else 0.13 if priority=="Secondary" else 0.05 if priority=="Tertiary" else 0.09 if priority=="Actor_subcategory" else 0.25 if priority=="Actors" else 0.05
                    #if find keywords 
                    if any(ext in paragraphs.loc[idx,"text_lemma"] for ext in words[themes.loc[th,"theme_id"]+"_"+priority] if isinstance(ext,str)):
                        num=[1 if ext in paragraphs.loc[idx,"text_lemma"] else 0 for ext in words[themes.loc[th,"theme_id"]+"_"+priority]]
                        score=score+(sum(num)*coeff)
                        a=pd.Series([paragraphs.loc[idx,"para_id"],themes.loc[th,"theme_id"],priority, sum(num)], index=df.columns )
                        df=df.append(a , ignore_index=True)

            # for each theme give the score
            paragraphs.loc[idx,themes.loc[th,"Theme"]]=score
            if score>0:
                score= 1 if score > 1 else score
                b=pd.Series([paragraphs.loc[idx,"para_id"],themes.loc[th,"theme_id"],score], index=themes_flat.columns )
                themes_flat=themes_flat.append(b , ignore_index=True)
            if score>0.35:
                paragraphs.loc[idx,"themes"]=paragraphs.loc[idx,"themes"]+","+themes.loc[th,"Theme"]

        # Keep first 3
        all_coeff=sorted(zip( paragraphs.iloc[idx,col],paragraphs.columns[col]), reverse=True)
        paragraphs.loc[idx,"first_theme"]=all_coeff[0][1]
        paragraphs.loc[idx,"first_value"]=100 if all_coeff[0][0] > 1 else round(all_coeff[0][0],2)*100
        paragraphs.loc[idx,"second_theme"]=all_coeff[1][1]
        paragraphs.loc[idx,"second_value"]=100 if all_coeff[1][0] > 1 else round(all_coeff[1][0],2)*100
        paragraphs.loc[idx,"third_theme"]=all_coeff[2][1]
        paragraphs.loc[idx,"third_value"]=100 if all_coeff[2][0] > 1 else round(all_coeff[2][0],2)*100

        if (idx % 100)==1:
            print("Loop "+str(round(idx/len(paragraphs)*100))+" %")
    paragraphs.loc[paragraphs["themes"]!="","themes"]=(paragraphs.loc[paragraphs["themes"]!="","themes"]).str[1:]
    return paragraphs, themes_flat

def extract_tags(paragraphs,tags, actors_list):

    acronyms=set(actors_list["#org+name"])
    words={}
    stems={}
    paragraphs=paragraphs.loc[paragraphs.text_lemma.isnull()==False]
    paragraphs=paragraphs.reset_index()
    before=len(paragraphs.columns)
    for idx in tags.index:
        # lemmatize words

        priority="Primary_words"
        if isinstance(tags.loc[idx,priority],str):
            keywords=tags.loc[idx,priority].split(",")
            keywords=[w.strip() for w in keywords if w.strip()!=""]
            tmp=[item.strip().split(" ") for item in keywords]
            tmp=[[w.lower() for w in item if not w in stop_words] for item in tmp]
            tmp=[[wnl.lemmatize(w) for w in item] for item in tmp]
            tmp=[(" ".join(w)).strip() for w in tmp]
            tmp=[" "+item+" " for item in tmp]

            stems[tags.loc[idx,"Tag_id"]]=tmp

    paragraphs["tags"]=""

    for idx in paragraphs.index:
        # for each sentence, look for actors specified, if not, look for words
        for th in tags.index:
            act=False
            if (isinstance(tags.loc[th,"Actors"],str)) and (isinstance(paragraphs.loc[idx,"actors"],str)) :
                if any(ext in paragraphs.loc[idx,"actors"].split(",") for ext in tags.loc[th,"Actors"]):
                    paragraphs.loc[idx,"tags"]=paragraphs.loc[idx,"tags"]+","+tags.loc[th,"Tag_id"]
                    act=True

            # No actors found, look for words
            if act==False:
                #if find keywords 
                if any(ext in paragraphs.loc[idx,"text_lemma"] for ext in stems[tags.loc[th,"Tag_id"]] if isinstance(ext,str)):
                    paragraphs.loc[idx,"tags"]=paragraphs.loc[idx,"tags"]+","+tags.loc[th,"Level 2"]

        if (idx % 100)==1:
            print("Loop "+str(round(idx/len(paragraphs)*100))+" %")
    paragraphs.loc[paragraphs["tags"]!="","tags"]=(paragraphs.loc[paragraphs["tags"]!="","tags"]).str[1:]

    return paragraphs

#For 03_Classification

def extract_recommendations_elements(df2,phenotype,actors):

    # list of actors to review for specific recommendations
    actors=actors[(actors.Category=="MINUSCA") | ( actors.Category=="UN_Secretariat")]
    actors=set(actors["#org+acronym"])

    # all elements considered
    df=pd.DataFrame(columns=["para_id","text_lower","text_tag","full_structure","file_name","title","modal0","modal1","must_has_to","may_might", \
    "can","could","will","would","toverb","actors_toverb", "ought_to", "should_shall", "expression","stronger_words","only_recommend", "weaker_words","last_sentence", \
        "title_word","nb_sentences","explicit","MD","TO","JJ","NN","PR","RB","WH","VB","various_part1","various_part2"])
    df[["para_id","text_lower","text_tag","full_structure","file_name"]]= df2[["para_id","text_lower","text_tag","full_structure","file_name"]]
    df["title"]=[1 if item=="yes" else 0 for item in df2["title"]]
    df.full_structure=df.full_structure.fillna(" ")
    df=df.fillna(0)
    df.full_structure.fillna(" ")
    exclude=["(",")",",","."]
    for phrase_id in df.index:
        phrase=df.text_lower[phrase_id]
        pheno=phenotype.loc[phenotype.file_name==df.file_name[phrase_id],"type"]
        if isinstance(phrase,str):
            word_tokens = word_tokenize(phrase)
            sentence=sent_tokenize(phrase)
            tag=nltk.pos_tag(word_tokens)
            tag=[(w,pos_tag) for w,pos_tag in tag if pos_tag not in exclude]
            pos_tags = [pos_tag for _,pos_tag in tag]
            w_tags = [w for w,_ in tag]
            fd = nltk.FreqDist(pos_tags)
            # Number of modals
            if fd["MD"]>0:
                df.loc[phrase_id,"modal0"]=1
            if fd["MD"]/len(sentence)>=1:
                df.loc[phrase_id,"modal1"]=1
            # look for "actor+to+verb" in the sentence
            if fd["TO"]>0 and fd["VB"]>0:
                # only the first one
                if pos_tags.index('VB')-pos_tags.index('TO')==1:
                    phrase_tag=df.text_tag[phrase_id]
                    word_tokens = word_tokenize(phrase_tag)
                    tag2=nltk.pos_tag(word_tokens)
                    tag2=[(w,pos_tag) for w,pos_tag in tag2 if pos_tag not in exclude]
                    pos_tags2 = [pos_tag for _,pos_tag in tag2]
                    w_tags2 = [w for w,_ in tag2]
                    if (w_tags2[pos_tags2.index('TO')-1].find("actor")!=-1):
                        df.loc[phrase_id,"actors_toverb"]=1
                        
            # "to+verb" in the first 5 words need to remove parenthesis, comma
            
            begin_tag=[item for item in pos_tags if item not in exclude][:5]
            fd_deb = nltk.FreqDist(begin_tag)
            if fd_deb["TO"]>0 and fd_deb["VB"]>0:
                # following
                if begin_tag.index('VB')-begin_tag.index('TO')==1:
                    df.loc[phrase_id,"toverb"]=1
           
            # Last sentence number of modals (for long paragraphs)             
            if len(sentence)>1:
                last_sentence=sentence[-1]
                word_tokens = word_tokenize( last_sentence)
                tag=nltk.pos_tag(word_tokens)
                pos_tags = [pos_tag for _,pos_tag in tag]
                fd = nltk.FreqDist(pos_tags)
                df.loc[phrase_id,"last_sentence"]=fd["MD"]
            
            # Decomposition not used in the model
            if fd["MD"]>0:
                df.loc[phrase_id,"MD"]=fd["MD"]/len(sentence)
            if fd["TO"]>0:
                df.loc[phrase_id,"TO"]=fd["TO"]/len(sentence)
            if fd["JJ"]+fd["JJR"]+fd["JJS"]>0:
                df.loc[phrase_id,"JJ"]=(fd["JJ"]+fd["JJR"]+fd["JJS"])/len(sentence)
            if fd["NN"]+fd["NNP"]+fd["NNS"]+fd["NNPS"]>0:
                df.loc[phrase_id,"NN"]=(fd["NN"]+fd["NNP"]+fd["NNS"]+fd["NNPS"])/len(sentence)
            if fd["PRP"]+fd["PRP$"]+fd["POS"]>0:
                df.loc[phrase_id,"PR"]=(fd["PRP"]+fd["PRP$"]+fd["POS"])/len(sentence)
            if fd["RB"]+fd["RBR"]+fd["RBS"]+fd["WRB"]>0:
                df.loc[phrase_id,"RB"]=(fd["RB"]+fd["RBR"]+fd["RBS"]+fd["WRB"])/len(sentence)
            if fd["WDT"]+fd["WP"]+fd["WPS"]>0:
                df.loc[phrase_id,"WH"]=(fd["WDT"]+fd["WP"]+fd["WPS"])/len(sentence)
            if fd["VB"]+fd["VBD"]+fd["VBG"]+fd["VBZ"]+fd["VBN"]+fd["VBP"]>0:
                df.loc[phrase_id,"VB"]=(fd["VB"]+fd["VBD"]+fd["VBG"]+fd["VBZ"]+fd["VBN"]+fd["VBP"])/len(sentence)
            if fd["CC"]+fd["CD"]+fd["DT"]+fd["EX"]+fd["IN"]>0:
                df.loc[phrase_id,"various_part1"]=(fd["CC"]+fd["CD"]+fd["DT"]+fd["EX"]+fd["IN"])/len(sentence)
            if fd["PDT"]+fd["RP"]+fd["UΗ"]>0:
                df.loc[phrase_id,"various_part2"]=(fd["PDT"]+fd["RP"]+fd["UΗ"])/len(sentence)

            # Recommendation in the structure
            if re.search("recommendation",df.full_structure[phrase_id].lower())!=None:
                df.loc[phrase_id,"title_word"]=1
            df.loc[phrase_id,"nb_sentences"]=len(sentence) 
            # Explicitness
            if list(pheno)[0].find("unfound")!=-1 and df.loc[phrase_id,"explicit"]!=1:
                df.loc[phrase_id,"explicit"]=1 if phrase.find("!-!")!=-1 else 0
            if list(pheno)[0].find("letter_low_notitle")!=-1 and df.loc[phrase_id,"explicit"]!=1:
                df.loc[phrase_id,"explicit"]=1 if re.search("[a-z]{1}\.",phrase[:10])!=None else 0
            if list(pheno)[0].find("letter_par")!=-1 and df.loc[phrase_id,"explicit"]!=1:
                df.loc[phrase_id,"explicit"]=1 if re.search("\([a-z]\)",phrase[:10])!=None else 0
            if df.loc[phrase_id,"explicit"]!=1:
                df.loc[phrase_id,"explicit"]=1 if (re.search("•",phrase)!=None or re.search("●",phrase)!=None or re.search("!-!",phrase)!=None) else 0
            if df.loc[phrase_id,"title_word"]==1:
                df.loc[phrase_id,"explicit"]=1

            if re.search("must",phrase.lower())!=None or re.search("has to",phrase.lower())!=None or re.search("have to",phrase.lower())!=None:
                df.loc[phrase_id,"must_has_to"]=1    
            if re.search("may",phrase.lower())!=None or re.search("might",phrase.lower())!=None :
                df.loc[phrase_id,"may_might"]=1
            if re.search("can",phrase.lower())!=None :
                df.loc[phrase_id,"can"]=1
            if re.search("could",phrase.lower())!=None :
                df.loc[phrase_id,"could"]=1
            if re.search("will",phrase.lower())!=None :
                df.loc[phrase_id,"will"]=1
            if re.search("would",phrase.lower())!=None :
                df.loc[phrase_id,"would"]=1
            if re.search("ought to",phrase.lower())!=None :
                df.loc[phrase_id,"ought_to"]=1
            if re.search("should",phrase.lower())!=None or re.search("shall",phrase.lower())!=None :
                df.loc[phrase_id,"should_shall"]=1
            if  re.search("not yet",phrase.lower())!=None or \
                re.search("there is a need",phrase.lower())!=None:
                df.loc[phrase_id,"expression"]=1
            if re.search("require",phrase.lower())!=None or re.search("suggest",phrase.lower())!=None or \
                (re.search("recommend",phrase.lower())!=None and len(phrase.strip())<20) or \
                (re.search("need",phrase.lower())!=None and re.search("there is a need",phrase.lower())==None):
                df.loc[phrase_id,"stronger_words"]=1
            if re.search("recommendation",phrase.lower())!=None and len(phrase.strip())<25:
                df.loc[phrase_id,"only_recommend"]=1
            if re.search("welcome",phrase.lower())!=None or re.search("acknowledg",phrase.lower())!=None \
                or re.search("propose",phrase.lower())!=None  or \
                ((re.search("urge",phrase.lower())!=None or re.search("urgent",phrase.lower())!=None) and \
                (re.search("urgent temporary measures",phrase.lower())==None or re.search("surge",phrase.lower())==None)):
                    df.loc[phrase_id,"weaker_words"]=1

    return df

def calculate_recommendations(quant,qual,folder_out):
   
    # Merge the 2 filrs
    df=quant.merge(qual,how="right",on="para_id")
    # Filter only for lines that have been qualitatively validated
    df=df.loc[df.KP_recommendation.isnull()==False]
    
    df["kp"]=[1 if item=="yes" else 0 for item in df.KP_recommendation]
    df["full_structure"]=[item if isinstance(item,str) else "" for item in df.full_structure]
    col=["modal0","must_has_to","may_might", \
    "can","could","will","would", "ought_to", "should_shall", "expression","stronger_words",  \
        "weaker_words","last_sentence","title","explicit","toverb","actors_toverb"]
    X_train, X_test = train_test_split(df,test_size=0.2, random_state = 0)
    # Only the significative features have been incorporated in the sytem
    model = sm.formula.glm("kp ~ modal0+must_has_to+may_might+can+could+should_shall+expression+weaker_words+stronger_words+title+explicit+toverb+actors_toverb",
                       family=sm.families.Binomial(), data=X_train).fit()

    y_test=X_test["kp"]
    # difference between train and test
    df["predict"]=0
    y_v=model.predict(X_train)
    y_pred = model.predict(X_test)

    # Add to main table to review
    comp=0
    for i in X_test.index:
        df.loc[i,"predict_linear"]=y_pred[i]
        df.loc[i,"predict"]=1
        comp+=1

    comp=0
    for i in X_train.index:
        df.loc[i,"predict_linear"]=y_v[i]
        comp+=1

    # Comparation if only boolean answer is used : 1 if >0.5, else 0
    df["cat_recom"]=[round(item) for item in df.predict_linear]
    df["valid_recom"]=df["cat_recom"]-df["kp"]
    # Create 4 categories, easier to interprate results and buid confidence
    df["cat_recom"]=["Yes" if item>=0.75  else "Very Likely" if item>=0.5  else "Unlikely" if item>=0.25 else "No" for item in df.predict_linear]

    model_filename = folder_out+"linear_model.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    return model_filename,col

def apply_model_recommendation(df,col,recommendations_model):
 
    # Read model
    with open(recommendations_model, 'rb') as file:
        model = pickle.load(file)
    x_app = df[col]
    y_pred_app = model.predict(x_app)
    df["score"]=y_pred_app
    df["recommendation"]=["Yes" if item>=0.75  else "Very Likely" if item>=0.5  else "Unlikely" if item>=0.25 else "No" for item in df.score]

    return df

# For 04_Similarities

def link_recommandations(df,tags):
    # For all explicit recommendations, look for all recommendations that shares at least 1 theme or 1 tag
    df=df.loc[(df["recommendation"]=="Yes") | (df["recommendation"]=="Very Likely")]
    keep_explicit=df.loc[(df["explicit"]==1)].index
    df["keep"]=0
    # All the themes
    cols=["Politics","Rule of Law and Justice","Human rights","Protection of civilians","Women peace and security", \
            "Disarmament, Demobilization and Reintegration","Arms control","Climate and natural resources", \
                "Security sector reform","Restoration of State authority","Performance and accountability", \
                    "Safety and security","Elections","Humanitarian assistance","Child protection","Gender based violence", \
                        "Support to security institutions"]
    df["tags"]= [item if isinstance(item,str)  else " " for item in df["tags"]]               
    keep={}
    # For each theme
    for col in cols:
        df["keep"]=0
        df["keep"]=[1 if item > 0.35 else 0 for item in df[col]]
        list_index=list(set(list(df[df["keep"]==1].index)))
        res = [(a, b) for idx, a in enumerate(list_index) for b in list_index[idx + 1:]] 
        keep[col]=res

    # For each tag
    for tag in tags:
        df["keep"]=0
        df["keep"]=[1 if (item.find(tag)!=-1)  else 0 for item in df["tags"]]
        list_index=list(set(list(df[df["keep"]==1].index)))
        res = [(a, b) for idx, a in enumerate(list_index) for b in list_index[idx + 1:]] 
        keep[tag]=res

    all_pairs=[]
    for k in keep.keys():
        all_pairs=all_pairs+keep[k]

    # Remove doubles
    all_pairs=list(set(all_pairs))
    inversed = [(item[1],item[0]) for item in all_pairs]
    all_pairs=all_pairs+inversed
    all_pairs=list(set(all_pairs))
    all_pairs=[(item[0],item[1]) for item in all_pairs if item[0] in keep_explicit]
    # add a pair with the same elements (from explicit) for extraction purposes
    add_own=[(item,item) for item in keep_explicit]
    all_pairs=all_pairs+add_own
    
    return pd.DataFrame(all_pairs)

def lemma_wmd(df,liste_pairs,model_filename):
    
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    model.init_sims(replace=True)
    # Initilalization
    liste_pairs["wmd"]=1000
    liste_pairs["Similarity"]=1000
    for idx in liste_pairs.index:
        # find back the lemmatized tagged version
        a=df.loc[df.index==liste_pairs.loc[idx,"idx1"],"text_lemma_tag"][liste_pairs.loc[idx,"idx1"]]
        b=df.loc[df.index==liste_pairs.loc[idx,"idx2"],"text_lemma_tag"][liste_pairs.loc[idx,"idx2"]]
        
        # Distance
        liste_pairs.loc[idx,"wmd"]=round(model.wmdistance(a,b)*100,2)

        # Transform to probability
        liste_pairs.loc[i,"Similarity"]=math.floor(1000*math.exp(-liste_pairs.loc[idx,"wmd"]/3))/1000

    # Filter to reduce size threshold is arbitrary
    threshold=8
    liste_pairs=liste_pairs[liste_pairs["wmd"]<threshold]    
    liste_pairs=liste_pairs[["idx1","idx2","Similarity"]]

    return pd.DataFrame(liste_pairs)            


def reduce_path(full_path):
    file_name = full_path.rsplit('\\', 1)[-1].replace(" ","_").replace(".","_")
    return file_name
