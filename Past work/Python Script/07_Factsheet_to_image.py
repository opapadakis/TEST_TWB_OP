# Transform the factsheet from PDF to images and create reference link
import fitz
import glob
import pandas as pd
import numpy as np
# Images processing
from PIL import Image

# Repository with Factsheets should connect to sharepoint
# https://unitednations.sharepoint.com/sites/DPPA_DPO-ODCSS-IMU/DPPADPO%20Data%20Reports/Forms/AllItems.aspx?csf=1&e=csaXoJ&cid=b02c736c%2D399e%2D4761%2D93f0%2D41a9e3523997&RootFolder=%2Fsites%2FDPPA%5FDPO%2DODCSS%2DIMU%2FDPPADPO%20Data%20Reports%2FMission%20Fact%20Sheets&FolderCTID=0x0120007C9923321A4BA047B472935B9E0899A5
folder_in="../../../Data/Internal_Data/Factsheets/"

# Read subfolder
pdf_files=glob.glob(folder_in+"**/MINUSCA*.pdf", recursive=True)
# Filter all files with "minusca" in name
pdf_files=[item for item in pdf_files if item.lower().find("minusca")!=-1]
df=pd.DataFrame(columns=["file_name","date","personnal_onedrive","xb_onedrive"])
months=["january","february","march","april","may","june","july","august","september","october","november","december"]
months_3l=[item[:3] for item in months]
for doc in pdf_files:
    doc_name_short=doc.rsplit('\\', 1)[-1]
    doc_name_image=doc_name_short.replace(".pdf",".png")
    month_year=doc_name_short.split("_")[2]
    month=months_3l.index(month_year[:3].lower())+1
    year=int(month_year[3:].replace(".pdf",""))
    if month<10:
        date_fin=str(year)+"-0"+str(month)+"-"+"01"
    else:
        date_fin=str(year)+"-"+str(month)+"-"+"01"
    web_adress="https://unitednations-my.sharepoint.com/:i:/r/personal/olivier_papadakis_un_org/Documents/Factsheets_MINUSCA_only/"+doc_name_image
    xb_adress=""
    a=pd.Series([doc_name_image, date_fin, web_adress, xb_adress], index=df.columns )
    df=df.append(a , ignore_index=True)

df.to_csv(folder_in+"Factsheets.csv", sep =";")
