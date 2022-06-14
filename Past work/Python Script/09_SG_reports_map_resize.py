# Resize images already in the folder for visualization
# They are extracted in the images_pages function in Images BUT NOT classified directly
# Put manually in the folder

from PIL import Image 
import glob
from Functions import reduce_path
# Dataframe
import pandas as pd
import numpy as np
import csv

folder_in="../../../Data/Reports/SG_maps/"
images=glob.glob("*.png")
newsize = (1541,1176) 

sg_map=pd.DataFrame(columns=["filename","pc_address","share_address","xb_address"])
pc_dep=images[0].rsplit("\\",1)[0]+"/formatted/"
share_dep="https://unitednations-my.sharepoint.com/:i:/r/personal/olivier_papadakis_un_org/Documents/Images/SG_report_maps/"
xb_dep="https://unitednations.sharepoint.com/:i:/r/sites/DPPA_DPO-PROJ-CARIOT-MINUSCAperformance/Shared%20Documents/Reports_for_Analysis/Images/SG_report_maps/"
for image in images:
    filename=image.rsplit("\\",1)[1].split(".")[0]+".png"
    im = Image.open(image)  
    width, height = im.size 
    if height>1.1*width:
        im = im.rotate(90,expand=True)

    out = im.resize(newsize) 
    out.save(filename)
    a=pd.Series([filename,pc_dep+filename,share_dep+filename,xb_dep+filename], index=sg_map.columns )
    sg_map=sg_map.append(a , ignore_index=True)


sg_map.to_csv(folder_in+"SG_maps.csv",sep=";",index=False,encoding="UTF-8")
