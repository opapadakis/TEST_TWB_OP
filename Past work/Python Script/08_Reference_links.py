# Collect all information 

import fitz
import glob
import pandas as pd
import numpy as np
import requests
import wget
from Functions import *
from temp_test import *


# Need to separate in files categories.

pdf_files=glob.glob("../../../Data/Reports/SC_report/*.pdf")
downloaded=glob.glob("../../../Data/Reports/Downloaded_Uncategorized/*")
downloaded=[item.rsplit('\\', 1)[-1].replace("%20"," ") for item in downloaded]
links = pd.DataFrame(columns=["Resolution","Link","Document","Page","Downloaded"])

for doc in pdf_files:
    doc_open = fitz.open(doc) 
    for page in doc_open:
        # Extract links
        info = page.getLinks()
        if len(info)>0:
            for elem in info: 
                #The PHP links are not the final ones, so we change it to fit the doc
                if elem["uri"][-3:]=="php":
                    new_url=elem["uri"].rsplit('/', 1)[-1].rsplit(".",1)[0]
                    # if SC press release
                    if new_url[:2]=="sc":
                        new_url=new_url.replace("-","")
                        elem["uri"]="https://www.un.org/press/en/2020/" + new_url +".doc.htm"
                    else :
                        new_url=new_url.replace("-","_")
                        elem["uri"]="https://www.securitycouncilreport.org/atf/cf/%7B65BFCF9B-6D27-4E9C-8CD3-CF6E4FF96FF9%7D/"+ new_url+ ".pdf"
                # One link on another website
                elif elem["uri"][-1:]=="/":
                        elem["uri"]=elem["uri"][:-1]+".html"

                links_to_add=[elem["uri"].rsplit('/')[-1].replace("%20"," "),elem["uri"],reduce_path(doc),page.number+1,""]
                links= links.append(dict(zip(links.columns,links_to_add)), ignore_index=True)


column_names=links.columns
np.savetxt('../../../Processing/Out/Explore/links_sc.csv', links,
delimiter=';', header=";".join(column_names), fmt='%s', encoding='UTF-8')

listing=links["Resolution"].unique()
listing=listing["Link"!="http://daccess-dds-ny.un.org/doc/UNDOC/GEN/N09/661/45/PDF/N0966145.pdf?OpenElement"]

for file_name in links["Resolution"].unique():
    if file_name in downloaded:
        links.loc[links["Resolution"]==file_name,"Downloaded"]="Yes"
    else:
        url = str(links.loc[links["Resolution"]==file_name,"Link"].values[0])
        # using header for allowing access
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        #if r.reason=="OK":
            with open("../../../Data/Reports/Downloaded_Uncategorized/"+file_name, 'wb') as f:
                f.write(r.content) 
            links.loc[links["Resolution"]==file_name,"Downloaded"]="Yes"
        #else: 
        except requests.exceptions.RequestException as e:
            continue
            #links.loc[links["Resolution"]==file_name,"Downloaded"]=r.reason
    print(file_name)

np.savetxt('../../../Processing/Out/Explore/links_sc.csv', links,
delimiter=';', header=";".join(column_names), fmt='%s', encoding='UTF-8')

