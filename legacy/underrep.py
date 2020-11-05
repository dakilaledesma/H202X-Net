import pandas as pd
from bs4 import BeautifulSoup
import requests
from zipfile import ZipFile
import os
from tqdm import tqdm

df = pd.read_csv("data/full_train_data.csv")
sp_names = df["category_name"]
filenames = df["file_name"]


name_counts = {}
name_fp = {}

for idx, name in enumerate(sp_names):
    try:
        name_counts[name] += 1
    except:
        name_counts[name] = 0

    if name not in name_fp.keys():
        filepath = f"{'/'.join(str(filenames.loc[idx]).split('/')[:3])}/"
        name_fp[name] = filepath

name_counts = {k: v for k, v in sorted(name_counts.items(), key=lambda item: item[1])}

names = list(name_counts.keys())
counts = list(name_counts.values())

for i in range(len(counts)):
    if counts[i] == 10:
        names = names[:i]
        counts = counts[:i]
        break

name_counts = {k: v for k, v in zip(names, counts)}

# Get DWCs
#
# page = requests.get("https://sernecportal.org/portal/collections/datasets/datapublisher.php")
# soup = BeautifulSoup(page.text, 'html.parser')
# links = str(soup).split('<a href="')
#
# archives = [f"{link.split('.zip')[0]}.zip" for link in links if ".zip" in link]
#
# for link in archives:
#     archive_name = link.split('/')[-1].replace('.zip', '')
#     os.mkdir(f"data/dwc_a/{archive_name}")
#     r = requests.get(link)
#
#     with open(f"data/dwc_a/{archive_name}.zip", 'wb') as f:
#         f.write(r.content)
#
#     with ZipFile(f"data/dwc_a/{archive_name}.zip", 'r') as zip_file:
#         zip_file.extractall(path=f"data/dwc_a/{archive_name}")

dwc_folders = [x[0].replace("\\", '/') for x in os.walk("data\\dwc_a") if x[0] != 'data\\dwc_a']

occurences = pd.DataFrame()
images = pd.DataFrame()

sp_image_urls = {}
for folder in tqdm(dwc_folders):
    try:
        dwc_occurrences = pd.read_csv(f"{folder}/occurrences.csv", index_col=0, low_memory=False)
        dwc_images = pd.read_csv(f"{folder}/images.csv", low_memory=False)
    except FileNotFoundError:
        continue

    for index in range(dwc_images["coreid"].size):
        try:
            scientific_name = dwc_occurrences.loc[dwc_images["coreid"].iloc[index]]["scientificName"]
        except KeyError:
            continue

        if "." in str(scientific_name):
            split = scientific_name.split(" ")
            for idx, word in enumerate(split):
                if "." in word:
                    split = split[:idx]
                    break

            scientific_name = ' '.join(split)
        try:
            sp_image_urls[scientific_name].append(dwc_images["accessURI"].iloc[index])
        except:
            sp_image_urls[scientific_name] = [dwc_images["accessURI"].iloc[index]]

urls_file = open("data/sp_image_urls.txt", 'w', encoding="latin-1")
csv_string = ""
tab_char = "\t"
for key in sp_image_urls.keys():
    csv_string += f"{key}\t{tab_char.join(sp_image_urls[key])}\n"
urls_file.write(csv_string)
urls_file.close()
