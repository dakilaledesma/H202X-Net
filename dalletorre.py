import pandas as pd

dalle_file = open("data/DalleTorre.txt", encoding="latin-1")
dalle_lines = dalle_file.readlines()

dalle_list = [dalle_lines[0].replace('"', '').strip().split("\t")]
for line in dalle_lines[1:]:
    line = line.replace('"', '').strip().split("\t")

    if line[0] not in ["Hepatic", "Moss", "Lichen", "Algae"]:
        dalle_list.append(line)

df = pd.DataFrame(dalle_list)
df.columns = df.iloc[0]
df = df[1:]

df = df.sort_values(by=["Genus_No"])

df.to_csv("output/dax_dalletorre.csv", index=False)
