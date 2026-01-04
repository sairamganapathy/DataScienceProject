import pandas as pd
##direct loading of dataframe
# df = pd.DataFrame({"color":["red","blue","green","yellow","purple"]})
# mapping = {"red":1, "blue":2, "green":3, "yellow":4, "purple":5}
# df["color_numeric"] = df["color"].map(mapping)
# print(df)
##creating an object and loading a dataframe
dict_data = {"color":["red","blue","green","yellow","purple"]}
df = pd.DataFrame(dict_data)
mapping = {"red":1, "blue":2, "green":3, "yellow":4, "purple":5}
df["color_numeric"] = df["color"].map(mapping)
print(df)