import pandas as pd

df = pd.DataFrame({
    "gender":["M","F","M","F"],
    "status":["Single","Married","Single","Divorced"]
})
mapping = {
    "M":0,
    "F":1,
    "Single":2,
    "Married":3,
    "Divorced":4
}
df = df.replace(mapping)
print(df)