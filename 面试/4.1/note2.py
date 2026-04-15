# import re

# line = "[2024-01-01 12:00:00] INFO User123 Login success"
# pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (\w+) (\w+) (.+)'
# match = re.match(pattern, line)
# print(match.group(0))
# print(match.groups())


import pandas as pd

df = pd.Series([1,2,3])
dic = {1:"a",2:"b"}
print(df.map(dic))

df = pd.DataFrame([
    {"a":"1a","b":"2b","c":"3c"},
    {"a":"1a","b":"5b","c":"6c"},
    {"a":"2a","b":"8b","c":"9c"},
        {"a":"3a","b":"11b","c":"12c"},
])
df['d'] = df["a"].apply(lambda x: df.loc[df["a"] == x, "b"].unique().tolist())
print(df)


str1 = "Runoob example....wow!!!"
str2 = "exam"
print(str1.find(str2, 5))

print("{:#>4d}".format(23))
print("{:#4d}".format(23))