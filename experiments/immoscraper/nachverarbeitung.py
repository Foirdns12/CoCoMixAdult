import os
import pandas as pd
import os


PATH = os.path.dirname(os.path.abspath(__file__))

df = pd.DataFrame()
n = 0
for i in os.listdir("C:/Users/huehn/Desktop/rohdatenkombiniert/"):
    n += 1
    df = df.append(pd.read_csv("C:/Users/huehn/Desktop/rohdatenkombiniert/" + str(i), sep=";", decimal=",", encoding="utf-8"))
    print("Durchgang " + str(n))

df.shape
df = df.drop_duplicates(subset="URL",keep='last')
df.shape
df.to_csv(r'C:/Users/huehn/Desktop/Immofinal.csv', index = False)