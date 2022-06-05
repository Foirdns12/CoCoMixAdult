import os

import pandas as pd

from data.immodata import load_df

NUM = 100

PATH = os.path.dirname(os.path.abspath(__file__))

df = load_df(fillna=False, keep=["obj_regio1", "obj_numberOfFloors"])

real_df = df.sample(n=NUM, replace=False)
real_df["label"] = "real"

fake_df = pd.DataFrame({column: list(df[column].sample(NUM))
                        for column in df.columns})
fake_df["label"] = "fake"

study_data = pd.concat([real_df, fake_df], ignore_index=True)

study_data.to_csv(os.path.join(PATH, "real_fake.csv"))
