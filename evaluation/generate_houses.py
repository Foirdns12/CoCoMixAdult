import os
import numpy as np
import pickle
import pandas as pd
import random

from demonstration.demonstration_data import load_df

NUM = 51

PATH = os.path.dirname(os.path.abspath(__file__))

randomState = np.random.RandomState(seed=3239)
test_df = load_df(train=False)
real_dft = []
i = 0
while i <= 50:
    sample = test_df.sample(n=1, random_state=randomState)
    if sample['geo_krs'].values in ['Ulm', "Neu_Ulm_Kreis", "Günzburg_Kreis", "Biberach_Kreis", "Alb_Donau_Kreis",
                                    "Heidenheim_Kreis", "Reutlingen_Kreis", "Göppingen_Kreis",
                                    "Dillingen_an_der_Donau_Kreis"] and sample['obj_buildingType'].values not in [
        'other_real_estate', 'special_real_estate']:
        i = i + 1
        all = np.append(sample.values[0], sample.index.values[0])
        real_dft.append(all)
# print(real_dft)

real_df = pd.DataFrame(real_dft)

col = test_df.columns
# print(col)
# print(np.append(col.values,'id'))
real_df.columns = np.append(col.values, 'id')
real_df["label"] = "real"
fake_df = pd.DataFrame({column: list(test_df[column].sample(NUM))
                        for column in test_df.columns})
fake_df["label"] = "fake"
fake_df["id"] = 0
for i in range(len(fake_df["obj_regio1"])):
    fake_df["obj_regio1"][i]=random.choice(['Bayern','Baden_Württemberg'])
    if fake_df["obj_regio1"][i] == 'Bayern':
        fake_df["geo_krs"][i] = random.choice(["Neu_Ulm_Kreis", "Günzburg_Kreis", "Dillingen_an_der_Donau_Kreis"])
    if fake_df["obj_regio1"][i] == 'Baden_Württemberg':
        fake_df["geo_krs"][i] = random.choice(['Ulm', "Biberach_Kreis", "Alb_Donau_Kreis",
                                    "Heidenheim_Kreis", "Reutlingen_Kreis", "Göppingen_Kreis"])
cocomix_foils = []
wachter_foils = []
for fname in os.listdir(os.path.join(PATH, "..", "cocomix")):
    if "_foils.pickle" in fname:
        with open(os.path.join(PATH, "..", "cocomix", fname), "rb") as f:
            report = pickle.load(f)

        # print(type(report), report[:10])
        for _, foil, _, price, _, id, predclass, fact_cl, factprice in report:
            named_foil = {k: v for k, v in foil.items()
                          if k in real_df.columns}
            price = np.around(price/10,0)
            factprice = np.around(factprice/10,0)
            named_foil["obj_purchasePrice"]=price*10000
            named_foil["wst_price_class"]=predclass
            named_foil["est_fact_purchasePrice"]= factprice*10000
            named_foil["id"] = id
            named_foil["true"] = (abs(fact_cl - predclass) == 2)
            if "cocomix" in fname:
                cocomix_foils.append(named_foil)
            elif "wachter" in fname:
                wachter_foils.append(named_foil)

cocomix_df = pd.DataFrame(cocomix_foils)
cocomix_df["label"] = "cocomix"

wachter_df = pd.DataFrame(wachter_foils)
wachter_df["label"] = "wachter"
ges = real_df.merge(cocomix_df, left_index=True, right_index=True, suffixes=('_real', '_cocomix'))
ges2 = ges.merge(wachter_df, left_index=True, right_index=True, suffixes=('', '_wachter'))
print(ges2)
real_df = real_df.sample(n=NUM, replace=False)
cocomix_df = cocomix_df.sample(n=NUM, replace=False)
wachter_df = wachter_df.sample(n=NUM, replace=False)
study_data = pd.concat([real_df, fake_df, cocomix_df, wachter_df], ignore_index=True)
study_data.to_csv(os.path.join(PATH, "real_fake_cocomix_wachter.csv"))
ges2.to_csv(os.path.join(PATH, "real_cocomix_wachter_list.csv"))
