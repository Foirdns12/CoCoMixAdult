from CERTIFAI import CERTIFAI
import numpy as np
from model_instances import immo_model
import tensorflow as tf
from demonstration.demonstration_data import load_df, fill_numerical_column_by_cond_median, FEATURES, VAR_TYPES, ALL_CATEGORICAL_VALUES
from demonstration.proxy_measures.sparsity import make_sparsity, make_categorical_sparsity
import pandas as pd
from demonstration.proxy_measures.correctness import make_correctness
from demonstration.proxy_measures.closeness import make_distance, weighted_distance, wachter_distance
from demonstration.competing_approaches.wachter.util import calculate_mad
from demonstration.density_estimation.compute_kde import kde, preprocess
import uuid
import datetime
import os
import json

PATH = os.path.dirname(os.path.abspath(__file__))

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CustomModel:
    FEATURES = FEATURES

    def __init__(self):
        self.model = immo_model.load_model(code="20210222-161559")

    def predict(self, x):
        p_fact = {k: np.array(v, dtype=t) for k, v, t in
                  zip(self.FEATURES, x.transpose(), df_test[self.FEATURES].dtypes)}
        prediction = self.model.predict(p_fact)
        return prediction



df_train = load_df(train=True,WithID=True)
df_test = load_df(train=False,WithID=True)
df_test = fill_numerical_column_by_cond_median(source_df=df_train,
                                               condition_column="obj_buildingType",
                                               target_columns=["obj_numberOfFloors", "obj_noParkSpaces",
                                                               "obj_livingSpace", "obj_lotArea", "obj_noRooms",
                                                               "obj_yearConstructed"],
                                               target_df=df_test)
assert not np.any([np.any(df_test[column].isna()) for column in df_test.columns])


cat_ids = [(i, len(ALL_CATEGORICAL_VALUES[feature]), ALL_CATEGORICAL_VALUES[feature]) for i, feature in
           enumerate(FEATURES) if VAR_TYPES[i] != "c"]
con = [i for i in range(len(FEATURES)) if VAR_TYPES[i] == "c"]
cat = [i for i in range(len(FEATURES)) if VAR_TYPES[i] != "c"]


def to_dict(x):
    return {k: np.array([v], dtype=t) for k, v, t in
            zip(FEATURES, x, df_test[FEATURES].dtypes)}

def pdf(sample):
    return kde.pdf(preprocess(sample.reshape(1, -1)))


buckets = [None, 100, 150, 200, 350, 600, 800, 1000, None]
factor = 1.1


def pricecalc(predclass, predvect):
    if predclass == 0:
        price = np.around((1.125 - predvect[0]) * 100, 0)
    elif predclass == 7:
        price = np.around(((0.875 + predvect[7]) ** 2) * 1000, 0)
    else:
        try:
            price = _calculate_price(factor, predclass, predvect, buckets[predclass], buckets[predclass + 1])
        except IndexError:
            raise ValueError(f"predclass {predclass} is out of range.")
    return price


def _calculate_price(factor, predclass, predvect, lower, upper):
    return np.around((predvect[predclass - 1] ** factor / (
            predvect[predclass - 1] + predvect[predclass + 1]) ** factor) * lower + (
                             predvect[predclass + 1] ** factor / (
                             predvect[predclass - 1] + predvect[predclass + 1]) ** factor) * upper, 0)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

if __name__ == "__main__": #3foils pro sample erzeugen, 6a9bf444863b69e3d5c6f20b7c1fbf1b","343644f6f063af529efd467b9d913ed4",
    #samples=["0bbc5dd87e39ec3cc43ba862cc7a8b43","b7c431125894b2685d5f265da0117251","3742ba7d54477898609ac863ccc5e0d1","d0ac67b536a21eacd6b17fb8404f8f3e","03b3c40cae18f739cadab8e58a91f9d5","cb396c526074a19d98fe72c9889e1b8c","0923795407bdb14760e91500de0f94dd","4fd33f6bd560d58bcc94998110d1e685","bc1bd513ffd82c3054f9ac5cf229fad4","b680df519b2ff39e5a87dd828fab7304","6cfe75454edd304adf64af17acf1deef","cd36dd42525fea917f289fd9e1ab9bd7","3264313e64b7d716710207a08124bb06","bc0320cf9e7067768d167001a0f94c5c","e9ee4bfc475fbd441bcf76f233c0425b","04ffd88fe4e3f3aeb7c8d6c6a3124cd9","8cbc7f807de4a2ecb489a4b2bb9a32cb","859b2f4366570dcabb26361becc21e1e","26283664f9e04f68c2085a5fbb77e54b","79a52c007bf08b2ff44be507a176fdfa","405303d4f5c4a8bd6de835309d77c69b","0b3efdf3c210491abcc2794fdb2342ce","006c7532d5ca2c9b1bfe31e9deec35cc","6938279d0b1ef3008d7449f7f02a9531"]
    samplesstudie=["13466412e020f9f44e8a37a45ad72c24","6a9bf444863b69e3d5c6f20b7c1fbf1b","343644f6f063af529efd467b9d913ed4","94408a247e351d4b8dd22e1d5bc6a6eb","6c1a2480dfdd9ed8f010d20bb7dc4b01","fcbaddd63f3eabdc4b89aceadf0054f4","2fdb3cb813924672ede60f052cd4738b","11eab2b9e5b2272d64538993b2761003","be1e700ffaec7661aaacbcc2aab29220","9a0f0edf997d6261b0dc76625324acfa","13c268fd938a60a413745bc900bfe9bf","7eaf6cd9c85f6f50cbf8194421889572","61a1d594125b04ae8e50ad549a07f7b2","747d857933e56b038c25eb18750dc2fd","883024382ea437dc591f02fc498fc3b5","262169ea019bf6557574c787f0b583a1","245b29d60072312e75a6685da5bce6a8","c16a6927dd8519899f25d4b70282d54e","de8d97d1ab412ed1dd47ab43bad9138f","4f8bd369272f613d5dfdcc61fa39b6fa","8bcaeb8949cff1da048b4652b21de0af","97ed2d6f564ecfae4285bf6832cb13a3","fb8ce979e1661ce27fb5e5c6cf028caa","979fe059602e1beff7b234995c2928b6","96e29ee8102f83609bd11be2f11b2fa3","d4d2978d00ac8070507c17ea22c6fc83","b3cc6f62afaad948bdf39bc79ff110d9","abcaf3998694d58c67b23fc2859a483f","8627a80a9972860560e249927bea6328","1499e800623ab1a67a5753eb80b06184","b7f4af43905f8e5bbcd00c8da37a13f3","c2cb6b48d46d162a436f7e359b764654","a7dd0e9ff601cb6acba005e111f78d81","d8ad168606be789bcbb74a05395ce9ec","9ddc4e3289090d270a7def167af5389d","769fae66ce0250ef5a0b756cb2b3e969","625ff087b30a4239ded9d07f9b5270e8","23191ee679bba715c5b6555814c7c661","4bbfa046b609561086f162dbc6016855","e32e752d15d92318a7e5de1f5ff8005f","a846eb75aa817234ac5d9407b478baad"]
    studie=True
    sparsities=[]
    #categorical_sparsities=[]
    density_foils=[]
    cocomixdists=[]
    foillist=[]
    tabdistance=[]
    mad = calculate_mad(df_train[[feature for feature, var_type in zip(FEATURES, VAR_TYPES)
                                  if var_type == "c"]].to_numpy())
    mad[mad == 0.0] = 0.5
    cocomixdist=make_distance(weighted_distance, wachter_distance(mad), beta=0.5)
    sparsity = make_sparsity(FEATURES)
    categorical_sparsity = make_categorical_sparsity(FEATURES, VAR_TYPES)
    model = CustomModel()
    immo_model = immo_model.load_model(code="20210222-161559")
    ind=0
    # i in range(3):
    for factid in samplesstudie:
        addconstraints = True
        constraints = []
        if addconstraints == False:
            for i, j in zip(FEATURES, VAR_TYPES):
                if j != 'c':
                    constraints.append((0, len(pd.unique(df_train[i]))))
                else:
                    constraints.append((np.nanmin(df_train[i]), np.nanmax(df_train[i])))
        certifai_instance = CERTIFAI()
        certifai_instance.cat_ids = cat_ids
        certifai_instance.con = con
        certifai_instance.cat = cat
        print(factid)
        if studie:
            pos = df_test.index[df_test["factID"].astype(str) == factid]
            #print(pos)
            sample = df_test.loc[pos, FEATURES].to_numpy()
            #if len(sample)<2:
             #   pos = df_train.index[df_train["factID"].astype(str) == factid]
            #    sample = df_train.loc[pos, FEATURES].to_numpy()
             #   print(pos)
        else:
            pos=df_train.index[df_train["factID"].astype(str)==factid]
            sample = df_train.loc[pos, FEATURES].to_numpy()
            if len(sample)<2:
                pos = df_test.index[df_test["factID"].astype(str) == factid]
                sample = df_test.loc[pos, FEATURES].to_numpy()
                #print(pos)
        #print(sample)
        if addconstraints:
            constraints.append((max(np.nanmin(df_train['obj_yearConstructed']),sample[0][0]-100),2023))
            constraints.append((max(np.nanmin(df_train['obj_livingSpace']),sample[0][1]/3),min(np.nanmax(df_train['obj_livingSpace']), sample[0][1]*3)))
            constraints.append((max(np.nanmin(df_train['obj_noRooms']),sample[0][2]/3),min(np.nanmax(df_train['obj_noRooms']), sample[0][2]*3)))
            constraints.append((max(np.nanmin(df_train['obj_numberOfFloors']),sample[0][3]/3),min(np.nanmax(df_train['obj_numberOfFloors']), sample[0][3]*3)))
            constraints.append((max(np.nanmin(df_train['obj_lotArea']),sample[0][4]/3),min(np.nanmax(df_train['obj_lotArea']), sample[0][4]*3)))
            constraints.append((max(np.nanmin(df_train['obj_noParkSpaces']),sample[0][5]/3),min(np.nanmax(df_train['obj_noParkSpaces']), sample[0][5]*3)))
            for i, j in zip(FEATURES, VAR_TYPES):
                if j != 'c':
                    constraints.append(((0, len(pd.unique(df_train[i])))))
            certifai_instance.constraints = constraints
        certifai_instance.fit(model, x=np.array([sample]), pytorch=False, distance="tab_distance")
        fact = to_dict(certifai_instance.results[0][0])
        print("Fact: ", fact)
        print("Fact Class: ", immo_model.predict(fact).argmax())
        foil = to_dict(certifai_instance.results[0][1][0])
        print("Foil: ", foil)
        print("Foil Class: ", immo_model.predict(foil).argmax())
        #print(foil)
        foil['obj_yearConstructed'][0]=round(foil['obj_yearConstructed'][0])
        foil['obj_livingSpace'][0]=round(foil['obj_livingSpace'][0])
        foil['obj_noRooms'][0]=round(foil['obj_noRooms'][0])
        foil['obj_numberOfFloors'][0]=round(foil['obj_numberOfFloors'][0])
        foil['obj_lotArea'][0]=round(foil['obj_lotArea'][0])
        foil['obj_noParkSpaces'][0]=round(foil['obj_noParkSpaces'][0])
        fact['obj_yearConstructed'][0]=round(fact['obj_yearConstructed'][0])
        fact['obj_livingSpace'][0]=round(fact['obj_livingSpace'][0])
        fact['obj_noRooms'][0]=round(fact['obj_noRooms'][0])
        fact['obj_numberOfFloors'][0]=round(fact['obj_numberOfFloors'][0])
        fact['obj_lotArea'][0]=round(fact['obj_lotArea'][0])
        fact['obj_noParkSpaces'][0]=round(fact['obj_noParkSpaces'][0])
        #print(foil)
        sparsities.append(sparsity(fact, foil))
        cocomixdists.append(cocomixdist(fact, foil))
        density_foils.append(pdf(pd.DataFrame(certifai_instance.results[0][1][0]).to_numpy()[:, 0]))
        #print(density_foils)
        predclass = immo_model.predict(foil).argmax()
        predvect = immo_model.predict(foil)[0]
        price = pricecalc(predclass, predvect)
        factprice = pricecalc(immo_model.predict(fact).argmax(), immo_model.predict(fact)[0])
        id = uuid.uuid4()
        first = {'fact': fact, 'factid': factid, 'foil': foil,'foilid': str(id),'history': "None",
                 'conf': "generations=3000",'predclass': immo_model.predict(foil).argmax(),
                 'fact_cl': immo_model.predict(fact).argmax(),'foilprice': price, 'factprice': factprice}
        foillist.append(first)
        tabdistance.append(certifai_instance.results[0][2][0])
        #categorical_sparsities.append(categorical_sparsity(fact, foil))
    print('sparsity')
    print(np.mean(sparsities))
    print(np.mean(cocomixdists))
    print(np.mean(density_foils))
    print(np.mean(tabdistance))
    print(np.median(cocomixdists))
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fnamejsn = f"{current_time}_foilshs_certifai.json"
    with open(os.path.join(PATH, 'datenstudie', fnamejsn), "wt") as f:
        json.dump(foillist, f, indent=4, cls=NpEncoder)


