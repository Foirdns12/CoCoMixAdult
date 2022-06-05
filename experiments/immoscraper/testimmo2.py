from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import mean, absolute
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
scaler=StandardScaler()
df = pd.ExcelFile("C:/Users/phili/Desktop/MA/Datensatz/Immoalt/Immobearbkurzohnepreistest4.xlsx").parse(0)
dfy = pd.ExcelFile("C:/Users/phili/Desktop/MA/Datensatz/Immoalt/Immobearbkurzpricetest2.xlsx").parse(0).values.ravel()
lab_enc = preprocessing.LabelEncoder()
dfyenc = lab_enc.fit_transform(dfy)

dftrain,dftest,dfytrain,dfytest = train_test_split(df,dfyenc,train_size=.7)


scaler.fit(dftrain)
dftrains = scaler.transform(dftrain)
dftests=scaler.transform(dftest)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 5), random_state=5)
clf.fit(dftrain,dfytrain)
pred=clf.predict(dftest)
print(clf.score(dftest, dfytest))
print(mean(absolute(pred-dfytest)))