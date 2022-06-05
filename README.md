# Categorical Explanations

Experiments to handle various kinds of categorical features when generating counterfactual explanations.

## Setup

Only runs on Python **3.7.x**.

### with Anaconda on Windows
```
conda install --file requirements-conda.txt
pip install -r requirements.txt
``` 

### with PyPi on Linux/Mac
```
pip install -r requirements-pypi.txt
```

## Adult data set
Download the *Adult Data Set* (US census 1994) from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult)
and place the data files in a new folder `/data/adult`.

To train a test model, run `models/adult_rf.py`. 

## Immo data set
Place the *Immobereinigtv2.xlsx* in `/data/immo`.
