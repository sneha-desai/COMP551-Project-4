import pandas as pd
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import matplotlib as plt
from sklearn import linear_model


FILENAME = "Data/compas-scores-two-years-violent.csv"

#Columns
ID = "id"
AGE = "age"
RACE = "race"
SEX = "sex"
PRIORS_COUNT = "priors_count"
JAIL_TIME_IN = "c_jail_in"
JAIL_TIME_OUT = "c_jail_out"
CHARGE_DEG = "c_charge_degree"
CHARGE_DES = "c_charge_desc"

COLS = [AGE, RACE, SEX, PRIORS_COUNT, JAIL_TIME_IN, JAIL_TIME_OUT, CHARGE_DEG, CHARGE_DES]

df = pd.read_csv(FILENAME, usecols=COLS)

#.split(" ")[0]