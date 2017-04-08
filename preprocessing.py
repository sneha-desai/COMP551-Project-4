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

FILENAME = "Data/3-Year_Recidivism_for_Offenders_Admitted_to_Probation.csv"

# Columns
CRIME_ID = "Offender"
OFFENSE_CLASSIFICATION = "Convicting Offense Classification"
OFFENSE_TYPE = "Convicting Offense Type"
OFFENSE_SUBTYPE = "Convicting Offense Subtype"
RACE = "Race - Ethnicity"
SEX = "Sex"
LEVEL_OF_SUPERVISION = "Level of Supervision"
RECIDIVISM = "Recidivism - Prison Admission"

COLS = [OFFENSE_CLASSIFICATION, OFFENSE_TYPE, OFFENSE_SUBTYPE, RACE, SEX, LEVEL_OF_SUPERVISION, RECIDIVISM]
TRAINING_THRESHOLD = 0.9

print("Preprocessing data...")
# Reading csv as dataframe
df = pd.read_csv(FILENAME, usecols=COLS)

#Numerical Categorization of Offense Classification
df[OFFENSE_CLASSIFICATION].replace("Simple Misdemeanor", 0.1, inplace=True)
df[OFFENSE_CLASSIFICATION].replace("Serious Misdemeanor", 1, inplace=True)
df[OFFENSE_CLASSIFICATION].replace("Aggravated Misdemeanor", 2, inplace=True)
df[OFFENSE_CLASSIFICATION].replace("D Felony", 5, inplace=True)
df[OFFENSE_CLASSIFICATION].replace("C Felony", 10, inplace=True)
df[OFFENSE_CLASSIFICATION].replace("B Felony", 25, inplace=True)
df[OFFENSE_CLASSIFICATION].replace("A Felony", 50, inplace=True)
df[OFFENSE_CLASSIFICATION].replace("Felony - Enhancement to Original Penalty", 100, inplace=True)
df[OFFENSE_CLASSIFICATION].replace("Special Sentence 2005", 150, inplace=True)
df[OFFENSE_CLASSIFICATION].replace("Other Misdemeanor", 0, inplace=True)

#Numerical Categorization of Level of Supervision
df[LEVEL_OF_SUPERVISION].replace("Unknown", 0, inplace=True) #maybe replace this
df[LEVEL_OF_SUPERVISION].replace("None", 1, inplace=True)
df[LEVEL_OF_SUPERVISION].replace("Minimum", 2, inplace=True)
df[LEVEL_OF_SUPERVISION].replace("Minimum Risk Program", 2, inplace=True)
df[LEVEL_OF_SUPERVISION].replace("Administrative", 3, inplace=True)
df[LEVEL_OF_SUPERVISION].replace("Low Risk Probation", 4, inplace=True)
df[LEVEL_OF_SUPERVISION].replace("Low Normal", 5, inplace=True)
df[LEVEL_OF_SUPERVISION].replace("High Normal", 6, inplace=True)
df[LEVEL_OF_SUPERVISION].replace("Intensive", 7, inplace=True)
df[LEVEL_OF_SUPERVISION].replace("Not Available for Supervision", 0, inplace=True)
df[LEVEL_OF_SUPERVISION].fillna(0, inplace=True)



#Numerical Categorization of Sex
df[SEX].replace("Male", 1, inplace=True)
df[SEX].replace("Female", 0, inplace=True)
df[SEX].replace("Unknown", 2, inplace=True)

#Numerical Categorization of Offense Type
df[OFFENSE_TYPE].replace("Drug", 3, inplace=True)
df[OFFENSE_TYPE].replace("Violent", 4, inplace=True)
df[OFFENSE_TYPE].replace("Public Order", 2, inplace=True)
df[OFFENSE_TYPE].replace("Property", 1, inplace=True)
df[OFFENSE_TYPE].replace("Other", 5, inplace=True)

#Numerical Categorization of Offense SubType
df[OFFENSE_SUBTYPE].replace("Assault", 4.1, inplace=True)
df[OFFENSE_SUBTYPE].replace("Sex", 4.2, inplace=True)
df[OFFENSE_SUBTYPE].replace("Other Violent", 4.3, inplace=True)
df[OFFENSE_SUBTYPE].replace("Drug Possession", 3.1, inplace=True)
df[OFFENSE_SUBTYPE].replace("Trafficking", 3.2, inplace=True)
df[OFFENSE_SUBTYPE].replace("Other Drug", 3.3, inplace=True)
df[OFFENSE_SUBTYPE].replace("Traffic", 2.1, inplace=True)
df[OFFENSE_SUBTYPE].replace("Alcohol", 2.2, inplace=True)
df[OFFENSE_SUBTYPE].replace("Flight/Escape", 2.4, inplace=True)
df[OFFENSE_SUBTYPE].replace("OWI", 2.3, inplace=True)
df[OFFENSE_SUBTYPE].replace("Other Public Order", 2.5, inplace=True)
df[OFFENSE_SUBTYPE].replace("Theft", 1.1, inplace=True)
df[OFFENSE_SUBTYPE].replace("Burglary", 1.2, inplace=True)
df[OFFENSE_SUBTYPE].replace("Forgery/Fraud", 1.3, inplace=True)
df[OFFENSE_SUBTYPE].replace("Vandalism", 1.4, inplace=True)
df[OFFENSE_SUBTYPE].replace("Other Property", 1.5, inplace=True)
df[OFFENSE_SUBTYPE].replace("Weapons", 2.6, inplace=True)
df[OFFENSE_SUBTYPE].replace("Prostitution/Pimping", 2.7, inplace=True)
df[OFFENSE_SUBTYPE].replace("Animals", 5.1, inplace=True)
df[OFFENSE_SUBTYPE].replace("Arson", 1.6, inplace=True)
df[OFFENSE_SUBTYPE].replace("Other Criminal", 5.2, inplace=True)
df[OFFENSE_SUBTYPE].replace("Kidnap", 4.4, inplace=True)
df[OFFENSE_SUBTYPE].replace("Murder/Manslaughter", 4.5, inplace=True)
df[OFFENSE_SUBTYPE].replace("Gambling", 2.8, inplace=True)
df[OFFENSE_SUBTYPE].replace("Health/Medical", 5.3, inplace=True)
df[OFFENSE_SUBTYPE].replace("Other Government", 5.4, inplace=True)
df[OFFENSE_SUBTYPE].replace("Natural Resources", 2.9, inplace=True)
df[OFFENSE_SUBTYPE].replace("Stolen Property", 1.7, inplace=True)
df[OFFENSE_SUBTYPE].replace("Business", 5.5, inplace=True)
df[OFFENSE_SUBTYPE].replace("Tax Laws", 5.6, inplace=True)
df[OFFENSE_SUBTYPE].replace("Robbery", 5.7, inplace=True)

#Numerical Categorization of Race
le = preprocessing.LabelEncoder()
b = le.fit(df[RACE])
a = le.transform(df[RACE])
df[RACE].replace("American Indian or Alaska Native - Hispanic", 0, inplace=True)
df[RACE].replace("American Indian or Alaska Native - Non-Hispanic", 1, inplace=True)
df[RACE].replace("Asian or Pacific Islander - Non-Hispanic", 3, inplace=True)
df[RACE].replace("Asian or Pacific Islander - Hispanic", 2, inplace=True)
df[RACE].replace("Black - Hispanic", 4, inplace=True)
df[RACE].replace("Black - Non-Hispanic", 5, inplace=True)
df[RACE].replace("Unknown", 6, inplace=True)
df[RACE].replace("White - Hispanic", 7, inplace=True)
df[RACE].replace("White - Non-Hispanic", 8, inplace=True)

#Numerical Categorization of Recidivism
df[RECIDIVISM].replace("Yes", 1, inplace=True)
df[RECIDIVISM].replace("No", 0, inplace=True)

df = df[df.Sex != 2]
df = df[np.isfinite(df["Convicting Offense Classification"])]
df.to_csv("output/preprocessed_data.csv")


x_values = df[["Convicting Offense Classification", "Convicting Offense Type", "Convicting Offense Subtype", "Race - Ethnicity", "Sex", "Level of Supervision"]]
x_values = x_values.to_sparse()
y_values = df['Recidivism - Prison Admission']
#print("Y-mean : " + str(y_values.mean()))
y_values = df['Recidivism - Prison Admission'].tolist()
y_values = np.asarray(y_values)



print("Splitting data to train and test sets")
x_train, x_val, y_train, y_val = train_test_split(x_values, y_values, test_size=0.1)

print("Fitting data for Random Forest Classification")
rf = RandomForestClassifier(min_samples_leaf=20)
rf.fit(x_train, y_train)

print("Prediction training...")
pred_train = rf.predict(x_train)
pred_train_df = pd.DataFrame({
    'prediction': pred_train,
    'category': y_train
})


# Calculate training accuracy
j = 0
y = 0
result = []
for j in range(len(pred_train_df)):
    if pred_train_df['prediction'][j] == pred_train_df['category'][j]:
        y = 1
    else:
        y = 0
    result.append(y)

b = pd.Series(result)
pred_train_df['result'] = b.values
summation_train = pred_train_df['result'].sum(axis=0)
accuracy_train = float(summation_train) / float(len(pred_train_df))
print("Training accuracy: " + str(accuracy_train))

print("Prediction validation...")
pred_val = rf.predict(x_val)
pred_val_df = pd.DataFrame({
    'prediction': pred_val,
    'category': y_val
})
i = 0
x = 0
result = []
for i in range(len(pred_val_df)):
    if pred_val_df['prediction'][i] == pred_val_df['category'][i]:
        x = 1
    else:
        x = 0
    result.append(x)

a = pd.Series(result)
pred_val_df['result'] = a.values
summation_val = pred_val_df['result'].sum(axis=0)
accuracy_val = float(summation_val) / float(len(pred_val_df))
print("Validation accuracy: " + str(accuracy_val))
