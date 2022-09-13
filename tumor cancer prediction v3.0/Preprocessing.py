import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessing:

    def preproceccing(self,cancer_prediction):
        cancer_prediction.dropna(inplace=True)
        cancer_prediction.drop('Index',inplace=True,axis=1)
        cancer_prediction.drop_duplicates(inplace=True)
        cancer_prediction['diagnosis'] = cancer_prediction['diagnosis'].replace("M", 0)
        cancer_prediction['diagnosis'] = cancer_prediction['diagnosis'].replace("B", 1)
        cancer_prediction['diagnosis'] = cancer_prediction['diagnosis'].astype('int64')
        print(type(cancer_prediction))
        return cancer_prediction

    def scale(self,cancer_prediction):
        scaling = StandardScaler()
        x_test_c=pd.DataFrame(scaling.fit_transform(cancer_prediction.iloc[:, :30]))
        return x_test_c