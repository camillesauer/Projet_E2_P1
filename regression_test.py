import unittest
import pandas as pd
import streamlit as st
import pickle

def user_input_features_test():
    X = pd.read_csv("clean_X.csv")
    Age = st.sidebar.slider('Ancienneté du bien', int(X.Age.min()), int(X.Age.max()), int(X.Age.mean()))
    OverallQual = st.sidebar.slider('Qualité du matériel et de la finition de la maison', int(X.OverallQual.mean()))
    GrLivArea = st.sidebar.slider('Surface habitable au dessus du niveau du sol', int(X.GrLivArea.mean()))
    GarageCars = st.sidebar.slider('Capacité du garage', int(X.GarageCars.mean()))
    GarageArea = st.sidebar.slider('Taille du garage', int(X.GarageArea.mean()))
    FullBath = st.sidebar.slider('Salle de bains entières hors sous-sol', int(X.FullBath.mean()))
    TotRmsAbvGrd = st.sidebar.slider('Nombre de pièces hors sous-sol et hors salles de bains', int(X.TotRmsAbvGrd.mean()))
    return Age, GrLivArea, OverallQual, FullBath, GarageArea, TotRmsAbvGrd, GarageCars


class MyTestCase(unittest.TestCase):

    def test_user_input_feature_test(self):#je vérifie la présence de mes données dans le csv
        Age, GrLivArea, FullBath, OverallQual, GarageArea, GarageCars, TotRmsAbvGrd = user_input_features_test()
        self.assert_(Age is not None)
        self.assert_(GrLivArea is not None)
        self.assert_(OverallQual is not None)
        self.assert_(FullBath is not None)
        self.assert_(GarageArea is not None)
        self.assert_(GarageCars is not None)
        self.assert_(TotRmsAbvGrd is not None)

    def test_import_pickle_model(self): #je vérifie la bonne importation de ma data et que ma prédiction est un float
        Age, GrLivArea, OverallQual, FullBath, GarageArea, GarageCars, TotRmsAbvGrd = user_input_features_test()
        data = {'Age': Age,
                'GrLivArea': GrLivArea,
                'OverallQual': OverallQual,
                'FullBath': FullBath,
                'GarageArea': GarageArea,
                'GarageCars': GarageCars,
                'TotRmsAbvGrd': TotRmsAbvGrd
                }
        df = pd.DataFrame(data, index=[0])
        loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
        prediction = loaded_model.predict(df)
        self.assert_(type(prediction), "float")

if __name__ == '__main__':
    unittest.main()
