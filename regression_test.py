import unittest
import pandas as pd
import streamlit as st
import pickle

def user_input_features_test():
    X = pd.read_csv("clean_X.csv")
    Age = st.sidebar.text_input('ancienneté du bien', int(X.Age.mean()))
    GrLivArea = st.sidebar.text_input('Surface au sol', int(X.GrLivArea.mean()))
    LotFrontage = st.sidebar.text_input('Taille de la façade', int(X.LotFrontage.mean()))
    LotArea = st.sidebar.text_input('Surface totale', int(X.LotArea.mean()))
    GarageArea = st.sidebar.text_input('Taille du garage', int(X.GarageArea.mean()))
    Fence = st.sidebar.select_slider('Présence de barrières', options=[False, True], value = False)
    Pool = st.sidebar.select_slider('Piscine souhaitée?', options=[False, True], value = False)
    return Age, GrLivArea, LotFrontage, LotArea, GarageArea, Fence, Pool


class MyTestCase(unittest.TestCase):

    def test_user_input_feature_test(self):#je vérifie la présence de mes données dans le csv
        Age, GrLivArea, LotFrontage, LotArea, GarageArea, Fence, Pool = user_input_features_test()
        self.assert_(Age is not None)
        self.assert_(GrLivArea is not None)
        self.assert_(LotFrontage is not None)
        self.assert_(LotArea is not None)
        self.assert_(GarageArea is not None)
        self.assert_(Fence is not None)
        self.assert_(Pool is not None)

    def test_import_pickle_model(self): #je vérifie la bonne importation de ma data et que ma prédiction est un float
        Age, GrLivArea, LotFrontage, LotArea, GarageArea, Fence, Pool = user_input_features_test()
        data = {'Age': Age,
                'GrLivArea': GrLivArea,
                'LotFrontage': LotFrontage,
                'LotArea': LotArea,
                'GarageArea': GarageArea,
                'Fence': Fence,
                'Pool': Pool
                }
        df = pd.DataFrame(data, index=[0])
        loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
        prediction = loaded_model.predict(df)
        self.assert_(type(prediction), "float")

if __name__ == '__main__':
    unittest.main()
