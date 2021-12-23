import streamlit as st
import pandas as pd
import pickle
from functions import viz_model, viz_params

st.write("""
# Prédiction du prix de vente des biens immobiliers à Ames (Iowa USA)
""")
st.write('---')


X = pd.read_csv("clean_X.csv")


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Quels sont vos critères?')

@st.cache
def counter():
    result = []
    for i in X.FullBath:
        result.append(i)
        print(result)

def user_input_features():
    Age = st.sidebar.slider('Ancienneté du bien', int(X.Age.min()), int(X.Age.max()), int(X.Age.mean()))
    OverallQual = st.sidebar.slider('Qualité du matériel et de la finition de la maison', int(X.OverallQual.min()), int(X.OverallQual.max()), int(X.OverallQual.mean()))
    GrLivArea = st.sidebar.slider('Surface habitable au dessus du niveau du sol', int(X.GrLivArea.min()), int(X.GrLivArea.max()), int(X.GrLivArea.mean()))
    GarageCars = st.sidebar.slider('Capacité du garage', int(X.GarageCars.min()), int(X.GarageCars.max()), int(X.GarageCars.mean()))
    GarageArea = st.sidebar.slider('Taille du garage', int(X.GarageArea.min()), int(X.GarageArea.max()), int(X.GarageArea.mean()))
    FullBath = st.sidebar.slider('Salle de bains entières hors sous-sol', int(X.FullBath.min()), int(X.FullBath.max()), int(X.FullBath.mean()))
    TotRmsAbvGrd = st.sidebar.slider('Nombre de pièces hors sous-sol et hors salles de bains', int(X.TotRmsAbvGrd.min()), int(X.TotRmsAbvGrd.max()), int(X.TotRmsAbvGrd.mean()))

    data = {'Age': Age,
            'GrLivArea': GrLivArea,
            'OverallQual': OverallQual,
            'FullBath': FullBath,
            'GarageArea': GarageArea,
            'GarageCars': GarageCars,
            'TotRmsAbvGrd': TotRmsAbvGrd
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
# Main Panel

# Print specified input parameters
st.header('Précisez vos critères')
st.write(df)
st.write('---')

# Apply Model to Make Prediction
loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
prediction = loaded_model.predict(df)

formated_prediction = '${:,}'.format(int(prediction))
st.header('Prediction du prix de vente')
st.write(formated_prediction)
st.write('---')

formated_precision = '{:}'.format(int(loaded_model.cv_results_["mean_train_score"][loaded_model.best_index_]))
st.header("Précision de l'estimation")
st.write("Plus le score négatif est proche de 0, plus l'estimation est fiable.")
st.write(formated_precision)
st.write('---')
st.pyplot(viz_model(loaded_model))
st.write('---')
st.pyplot(viz_params(loaded_model))
st.write('---')



