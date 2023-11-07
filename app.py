import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Classifying Iris Flowers')
st.markdown('Model to classify Iris flowers inyo setosa, virginica and versicolor based on their sepal/petal length and width')

st.header('Plant Features')
col1, col2 = st.columns(2)
with col1:
    st.text('Sepal Characteristics')
    sepal_l = st.slider('Sepal length (cm)', 4.2, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 1.9, 4.5, 0.5)
with col2:
    st.text('Petal Characteristics')
    petal_l = st.slider('Petal length (cm)', 0.9, 7.0, 0.5)
    petal_w = st.slider('Petal_width (cm)', 0.0, 2.5, 0.5)

st.text('')
if st.button('Predict'):
    result = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    )
    st.text(result[0])