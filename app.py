import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle



# load the model

from tensorflow.keras.models import load_model


model = load_model('ann_model.h5')


# load the scalar and encoders

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender= pickle.load(file)

with open('onehot_encoder_geography.pkl','rb') as file:
    onehot_encoder_geography=pickle.load(file)

with open('scalar.pkl','rb') as file:
    scalar = pickle.load(file)



# Streamlit app

st.title("Customer churn prediction")

# User input 

geography = st.selectbox('Geography',onehot_encoder_geography.categories_[0])
# print(label_encoder_gender.classes_)
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider('Age',18,100,25,)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has CR Card',[0,1])
is_active_member = st.selectbox('Is active member',[0,1])

# prepare the input_data

input_data_df = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})



# st.dataframe(input_data_df)

# now handelling the geography

geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geography.get_feature_names_out(['Geography']))
# st.write(geo_encoded_df)

input_data_df = pd.concat([input_data_df,geo_encoded_df],axis=1)

# st.write(input_data_df)

# now scale the data

input_scaled = scalar.transform(input_data_df)

# st.write(input_scaled)


#  now predict churn with the model
prediction = model.predict(input_scaled)
# st.write(prediction)
# print(prediction[0][0])

st.header("AI prediction (ANN model)")

if(prediction[0][0] >= 0.5):
    st.write("The Customer is likely to churn")
else:
    st.write("The Customer is not likely to churn")






