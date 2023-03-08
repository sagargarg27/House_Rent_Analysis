## House Rent Prediction App
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# import the model
best_model = pickle.load(open('E:/House_Rent_Analysis/best_model_final.pkl','rb'))

df = pickle.load(open('E:/House_Rent_Analysis/dataFrame_final.pkl','rb'))

with open('E:/House_Rent_Analysis/column_transformer.pkl', 'rb') as f:
    ct = pickle.load(f)

primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"


##HTML template
html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">House Rent Analysis and Prediction  </h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True,)

# Slidebar Latitude
latitude = st.sidebar.selectbox("**:blue[Latitude]**",df['latitude'] )

# Slidebar Longitude

longitude = st.sidebar.selectbox("**:blue[Longitude]**",df['longitude'])

# size_sq_ft
size_sq_ft = st.selectbox('**:blue[Room_Size]**',sorted(df['size_sq_ft'].unique()))

# propertyType
propertyType = st.selectbox('**:blue[PropertyType]**',['Independent_Floor','Apartment','Independent_House','Villa'])

# bedrooms
bedrooms= st.select_slider('**:blue[Bedrooms]**',[1,2,3,4,5,6,7,8,9,10,12,15])

# localityName
#localityName =st.selectbox('**:blue[LocalityName]**', df['localityName'].unique())
companyName =st.selectbox('**:blue[companyName]**',sorted(df['companyName'].unique()))


# suburbName
suburbName =st.selectbox('**:blue[SuburbName]**', sorted(df['suburbName'].unique()))

# closest_metro_station_km
closest_metro_station_km = st.number_input('**:blue[Closest_metro_station_km]**',min_value=0.00,max_value=100.00)

# AP_dist_km
AP_dist_km = st.number_input('**:blue[Airport_dist_km]**',min_value=0.00,max_value=100.00)

# Aiims_dist_km
Aiims_dist_km = st.number_input('**:blue[Aiims_dist_km]**',min_value=0.00,max_value=100.00)

# NDRLW_dist_km
NDRLW_dist_km = st.number_input('**:blue[NDRLW_dist_km]**',min_value=0.00,max_value=100.00)

## Prediction
if st.button('**Predict Rent**'):
    property_type_map = {'Independent_Floor': '1', 'Apartment': '2', 'Independent_House':' 3', 'Villa': '4'}
    propertyType = property_type_map[propertyType]
    
    tst= pd.DataFrame({'size_sq_ft':size_sq_ft,
                  'propertyType':propertyType,
                  'bedrooms':bedrooms,
                  'latitude':latitude,
                  'longitude':longitude,
                  'suburbName':suburbName,
                  'companyName':companyName,
                  'closest_metro_station_km':closest_metro_station_km,
                  'AP_dist_km':AP_dist_km,
                  'Aiims_dist_km':Aiims_dist_km,
                  'NDRLW_dist_km':NDRLW_dist_km} , index=[0])
    
    X_transf_tst = ct.transform(tst).toarray()
    # unpickle the best model
    y_pred = best_model.predict(X_transf_tst)
    progress_text = "*Operation in progress. Please wait...**"
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
        
    formatted_rent = '{:,}'.format(int(y_pred))
    st.success("**_Predicted monthly rent: ₹_**" + formatted_rent)
    if st.success:
            my_bar.empty()