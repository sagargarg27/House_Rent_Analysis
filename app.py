## House Rent Prediction App
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
from PIL import Image

#Enter the location of git repo in case you are having any error
#os.chdir('E:/House_Rent_Analysis/')

# import the model
best_model = pickle.load(open('Pickle_Files/best_model_final.pkl','rb'))

df = pickle.load(open('Pickle_Files/dataFrame_final.pkl','rb'))

with open('Pickle_Files/column_transformer.pkl', 'rb') as f:
    ct = pickle.load(f)
    
y_train = pickle.load(open('Pickle_Files/y_train.pkl','rb'))
    
primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

## logo
logo = Image.open('Images/WebsiteLogo.jpg')
st.columns(3)[1].image(logo)

##HTML template
html_temp = """
    <div style="background-color:tomato;padding:01px;margin:01px">
    <h2 style="color:white;text-align:center;">House Rent Prediction</h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True,)
dff= df.groupby('localityName').mean().reset_index()

# size_sq_ft
size_sq_ft1=st.text_input("**:blue[Enter House Size in Sq. ft.]**",'0')
size_sq_ft=np.int64(size_sq_ft1)

# propertyType
propertyType = st.selectbox('**:blue[Select Property Type]**',['Independent_Floor','Apartment','Independent_House','Villa'])

# bedrooms
bedrooms= st.select_slider('**:blue[Select No. of Bedrooms]**',[1,2,3,4,5,6,7,8,9,10,12,15])

# suburbName
suburbName =st.selectbox('**:blue[Select Suburb Name]**', sorted(df['suburbName'].unique()))

# localityName
sk = df.groupby('suburbName')
availableLocality = sk.get_group(suburbName)['localityName'].unique()
localityName =st.selectbox('**:blue[Select Locality Name]**', sorted(availableLocality))

## Lat and Long logic
def findlat(args, dff):
     lat = pd.to_numeric(dff.loc[dff["localityName"]==args, "latitude"]).iloc[0]
     return "{:.6f}".format(lat)

def findlong(args, dff):
    long = pd.to_numeric(dff.loc[dff["localityName"]==args, "longitude"]).iloc[0]
    return "{:.6f}".format(long)

selectedlat = float(findlat(localityName,dff))
selectedlong = float(findlong(localityName,dff))

# Text_input Longitude and long
st.text_input("**:blue[Latitude of selected location]**", selectedlat,disabled=True)
st.text_input("**:blue[Longitude of selected location]**", selectedlong,disabled=True)

#companyName
lk=df.groupby('localityName')
availableCompany=lk.get_group(localityName)['companyName'].unique()

for i in range(len(availableCompany)):
    if availableCompany[i] == 'Other':
        availableCompany[i] = 'I do not prefer any specific agent'
companyName =st.selectbox('**:blue[Select Real Estate Dealer Name]**',sorted(availableCompany))
if companyName == 'I do not prefer any specific agent':
    companyName = 'Other'

# closest_metro_station_km
def findmetro(args, dff):
      lat = pd.to_numeric(dff.loc[dff["localityName"]==args, "closest_metro_station_km"]).iloc[0]
      return "{:.6f}".format(lat)
selectedmetro=float(findmetro(localityName,dff))
st.text_input('**:blue[Distance of Nearest Metro Station from selected location in kms]**', selectedmetro,disabled=True) 

# AP_dist_km
def findairport(args, dff):
     lat = pd.to_numeric(dff.loc[dff["localityName"]==args, "AP_dist_km"]).iloc[0]
     return "{:.6f}".format(lat)
selectedairport=float(findairport(localityName,dff))
st.text_input('**:blue[Distance of Airport from selected location in kms]**', selectedairport,disabled=True) 

# Aiims_dist_km
def findaiims(args, dff):
     lat = pd.to_numeric(dff.loc[dff["localityName"]==args, "Aiims_dist_km"]).iloc[0]
     return "{:.6f}".format(lat)
selectedaiims=float(findaiims(localityName,dff))
st.text_input('**:blue[Distance of AIIMS from selected location in kms]**', selectedaiims,disabled=True) 

# NDRLW_dist_km
def findrailway(args, dff):
     lat = pd.to_numeric(dff.loc[dff["localityName"]==args, "NDRLW_dist_km"]).iloc[0]
     return "{:.6f}".format(lat)
selectedrailway=float(findrailway(localityName,dff))
st.text_input('**:blue[Distance of New Delhi Railway Station from selected location in kms]**', selectedrailway,disabled=True)

## Prediction
if st.button('**Predict Rent**'):
    if size_sq_ft<100:
        st.error("**_Please Enter Valid House Size_**")
    else:
    
        ## Property type mapping
        property_type_map = {'Independent_Floor': '1', 'Apartment': '2', 'Independent_House':' 3', 'Villa': '4'}
        propertyType = property_type_map[propertyType]
        
        ## Assigning Values
        tst= pd.DataFrame({'size_sq_ft':size_sq_ft,
                           'propertyType':propertyType,
                           'bedrooms':bedrooms,
                           'latitude':selectedlat,
                           'longitude':selectedlong,
                           'suburbName':suburbName,
                           'companyName':companyName,
                           'closest_metro_station_km':selectedmetro,
                           'AP_dist_km':selectedairport,
                           'Aiims_dist_km':selectedaiims,
                           'NDRLW_dist_km':selectedrailway} , index=[0])
        
        X_transf_tst = ct.transform(tst).toarray()
        
        # unpickle the best model
        y_pred = best_model.predict(X_transf_tst)
        
        #Residual Error
        resid = y_train - best_model.oob_prediction_
        
        # Prediction Interval
        lowq = resid.quantile(0.25)
        higq = resid.quantile(0.75)
        
        test_y = best_model.predict(X_transf_tst)
        lowt = (test_y + lowq).clip(0) #cant have negative numbers
        higt = (test_y + higq)
        
        
        progress_text = "*Operation in progress. Please wait...*"
        my_bar = st.progress(0, text=progress_text)
        
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
            
        # Presenting Results
        formatted_lowt = '{:,}'.format(int(lowt))
        formatted_higt = '{:,}'.format(int(higt))
        st.success("**_The Approximate monthly rent of selected Property will be between ₹" + formatted_lowt + " to ₹" + formatted_higt + "_**")
        
        if st.success:
            my_bar.empty()
    
    