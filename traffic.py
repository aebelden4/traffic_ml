# App to predict traffic volume
# Using four pre-trained ML models in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


### Intro to the app ###
st.title('Traffic Volume Prediction: A Machine Learning App') 

# Display the image
st.image('traffic_image.gif', width = 700)

st.subheader('Utilize our advanced Machine Learning application to predict traffic volume.') 

st.write('Use the following form to get started')


### Reading all of the pickled models ###
# Decision Tree
dt_pickle = open('dt_traffic.pickle', 'rb') 
clr_dt = pickle.load(dt_pickle) 
dt_pickle.close() 

# Random Forest
rf_pickle = open('rf_traffic.pickle', 'rb') 
clr_rf = pickle.load(rf_pickle) 
rf_pickle.close() 

# AdaBoost
ada_pickle = open('ada_traffic.pickle', 'rb') 
clr_ada = pickle.load(ada_pickle) 
ada_pickle.close() 

# XGBoost
xgb_pickle = open('xgb_traffic.pickle', 'rb') 
clr_xgb = pickle.load(xgb_pickle) 
xgb_pickle.close() 

### Load in dataset ###
# Load dataset as dataframe
traffic_df = pd.read_csv('Traffic_Volume.csv', keep_default_na=False)
# Drop weather description column
traffic_df = traffic_df.drop('weather_description', axis=1)
traffic_df['date_time'] = pd.to_datetime(traffic_df['date_time'])
# Add a column for month name
traffic_df['month'] = traffic_df['date_time'].dt.strftime('%B')
# Add a column for day of week name
traffic_df['day_of_week'] = traffic_df['date_time'].dt.strftime('%A')
# Add a column for time
traffic_df['time'] = traffic_df['date_time'].dt.strftime('%H:%M:%S')
traffic_df =  traffic_df.drop('date_time', axis=1)
# Output column for prediction
output = traffic_df['traffic_volume'] 

traffic_df = traffic_df.drop('traffic_volume', axis=1)
# Input features (excluding year column)
features = traffic_df[['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'month', 'day_of_week', 'time']] 
# One-hot-encoding for categorical variables
features = pd.get_dummies(features)
train_X, test_X, train_y, test_y = train_test_split(features, output, test_size = 0.2, random_state = 1) 

### Creating Model Metrics Dataframe ###
pred1 = clr_dt.predict(test_X)
pred2 = clr_rf.predict(test_X)
pred3 = clr_ada.predict(test_X)
pred4 = clr_xgb.predict(test_X)

# Calculate R^2 and RMSE values
r2_1 = r2_score(test_y, pred1)
rmse_1 = sqrt(mean_squared_error(test_y, pred1))

r2_2 = r2_score(test_y, pred2)
rmse_2 = sqrt(mean_squared_error(test_y, pred2))

r2_3 = r2_score(test_y, pred3)
rmse_3 = sqrt(mean_squared_error(test_y, pred3))

r2_4 = r2_score(test_y, pred4)
rmse_4 = sqrt(mean_squared_error(test_y, pred4))

# Create dataframe
data = {
    'ML Model': ['Decision Tree', 'Random Forest', 'AdaBoost', 'XGBoost'],
    'R2': [r2_1, r2_2, r2_3, r2_4],
    'RMSE': [rmse_1, rmse_2, rmse_3, rmse_4]
}

models_df = pd.DataFrame(data)

# Identify the row with the lowest and highest R^2 values
min_row = models_df[models_df['R2'] == models_df['R2'].min()]
max_row = models_df[models_df['R2'] == models_df['R2'].max()]

# Create a style DataFrame with background colors for the entire row
models_df = models_df.style.apply(lambda row: ['background-color: orange' if row.name in min_row.index else '' for col in row], axis=1)
models_df = models_df.apply(lambda row: ['background-color: green' if row.name in max_row.index else '' for col in row], axis=1)


### Form for Users ###
# Asking users to input their data using a form
with st.form('user_inputs'): 

  # All user inputs
  holiday = st.selectbox('Choose whether today is a designated holiday or not', options=[
    'None', 'Labor Day', 'Thanksgiving Day', 'Christmas Day', 'New Years Day',
    'Martin Luther King Jr Day', 'Columbus Day', 'Veterans Day', 'Washingtons Birthday',
    'Memorial Day', 'Independence Day', 'State Fair']) 
  
  temp = st.number_input('Average temperature in Kelvin', min_value=0.0) 

  rain_1h = st.number_input('Amount in mm of rain that occurred in the hour', min_value=0.00)

  snow_1h = st.number_input('Amount in mm of snow that occurred in the hour', min_value=0.00)

  clouds_all = st.number_input('Percentage of cloud cover', min_value=0)

  weather_main = st.selectbox('Choose the current weather', options=
                              ['Clouds', 'Clear', 'Mist', 'Rain', 'Snow', 'Drizzle', 'Haze',
                               'Thunderstorm', 'Fog', 'Smoke', 'Squall']) 
  
  month = st.selectbox('Choose month', options=
                       ['January', 'February', 'March', 'April', 'May', 'June', 'July', 
                        'August', 'September', 'October', 'November', 'December'])

  day_of_week = st.selectbox('Choose day of the week', options=
                             ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                              'Friday', 'Saturday', 'Sunday']) 
  
  time = st.selectbox('Choose hour', options=traffic_df["time"].unique())
                           #   ['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00', 
                           #    '05:00:00', '06:00:00', '07:00:00', '08:00:00',
                           #    '09:00:00', '10:00:00', '11:00:00', '12:00:00',
                           #    '13:00:00', '14:00:00', '15:00:00', '16:00:00',
                           #    '17:00:00', '18:00:00', '19:00:00', '20:00:00',
                           #    '21:00:00', '22:00:00', '23:00:00']) 

  model = st.selectbox('Select Machine Learning Model for Prediction', options=
                         ['Decision Tree', 'Random Forest', 'AdaBoost', 'XGBoost']) 
  
  # Print Model Metrics Dataframe
  st.dataframe(models_df)

  # Submit button
  st.form_submit_button() 


# ### Dummy Variables for Categorical ###
if model == 'Decision Tree': 
   model_selection = clr_dt 
elif model == 'Random Forest': 
   model_selection = clr_rf 
elif model == 'AdaBoost': 
   model_selection = clr_ada
elif model == 'XGBoost': 
   model_selection = clr_xgb

original = traffic_df.copy()
print(features)
original.loc[len(original)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, month, day_of_week, time]
cat_var = ['holiday', 'weather_main', 'month', 'day_of_week', 'time']
encode_dummy_df = pd.get_dummies(original, columns=cat_var)
user_encoded_df = encode_dummy_df.tail(1)


### Prediction for User Input ###
# Using predict() with new data provided by the user
new_prediction = model_selection.predict(user_encoded_df) 
new_prediction = new_prediction.round(0).astype(int)

# Show the predicted traffic volume on the app
st.write('{} Prediction: {}'.format(model, new_prediction[0])) 


### Fature Importance ###
st.subheader("Plot of Feature Importance")
if model == 'Decision Tree': 
   st.image('dt_feature_imp.svg')
elif model == 'Random Forest': 
   st.image('rf_feature_imp.svg')
elif model == 'AdaBoost': 
   st.image('ada_feature_imp.svg')
elif model == 'XGBoost': 
   st.image('xgb_feature_imp.svg')
