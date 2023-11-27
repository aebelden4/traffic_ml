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
# rf_pickle = open('rf_traffic.pickle', 'rb') 
# clr_rf = pickle.load(rf_pickle) 
# rf_pickle.close() 

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
traffic_df = pd.read_csv('Traffic_Volume.csv')
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
# Input features (excluding year column)
features = traffic_df[['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'month', 'day_of_week', 'time']] 
# One-hot-encoding for categorical variables
features = pd.get_dummies(features)
train_X, test_X, train_y, test_y = train_test_split(features, output, test_size = 0.2, random_state = 1) 

### Creating Model Metrics Dataframe ###
pred1 = clr_dt.predict(test_X)
# pred2 = clr_rf.predict(test_X)
pred3 = clr_ada.predict(test_X)
pred4 = clr_xgb.predict(test_X)

# Calculate R^2 and RMSE values
r2_1 = r2_score(test_y, pred1)
rmse_1 = sqrt(mean_squared_error(test_y, pred1))

# r2_2 = r2_score(test_y, pred2)
# rmse_2 = sqrt(mean_squared_error(test_y, pred2))

r2_3 = r2_score(test_y, pred3)
rmse_3 = sqrt(mean_squared_error(test_y, pred3))

r2_4 = r2_score(test_y, pred4)
rmse_4 = sqrt(mean_squared_error(test_y, pred4))

# Create dataframe
data = {
    'ML Model': ['Decision Tree', #'Random Forest', 
                 'AdaBoost', 'XGBoost'],
    'R2': [r2_1, #r2_2, 
           r2_3, r2_4],
    'RMSE': [rmse_1, #rmse_2, 
             rmse_3, rmse_4]
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
                       ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                        'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

  day_of_week = st.selectbox('Choose day of the week', options=
                             ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                              'Friday', 'Saturday', 'Sunday']) 
  
  time = st.selectbox('Choose hour', options=
                             ['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00', 
                              '05:00:00', '06:00:00', '07:00:00', '08:00:00',
                              '09:00:00', '10:00:00', '11:00:00', '12:00:00',
                              '13:00:00', '14:00:00', '15:00:00', '16:00:00',
                              '17:00:00', '18:00:00', '19:00:00', '20:00:00',
                              '21:00:00', '22:00:00', '23:00:00']) 

  model = st.selectbox('Select Machine Learning Model for Prediction', options=
                         ['Decision Tree', #'Random Forest', 
                          'AdaBoost', 'XGBoost']) 
  
  # Print Model Metrics Dataframe
  st.dataframe(models_df)

  # Submit button
  st.form_submit_button() 


### Dummy Variables for Categorical ###
# Holiday
holiday_Labor_Day, holiday_Thanksgiving_Day, holiday_Christmas_Day = 0,0,0
holiday_New_Years_Day, holiday_Martin_Luther_King_Jr_Day, holiday_Columbus_Day = 0,0,0
holiday_Veterans_Day, holiday_Washingtons_Birthday, holiday_Memorial_Day = 0,0,0
holiday_Independence_Day, holiday_State_Fair = 0,0

if holiday == 'Labor Day': 
   holiday_Labor_Day = 1 
elif holiday == 'Thanksgiving Day': 
   holiday_Thanksgiving_Day = 1 
elif holiday == 'Christmas Day': 
   holiday_Christmas_Day = 1 
elif holiday == 'New Years Day': 
   holiday_New_Years_Day = 1 
elif holiday == 'Martin Luther King Jr Day': 
   holiday_Martin_Luther_King_Jr_Day = 1 
elif holiday == 'Columbus Day': 
   holiday_Columbus_Day = 1 
elif holiday == 'Veterans Day': 
   holiday_Veterans_Day = 1 
elif holiday == 'Washingtons Birthday': 
   holiday_Washingtons_Birthday = 1 
elif holiday == 'Memorial Day': 
   holiday_Memorial_Day = 1 
elif holiday == 'Independence Day': 
   holiday_Independence_Day = 1 
elif holiday == 'State Fair': 
   holiday_State_Fair = 1 

# Weather Main
weather_main_Clouds, weather_main_Clear, weather_main_Mist = 0,0,0
weather_main_Rain, weather_main_Snow, weather_main_Drizzle = 0,0,0
weather_main_Haze, weather_main_Thunderstorm, weather_main_Fog = 0,0,0
weather_main_Smoke, weather_main_Squall = 0,0

if weather_main == 'Clouds': 
   weather_main_Clouds = 1 
elif weather_main == 'Clear': 
   weather_main_Clear = 1 
elif weather_main == 'Mist': 
   weather_main_Mist = 1 
elif weather_main == 'Rain': 
   weather_main_Rain = 1 
elif weather_main == 'Snow': 
   weather_main_Snow = 1 
elif weather_main == 'Drizzle': 
   weather_main_Drizzle = 1 
elif weather_main == 'Haze': 
   weather_main_Haze = 1 
elif weather_main == 'Thunderstorm': 
   weather_main_Thunderstorm = 1 
elif weather_main == 'Fog': 
   weather_main_Fog = 1 
elif weather_main == 'Smoke': 
   weather_main_Smoke = 1 
elif weather_main == 'Squall': 
   weather_main_Squall = 1 

# Month
month_Jan, month_Feb, month_Mar, month_Apr = 0,0,0,0
month_May, month_Jun, month_Jul, month_Aug = 0,0,0,0
month_Sep, month_Oct, month_Nov, month_Dec = 0,0,0,0
if month == 'Jan': 
   month_Jan = 1 
elif month == 'Feb': 
   month_Feb = 1 
elif month == 'Mar': 
   month_Mar = 1 
elif month == 'Apr': 
   month_Apr = 1 
elif month == 'May': 
   month_May = 1 
elif month == 'Jun': 
   month_Jun = 1 
elif month == 'Jul': 
   month_Jul = 1 
elif month == 'Aug': 
   month_Aug = 1 
elif month == 'Sep': 
   month_Sep = 1 
elif month == 'Oct': 
   month_Oct = 1 
elif month == 'Nov': 
   month_Nov = 1 
elif month == 'Dec': 
   month_Dec = 1 

# Day of Week
day_of_week_Monday, day_of_week_Tuesday, day_of_week_Wednesday, day_of_week_Thursday = 0,0,0,0
day_of_week_Friday, day_of_week_Saturday, day_of_week_Sunday = 0,0,0

if day_of_week == 'Monday': 
   day_of_week_Monday = 1 
elif day_of_week == 'Tuesday': 
   day_of_week_Tuesday = 1 
elif day_of_week == 'Wednesday': 
   day_of_week_Wednesday = 1 
elif day_of_week == 'Thursday': 
   day_of_week_Thursday = 1 
elif day_of_week == 'Friday': 
   day_of_week_Friday = 1 
elif day_of_week == 'Saturday': 
   day_of_week_Saturday = 1 
elif day_of_week == 'Sunday': 
   day_of_week_Sunday = 1 

# Time
time_00, time_01, time_02, time_03, time_04, time_05, time_06 = 0,0,0,0,0,0,0
time_07, time_08, time_09, time_10, time_11, time_12 = 0,0,0,0,0,0
time_13, time_14, time_15, time_16, time_17, time_18 = 0,0,0,0,0,0
time_19, time_20, time_21, time_22, time_23= 0,0,0,0,0
if time == '00:00:00': 
   time_00 = 1 
elif time == '01:00:00': 
   time_01 = 1 
elif time == '02:00:00': 
   time_02 = 1
elif time == '03:00:00': 
   time_03 = 1 
elif time == '04:00:00': 
   time_04 = 1
elif time == '05:00:00': 
   time_05 = 1 
elif time == '06:00:00': 
   time_06 = 1
elif time == '07:00:00': 
   time_07 = 1 
elif time == '08:00:00': 
   time_08 = 1
elif time == '09:00:00': 
   time_09 = 1 
elif time == '10:00:00': 
   time_10 = 1
elif time == '11:00:00': 
   time_11 = 1 
elif time == '12:00:00': 
   time_12 = 1
elif time == '13:00:00': 
   time_13 = 1 
elif time == '14:00:00': 
   time_14 = 1
elif time == '15:00:00': 
   time_15 = 1 
elif time == '16:00:00': 
   time_16 = 1
elif time == '17:00:00': 
   time_17 = 1 
elif time == '18:00:00': 
   time_18 = 1
elif time == '19:00:00': 
   time_19 = 1 
elif time == '20:00:00': 
   time_20 = 1
elif time == '21:00:00': 
   time_21 = 1 
elif time == '22:00:00': 
   time_22 = 1
elif time == '23:00:00': 
   time_23 = 1 

if model == 'Decision Tree': 
   model_selection = clr_dt 
# elif model == 'Random Forest': 
#    model_selection = clr_rf 
elif model == 'AdaBoost': 
   model_selection = clr_ada
elif model == 'XGBoost': 
   model_selection = clr_xgb


### Prediction for User Input ###
data = [[
   temp, rain_1h, snow_1h, clouds_all, holiday_Christmas_Day, holiday_Columbus_Day, 
   holiday_Independence_Day, holiday_Labor_Day, holiday_Martin_Luther_King_Jr_Day, 
   holiday_Memorial_Day, holiday_New_Years_Day, holiday_State_Fair, 
   holiday_Thanksgiving_Day, holiday_Veterans_Day, holiday_Washingtons_Birthday,
   weather_main_Clear, weather_main_Clouds, weather_main_Drizzle, 
   weather_main_Fog, weather_main_Haze, weather_main_Mist, weather_main_Rain, 
   weather_main_Smoke, weather_main_Snow, weather_main_Squall, weather_main_Thunderstorm,
   month_Apr, month_Aug, month_Dec, month_Feb, month_Jan, month_Jul, month_Jun, month_Mar, 
   month_May, month_Nov, month_Oct, month_Sep, 
   day_of_week_Friday, day_of_week_Monday, day_of_week_Saturday, day_of_week_Sunday, 
   day_of_week_Thursday, day_of_week_Tuesday, day_of_week_Wednesday, 
   time_00, time_01, time_02, time_03, time_04, time_05, time_06, 
   time_07, time_08, time_09, time_10, time_11, time_12, 
   time_13, time_14, time_15, time_16, time_17, time_18, 
   time_19, time_20, time_21, time_22, time_23]]

# Using predict() with new data provided by the user
new_prediction = model_selection.predict(data) 
new_prediction = new_prediction.round(0).astype(int)

# Show the predicted traffic volume on the app
st.write('{} Prediction: {}'.format(model, new_prediction[0])) 


### Fature Importance ###
st.subheader("Plot of Feature Importance")
if model == 'Decision Tree': 
   st.image('dt_feature_imp.svg')
# elif model == 'Random Forest': 
#    st.image('rf_feature_imp.svg')
elif model == 'AdaBoost': 
   st.image('ada_feature_imp.svg')
elif model == 'XGBoost': 
   st.image('xgb_feature_imp.svg')
