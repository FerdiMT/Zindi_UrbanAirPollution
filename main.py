import pandas as pd
from xgboost import XGBRegressor

train = pd.read_csv('Data/Train.csv')
test = pd.read_csv('Data/Test.csv')

# Data wrangling
# As a first test, we simply remove all the columns that have some NAS in them.
train.dropna(axis=1, how='any', inplace=True)
test.dropna(axis=1, how='any', inplace=True)

# We remove the additional columns that appear in train but not in test and separate label and features
label_train = train['target']
features_train = train[['precipitable_water_entire_atmosphere',
                        'relative_humidity_2m_above_ground',
                        'specific_humidity_2m_above_ground', 'temperature_2m_above_ground',
                        'u_component_of_wind_10m_above_ground',
                        'v_component_of_wind_10m_above_ground']]
features_test = test[['precipitable_water_entire_atmosphere',
                        'relative_humidity_2m_above_ground',
                        'specific_humidity_2m_above_ground', 'temperature_2m_above_ground',
                        'u_component_of_wind_10m_above_ground',
                        'v_component_of_wind_10m_above_ground']]

# Train model on all the training data. TODO: Cross validate, create train/validation split, etc...
model = XGBRegressor()
print('cava')
model.fit(features_train, label_train)

# Make predictions for test data
y_pred = model.predict(features_test)
predictions = [round(value) for value in y_pred]

# Ensamble predictions with the correct submission format
submission = test[['Place_ID X Date']].join(pd.DataFrame(predictions))
submission.columns = ['Place_ID X Date', 'target']
submission.to_csv('submission.csv', index=False)





