import pandas as pd
from xgboost import XGBRegressor

train = pd.read_csv('Data/Train.csv')
test = pd.read_csv('Data/Test.csv')

# Check NAS
nas_test = pd.DataFrame(test.isnull().sum())
nas_train = pd.DataFrame(train.isnull().sum())
# Data wrangling
# We remove the columns that have more than 300 NAS in the TEST SET
test.dropna(thresh=len(test) - 5000, axis=1, inplace=True)
# We get the test columns for the train features
label_train = train['target']
features_train = train[test.columns]

# Create day of the week feature as dummy columns
features_train['Date'] = pd.to_datetime(features_train['Date'])
features_train['day_of_week'] = features_train['Date'].dt.day_name()
features_train['month'] = pd.DatetimeIndex(features_train['Date']).month
features_train = pd.get_dummies(features_train, columns=['day_of_week', 'month'])
test['Date'] = pd.to_datetime(test['Date'])
test['day_of_week'] = test['Date'].dt.day_name()
test['month'] = pd.DatetimeIndex(test['Date']).month
test = pd.get_dummies(test, columns=['day_of_week', 'month'])


# We remove the additional columns that appear in train but not in test and separate label and features
features_train = features_train.iloc[:,3:]
features_test = test.iloc[:,3:]

# Train model on all the training data. TODO: Cross validate, create train/validation split, etc...
model = XGBRegressor()

model.fit(features_train, label_train)

# Make predictions for test data
y_pred = model.predict(features_test)
predictions = [round(value) for value in y_pred]

# Ensamble predictions with the correct submission format
submission = test[['Place_ID X Date']].join(pd.DataFrame(predictions))
submission.columns = ['Place_ID X Date', 'target']
submission.to_csv('submission.csv', index=False)





