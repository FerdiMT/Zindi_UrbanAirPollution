import pandas as pd
from xgboost import XGBRegressor

train = pd.read_csv('Data/Train.csv')
test = pd.read_csv('Data/Test.csv')

# Check NAS
nas_test = pd.DataFrame(test.isnull().sum())
nas_train = pd.DataFrame(train.isnull().sum())
# Data wrangling
# We remove the columns that have more than 300 NAS in the TEST SET
test.dropna(thresh=len(test) - 300, axis=1, inplace=True)
# We get the test columns for the train features
label_train = train['target']
features_train = train[test.columns]

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





