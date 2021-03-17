import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# We remove the additional columns that appear in train but not in test and separate label and features
features_train = features_train.iloc[:,3:]
features_test = test.iloc[:,3:]

# Train model on all the training data.
model = XGBRegressor()

# Adding pipeline and param_grid
pipeline = Pipeline([
    ('standard_scaler', StandardScaler()),
    ('model', model)
])
# Already searched, we leave only the parameters that are the winners after an extense gridsearch.
param_grid = {
    'model__max_depth': [1,2,3,4,5,6,10],
    'model__n_estimators': [5,10,15],
    'model__eta':[0.01,0.1,0.3],
}

grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error', verbose=2)

grid.fit(features_train, label_train)

# Make predictions for test data
y_pred = grid.predict(features_test)
predictions = [round(value) for value in y_pred]

# Ensamble predictions with the correct submission format
submission = test[['Place_ID X Date']].join(pd.DataFrame(predictions))
submission.columns = ['Place_ID X Date', 'target']
submission.to_csv('submission2.csv', index=False)





