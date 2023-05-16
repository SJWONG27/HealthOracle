import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('./heart prediction backend/heart_2020_cleaned.csv')
data.head()
data.info()

data['HeartDisease'].value_counts().sort_values()
data['BMI'].value_counts().sort_values()
data['Smoking'].value_counts().sort_values()
data['AlcoholDrinking'].value_counts().sort_values()
data['Stroke'].value_counts().sort_values()
data['PhysicalHealth'].value_counts().sort_values()
data['MentalHealth'].value_counts().sort_values()
data['DiffWalking'].value_counts().sort_values()
data['Sex'].value_counts().sort_values()
data['AgeCategory'].value_counts().sort_values()
data['Race'].value_counts().sort_values()
data['Diabetic'].value_counts().sort_values()
data['PhysicalActivity'].value_counts().sort_values()
data['GenHealth'].value_counts().sort_values()
data['SleepTime'].value_counts().sort_values()
data['Asthma'].value_counts().sort_values()
data['KidneyDisease'].value_counts().sort_values()
data['SkinCancer'].value_counts().sort_values()

clean_data = {'Sex': {'Male' : 0 , 'Female' : 1} ,
                'HeartDisease': {'No': 0 , 'Yes' : 1},
                'AgeCategory' : {'18-24': 0,'25-29': 1,'30-34': 2,'35-39':3,'40-44':4,'45-49':5,'50-54':6,'55-59':7,'60-64':8,'65-69':9,'70-74':10,'75-79':11,'80 or older':12},
                'Smoking': {'No': 0 , 'Yes' : 1},
                'AlcoholDrinking': {'No': 0 , 'Yes' : 1},
                'Stroke': {'No': 0 , 'Yes' : 1},
                'DiffWalking': {'No': 0 , 'Yes' : 1},
                'PhysicalActivity': {'No': 0 , 'Yes' : 1},
                'Asthma': {'No': 0 , 'Yes' : 1},
                'KidneyDisease': {'No': 0 , 'Yes' : 1},
                'SkinCancer': {'No': 0 , 'Yes' : 1},
                'GenHealth' : {'Poor':0, 'Fair':1,'Excellent':2,'Good':3, 'Very good':4},
                'Diabetic' : {'Yes (during pregnancy)': 0, 'No, borderline diabetes': 1, 'Yes': 2, 'No' : 3},
                'Race': {'American Indian/Alaskan Native': 0, 'Asian': 1, 'Other':2, 'Black':3, 'Hispanic':4, 'White':5},
               }

data_copy = data.copy()
data_copy.replace(clean_data, inplace=True)
data_copy.describe()
print(data_copy.describe())

corr = data_copy.corr()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr,cmap='BuPu',annot=True,fmt=".2f",ax=ax)
plt.title("Heart Disease Prediction")
plt.savefig('./images')
plt.show()

from sklearn.preprocessing import StandardScaler
data_pre = data_copy.copy()

tempAgeCategory = data_pre.AgeCategory
tempAgeCategory = tempAgeCategory.values.reshape(-1,1)
data_pre['AgeCategory'] = StandardScaler().fit_transform(tempAgeCategory)

tempStroke = data_pre.Stroke
tempStroke = tempStroke.values.reshape(-1,1)
data_pre['Stroke'] = StandardScaler().fit_transform(tempStroke)

tempDiffWalking = data_pre.DiffWalking
tempDiffWalking = tempDiffWalking.values.reshape(-1,1)
data_pre['DiffWalking'] = StandardScaler().fit_transform(tempDiffWalking)

data_pre.head()

X = data_pre.drop('HeartDisease',axis=1).values
y = data_pre['HeartDisease'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

print('Size of X_train : ', X_train.shape)
print('Size of y_train : ', y_train.shape)
print('Size of X_test : ', X_test.shape)
print('Size of Y_test : ', y_test.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data and preprocess if needed
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model and hyperparameters
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)

# Train the model on the train set
model.fit(X_train, y_train)

# Predict target values for the train and test sets
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate accuracy score for the train and test sets
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Print the evaluation metrics
print('Accuracy score (train): {:.3f}'.format(accuracy_train))
print('Accuracy score (test): {:.3f}'.format(accuracy_test))

# Random forest
from sklearn.ensemble import RandomForestClassifier

# Initialize the random forest model with 100 trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Convert y_train to a 1D array
y_train = y_train.ravel()

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model's accuracy
from sklearn.metrics import accuracy_score

accuracy_train = accuracy_score(y_train, rf_model.predict(X_train))
accuracy_test = accuracy_score(y_test, y_pred)

print('Accuracy score (train): {:.3f}'.format(accuracy_train))
print('Accuracy score (test): {:.3f}'.format(accuracy_test))

