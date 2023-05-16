import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the dataset
df = pd.read_csv("heart_2020_cleaned.csv")

# Data preprocessing
# Define categorical and numerical variables
categorical_columns = ["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity", "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]
numerical_columns = ["PhysicalHealth", "SleepTime", "MentalHealth", "BMI"]

print("Number of NaN rows:", df.isna().sum().sum())
print("Total rows:", len(df))

# Remove rows with any missing values
df = df.dropna(subset=numerical_columns+categorical_columns, how='any')
print("Number of rows after removing NaN rows:", len(df))
print("\n")

#Define Standard Scaler
scaler = StandardScaler()

# Define the threshold for outlier detection
outlier_threshold = 1.5

# Function to remove outliers using the IQR method
def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - outlier_threshold * IQR
        upper_bound = Q3 + outlier_threshold * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Create box plots for numerical columns
plt.figure(figsize=(10, 6))
df[numerical_columns].boxplot()
plt.title("Box Plot of Numerical Columns")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.show()

# Remove outliers from numerical columns
df = remove_outliers(df, numerical_columns)

# print out unique value in each cat columns
for col in categorical_columns:
    print(f"Unique categories in {col}: {df[col].unique()}")

# Define encoding mappings
smoking_mapping = {"No": 0, "Yes": 1}
alcohol_mapping = {"No": 0, "Yes": 1}
stroke_mapping = {"No": 0, "Yes": 1}
diff_walking_mapping = {"No": 0, "Yes": 1}
sex_mapping = {"Female": 0, "Male": 1}
age_cat_mapping = {"55-59": 0, "80 or older": 1, '65-69': 2, '75-79': 3, '40-44': 4, '70-74': 5, '60-64': 6, '50-54': 7, '45-49': 8, '18-24': 9, '35-39': 10, '30-34': 11, '25-29': 12}
race_mapping = {"White": 0, "Black": 1, "Asian": 2, "American Indian/Alaskan Native": 3, "Other": 4, "Hispanic": 5}
diabetic_mapping = {"Yes": 0, "No": 1, "No, borderline diabetes": 2, "Yes (during pregnancy)": 3}
physical_activity_mapping = {"No": 0, "Yes": 1}
gen_health_mapping = {"Very good": 0, 'Fair': 1, 'Good': 2, 'Poor':3, 'Excellent':4}
asthma_mapping = {"No": 0, "Yes": 1}
kidney_disease_mapping = {"No": 0, "Yes": 1}
skin_cancer_mapping = {"No":0, "Yes":1}

# Apply encoding mappings to categorical columns
df["Smoking"] = df["Smoking"].map(smoking_mapping)
df["AlcoholDrinking"] = df["AlcoholDrinking"].map(alcohol_mapping)
df["Stroke"] = df["Stroke"].map(stroke_mapping)
df["DiffWalking"] = df["DiffWalking"].map(diff_walking_mapping)
df["Sex"] = df["Sex"].map(sex_mapping)
df["AgeCategory"] = df["AgeCategory"].map(age_cat_mapping)
df["Race"] = df["Race"].map(race_mapping)
df["Diabetic"] = df["Diabetic"].map(diabetic_mapping)
df["PhysicalActivity"] = df["PhysicalActivity"].map(physical_activity_mapping)
df["GenHealth"] = df["GenHealth"].map(gen_health_mapping)
df["Asthma"] = df["Asthma"].map(asthma_mapping)
df["KidneyDisease"] = df["KidneyDisease"].map(kidney_disease_mapping)
df['SkinCancer'] = df['SkinCancer'].map(skin_cancer_mapping)

# Define target and features
target = "HeartDisease"
features = categorical_columns + numerical_columns
print()
print("Number of features:",len(features))

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2)

#Feature Scaling on numerical variables
scaler.fit(train_df[numerical_columns])

# Convert target column to numerical values
train_df[target] = pd.Categorical(train_df[target]).codes
test_df[target] = pd.Categorical(test_df[target]).codes

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_df[features] = train_df[features].astype(float)

# Add a regularization term to the loss function
#loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + \
    #tf.keras.regularizers.l2(0.001)(model.weights)

# Train the model
history = model.fit(train_df[features], train_df[target], epochs=10, batch_size=32, validation_data=(test_df[features], test_df[target]))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_df[features], test_df[target])
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


# Extract the training and validation loss and accuracy
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot the learning curve
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 4))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Learning Curve - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.title('Learning Curve - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('heart_disease_model.h5')