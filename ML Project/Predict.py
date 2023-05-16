from flask import Flask, render_template, request, g, flash, redirect, url_for, session
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import sqlite3

app = Flask(__name__, template_folder="C://Users//SJ//Desktop//ML Project//heart prediction//templates",
            static_folder="C://Users//SJ//Desktop//ML Project//heart prediction//static")
app.secret_key = 'MLproject'


# Establish a new connection for each thread
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect('HeartPrediction_History.db')
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


# Load the saved model
model = load_model('heart_disease_model.h5')

# Define the column names and ranges for validation
categorical_columns = ["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity", "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]
numerical_columns = ["PhysicalHealth", "SleepTime", "MentalHealth", "BMI"]
column_ranges = {
    "BMI": (0, 100),
    "SleepTime": (0, 24),
    "PhysicalHealth": (0, 30),
    "MentalHealth": (0, 30)
}

@app.before_request
def check_session():
    if request.endpoint in ['startup', 'history','heart'] and 'id' not in session:
        return redirect(url_for('index'))

@app.route("/startup")
def startup():
    if 'id' in session:
        # Retrieve the user ID from the session
        user_id = session.get('id')
        if user_id:
            # Retrieve the user data from the database using the user_id
            user_data = query_user_data_from_database(user_id)
            return render_template("startup.html", user_data=user_data)
        else:
            return redirect(url_for('index'))
    # User is not logged in, redirect to login page
    return redirect(url_for('index'))

@app.route("/register", methods=["POST", "GET"])
def register():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        # Create a new connection for the current thread
        conn = get_db()

        # Create the "users" table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')

        # Create the "prediction_history" table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                Smoking TEXT,
                AlcoholDrinking TEXT,
                Stroke TEXT,
                DiffWalking TEXT,
                Sex TEXT,
                AgeCategory TEXT,
                Race TEXT,
                Diabetic TEXT,
                PhysicalActivity TEXT,
                GenHealth TEXT,
                Asthma TEXT,
                KidneyDisease TEXT,
                SkinCancer TEXT,
                PhysicalHealth NUMERIC,
                SleepTime NUMERIC,
                MentalHealth NUMERIC,
                BMI NUMERIC,
                predicted_result NUMERIC,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # Check if the email already exists in the database
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email=?', (email,))
        existing_user = cursor.fetchone()
        cursor.close()

        if existing_user:
            # Email already exists in the database
            flash('Registration failed. Email already in use.', 'error')
            conn.close()    
            return redirect(url_for('register'))

        # Insert user data into the database
        conn.execute('INSERT INTO users (email, username, password) VALUES (?, ?, ?)', (email, username, password))
        conn.commit()

        # Close the connection
        conn.close()

        flash('User registered successfully', 'success')
        return redirect(url_for('index'))
    
    return render_template("registration.html")

def query_user_data_from_database(user_id):
    # Connect to the database
    conn = get_db()

    # Perform a query to retrieve the user data based on the user_id
    cursor = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user_data = cursor.fetchone()

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return user_data

def get_results():
  # Connect to the database
  conn = get_db()

  # Get the current session ID
  session_id = session['id']

  # Create a cursor
  cursor = conn.cursor()

  # Execute a SELECT query to retrieve the rows
  cursor.execute('SELECT prediction_time, predicted_result FROM prediction_history WHERE user_id=?', (session_id,))

  # Fetch the results
  results = cursor.fetchall()

  # Close the cursor and connection
  cursor.close()
  conn.close()

  # Convert the results to a list
  results_list = list(results)

  # Return the results
  return results_list



@app.route("/logout", methods=["GET", "POST"])
def logout():
    # Clear the user ID from the session
    session.pop('id', None)
    # Logout successful
    flash('Logout successful!', 'success')
    return redirect(url_for('index'))
    

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        # Create a new connection for the current thread
        conn = get_db()

        # Perform database operations using the connection
        cursor = conn.cursor()

        # Execute a SELECT query to retrieve the user with the given username and password
        cursor.execute('SELECT * FROM users WHERE email=? AND password=?', (email, password))

        # Fetch the first row from the result set
        user = cursor.fetchone()

        # Close the cursor and connection
        cursor.close()
        conn.close()

        if user:
            # Store the user ID in the session
            session['id'] = user[0]
            session['name'] = user[2]   
            # User credentials are valid
            return redirect(url_for('startup'))
        else:
            # User credentials are invalid
            flash('Invalid username or password', 'error')
            return redirect(url_for('index'))

    return render_template("login.html")

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("login.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/history")
def history():
    results = get_results()
    name = session.get('name')  # Get the name from the session
    return render_template("history.html", results=results, name=name)




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

reverse_smoking_mapping = {v: k for k, v in smoking_mapping.items()}
reverse_alcohol_mapping = {v: k for k, v in alcohol_mapping.items()}
reverse_stroke_mapping = {v: k for k, v in stroke_mapping.items()}
reverse_diff_walking_mapping = {v: k for k, v in diff_walking_mapping.items()}
reverse_sex_mapping = {v: k for k, v in sex_mapping.items()}
reverse_age_cat_mapping = {v: k for k, v in age_cat_mapping.items()}
reverse_race_mapping = {v: k for k, v in race_mapping.items()}
reverse_diabetic_mapping = {v: k for k, v in diabetic_mapping.items()}
reverse_physical_activity_mapping = {v: k for k, v in physical_activity_mapping.items()}
reverse_gen_health_mapping = {v: k for k, v in gen_health_mapping.items()}
reverse_asthma_mapping = {v: k for k, v in asthma_mapping.items()}
reverse_kidney_disease_mapping = {v: k for k, v in kidney_disease_mapping.items()}
reverse_skin_cancer_mapping = {v: k for k, v in skin_cancer_mapping.items()}

@app.route("/predict", methods=["POST"])
def predict():
    
    # Load the scaler
    #scaler = joblib.load(scaler_path)
    
    # Get the form data
    form_data = request.form.to_dict()

    name = form_data["name"]

    # Define the numerical columns used for normalization
    #numerical_columns = ["PhysicalHealth", "SleepTime", "MentalHealth", "BMI"]

    #Normalize the numerical values in form_data
    #normalized_data = form_data.copy()  # Create a copy of the form data

    #input_values = []
    #for col in numerical_columns:
    #   input_values.append(float(form_data[col]))

    #normalized_values = scaler.transform([input_values])[0]

    #normalized_data['PhysicalHealth'] = normalized_values[0]
    #normalized_data['SleepTime'] = normalized_values[1]
    #normalized_data['MentalHealth'] = normalized_values[2]
    #normalized_data['BMI'] = normalized_values[3]

    # Convert the form data to a list of values in the same order as your dataset
    new_sample = [form_data["Smoking"], form_data["AlcoholDrinking"], form_data["Stroke"], form_data["DiffWalking"],
              form_data["Sex"], form_data["AgeCategory"], form_data["Race"], form_data["Diabetic"],
              form_data["PhysicalActivity"], form_data["GenHealth"], form_data["Asthma"],
              form_data["KidneyDisease"], form_data["SkinCancer"], float(form_data["PhysicalHealth"]),
              float(form_data["SleepTime"]), float(form_data["MentalHealth"]), float(form_data["BMI"])]
    
    new_sample = [float(x) for x in new_sample]

    # Validate the input values
    for i, col in enumerate(numerical_columns):
        val = new_sample[i+len(categorical_columns)]
        if not column_ranges[col][0] <= val <= column_ranges[col][1]:
            return render_template("result.html", error=f"Invalid value {val} for column {col}")

    # Convert the list to a numpy array
    new_sample = np.array([new_sample])

    # Impute the missing values using KNN
    imputer = KNNImputer(n_neighbors=3)
    new_sample[:, :13] = imputer.fit_transform(new_sample[:, :13])

    # Scale the numerical features to the range [0,1]
    scaler = MinMaxScaler()
    new_sample[:, 13:17] = scaler.fit_transform(new_sample[:, 13:17])

    # Concatenate the imputed and normalized columns 0 to 13 with the normalized columns 14 to 16
    new_sample = np.concatenate([new_sample[:, :13], new_sample[:, 13:17]], axis=1)

    # Make the prediction
    prediction = model.predict(new_sample)

    # Print the predicted probability of heart disease in percentage
    predicted_prob = round(prediction[0][0]*100, 2)
    #decoded data
    decoded_data = [
        reverse_smoking_mapping[int(new_sample[0][0])],
        reverse_alcohol_mapping[int(new_sample[0][1])],
        reverse_stroke_mapping[int(new_sample[0][2])],
        reverse_diff_walking_mapping[int(new_sample[0][3])],
        reverse_sex_mapping[int(new_sample[0][4])],
        reverse_age_cat_mapping[int(new_sample[0][5])],
        reverse_race_mapping[int(new_sample[0][6])],
        reverse_diabetic_mapping[int(new_sample[0][7])],
        reverse_physical_activity_mapping[int(new_sample[0][8])],
        reverse_gen_health_mapping[int(new_sample[0][9])],
        reverse_asthma_mapping[int(new_sample[0][10])],
        reverse_kidney_disease_mapping[int(new_sample[0][11])],
        reverse_skin_cancer_mapping[int(new_sample[0][12])],
        float(form_data["PhysicalHealth"]),
        float(form_data["SleepTime"]),
        float(form_data["MentalHealth"]), 
        float(form_data["BMI"])
    ]

    # Store the user data in the database
    store_user_data_in_database(decoded_data, predicted_prob)

    return render_template("result.html", name=name, predicted_prob=predicted_prob)


def store_user_data_in_database(user_data, predicted_prob):
    # Create a new connection for the current thread
    conn = get_db()

    # Retrieve the user ID from the session
    user_id = session.get('id')

    # Insert user data into the prediction_history table
    conn.execute('''
        INSERT INTO prediction_history (user_id, prediction_time, Smoking, AlcoholDrinking, Stroke, DiffWalking, Sex, AgeCategory,
        Race, Diabetic, PhysicalActivity, GenHealth, Asthma, KidneyDisease, SkinCancer, PhysicalHealth, SleepTime,
        MentalHealth, BMI, predicted_result)
        VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, user_data[0], user_data[1], user_data[2], user_data[3], user_data[4], user_data[5], user_data[6],
          user_data[7], user_data[8], user_data[9], user_data[10], user_data[11], user_data[12], user_data[13],
          user_data[14], user_data[15], user_data[16], predicted_prob))

    # Commit the changes
    conn.commit()

    # Close the connection
    conn.close()


if __name__ == "__main__":
    app.run(debug=True)