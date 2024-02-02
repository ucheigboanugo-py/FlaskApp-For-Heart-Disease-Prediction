from flask import Flask, render_template, request
from app.services.model_utils import predict_with_visualizations, generate_prediction_chart
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

app = Flask(__name__, static_folder='static')

model_path = 'C:\\Users\\User\\Desktop\\heart_disease_prediction\\heart_disease_model.pkl'
encoder_path = 'C:\\Users\\User\\Desktop\\heart_disease_prediction\\heart_disease_prediction_encoder.pkl'

# Load the trained model
model = joblib.load(r'heart_disease_model.pkl')

# Load the encoder
encoder = joblib.load(r'heart_disease_prediction_encoder.pkl')

# Load your feature engineered dataset (replace 'path/to/your/dataset.csv' with the actual path)
your_data = pd.read_csv('C:\\Users\\User\\Desktop\\heart_disease_feature_engineering_intermediate_results.csv')

def handle_missing_values(value_str, strategy='mean'):
    if pd.isnull(value_str):
        imputer = SimpleImputer(strategy=strategy)
        return imputer.fit_transform([[0]])[0][0]
    cleaned_value = value_str.strip()
    return float(cleaned_value) if cleaned_value else 0.0

def clean_float(value_str):
    if value_str is not None:
        try:
            # Apply handle_missing_values first
            cleaned_value = handle_missing_values(value_str)
            
            # Check if the value is 'Up' and replace it with a numeric value
            if cleaned_value.lower() == 'up':
                return 1.0

            # Use str.strip() to remove unnecessary characters
            cleaned_value = cleaned_value.strip()
            return pd.to_numeric(cleaned_value)
        except ValueError:
            return 0.0  # Default value if the string is not convertible
    return 0.0  # Default value if the string is None

# Function to get age group
def get_age_group(age):
    if age < 30:
        return "Young"
    elif 30 <= age < 60:
        return "Adult"
    else:
        return "Senior"

# Function to get blood pressure category
def get_blood_pressure_category(restingBP):
    if restingBP < 120:
        return 0 # Representing 'Normal'
    elif 120 <= restingBP < 140:
        return 1 # Representing 'Elevated'
    else:
        return 2 # Representing 'High'

# Function to calculate age max HR interaction
def calculate_age_max_hr_interaction(age, maxHR):
    return age * maxHR

# Function to calculate average exercise HR
def calculate_avg_exercise_hr(exerciseAngina, maxHR):
    if exerciseAngina == 'Y':
        return maxHR - 10
    else:
        return maxHR + 5

# Function to check if chest pain is asymptomatic
def is_chest_pain_asy(chestPainType):
    return chestPainType == 'ASY'

# Function to check if chest pain is atypical
def is_chest_pain_ata(chestPainType):
    return chestPainType == 'ATA'

# Function to check if chest pain is non-anginal pain
def is_chest_pain_nap(chestPainType):
    return chestPainType == 'NAP'

# Function to check if chest pain is typical angina
def is_chest_pain_ta(chestPainType):
    return chestPainType == 'TA'

# Function to check if fasting blood sugar is 1
def is_fasting_bs_1(fastingBS):
    return fastingBS == 1

# Function to check if resting ECG is left ventricular hypertrophy
def is_resting_ecg_lv_h(restingECG):
    return restingECG == 'LVH'

# Function to check if resting ECG is normal
def is_resting_ecg_normal(restingECG):
    return restingECG == 'Normal'

# Function to check if resting ECG is showing ST-T wave abnormality
def is_resting_ecg_st(restingECG):
    return restingECG == 'ST'

# Function to check if exercise-induced angina is absent
def is_exercise_angina_n(exerciseAngina):
    return exerciseAngina == 'N'

# Function to check if exercise-induced angina is present
def is_exercise_angina_y(exerciseAngina):
    return exerciseAngina == 'Y'

# Function to encode categorical variables using manual mapping
def encode_categorical_features(age_group, blood_pressure_category, chest_pain_type, exercise_angina_n, exercise_angina_y, restingECG, fastingBS):
    # Define mappings for each categorical variable
    age_group_mapping = {'Young': 0, 'Adult': 1, 'Senior': 2}
    blood_pressure_mapping = {'Normal': 0, 'Elevated': 1, 'High': 2}
    chest_pain_mapping = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
    exercise_angina_mapping = {False: 0, True: 1}

    # Map categorical values to numerical representations
    age_group_encoded = age_group_mapping.get(age_group, 0)
    blood_pressure_encoded = blood_pressure_mapping.get(blood_pressure_category, 0)
    chest_pain_encoded = chest_pain_mapping.get(chest_pain_type, 0)
    exercise_angina_n_encoded = exercise_angina_mapping.get(exercise_angina_n, 0)
    exercise_angina_y_encoded = exercise_angina_mapping.get(exercise_angina_y, 0)

    # Add mappings for the remaining categorical features
    chestPainASY_encoded = 1 if chest_pain_type == 'ASY' else 0
    chestPainATA_encoded = 1 if chest_pain_type == 'ATA' else 0
    chestPainNAP_encoded = 1 if chest_pain_type == 'NAP' else 0
    chestPainTA_encoded = 1 if chest_pain_type == 'TA' else 0
    fastingBS0_encoded = 1 if fastingBS == 0 else 0
    fastingBS1_encoded = 1 if fastingBS == 1 else 0
    restingECGLVH_encoded = 1 if restingECG == 'LVH' else 0
    restingECGNormal_encoded = 1 if restingECG == 'Normal' else 0
    restingECGST_encoded = 1 if restingECG == 'ST' else 0
    exerciseAnginaN_encoded = 1 if exercise_angina_n == 'N' else 0
    exerciseAnginaY_encoded = 1 if exercise_angina_y == 'Y' else 0

    # Return the encoded values as a list
    return [age_group_encoded, blood_pressure_encoded, chest_pain_encoded, exercise_angina_n_encoded, exercise_angina_y_encoded,
            chestPainASY_encoded, chestPainATA_encoded, chestPainNAP_encoded, chestPainTA_encoded, fastingBS0_encoded,
            fastingBS1_encoded, restingECGLVH_encoded, restingECGNormal_encoded, restingECGST_encoded,
            exerciseAnginaN_encoded, exerciseAnginaY_encoded]


# Function to load the encoder
def load_encoder():
    encoder_path = 'heart_disease_prediction_encoder.pkl'  # Adjust the path if needed
    try:
        encoder = joblib.load(encoder_path)
        return encoder
    except Exception as e:
        print(f"Error loading encoder: {e}")
        return None
    
# Function to encode categorical features
def encode_categorical_features(age_group, blood_pressure_category, chest_pain_type, exercise_angina_n, exercise_angina_y, restingECG, fastingBS):
    # Define mappings for each categorical variable
    mappings = {
        'age_group': {'Young': 0, 'Adult': 1, 'Senior': 2},
        'blood_pressure': {'Normal': 0, 'Elevated': 1, 'High': 2},
        'chest_pain': {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3},
        'exercise_angina': {False: 0, True: 1},
    }

    # Initialize the encoded values list
    encoded_values = []

    # Loop through each categorical feature and encode
    for feature, value in zip(['age_group', 'blood_pressure', 'chest_pain', 'exercise_angina'],
                              [age_group, blood_pressure_category, chest_pain_type, exercise_angina_n]):
        mapping = mappings.get(feature, {})
        encoded_values.append(mapping.get(value, 0))

    # Add mappings for the remaining categorical features
    for feature, value in zip(['chest_pain_type', 'fastingBS', 'restingECG', 'exercise_angina_n', 'exercise_angina_y'],
                              [chest_pain_type, fastingBS, restingECG, exercise_angina_n, exercise_angina_y]):
        if feature == 'chest_pain_type':
            for cp_type in ['ASY', 'ATA', 'NAP', 'TA']:
                encoded_values.append(1 if value == cp_type else 0)
        elif feature == 'fastingBS':
            encoded_values.append(1 if value == 0 else 0)
            encoded_values.append(1 if value == 1 else 0)
        elif feature == 'restingECG':
            for ecg_type in ['LVH', 'Normal', 'ST']:
                encoded_values.append(1 if value == ecg_type else 0)
        elif feature == 'exercise_angina_n':
            encoded_values.append(1 if value == 'N' else 0)
        elif feature == 'exercise_angina_y':
            encoded_values.append(1 if value == 'Y' else 0)

    return encoded_values

    # Declare input_features at the beginning
    input_features = []  

@app.route('/')
def home():
    return render_template('heart_disease_prediction.html')

# Example test features and labels
X_test_numerical = [
    [63, 1, 145, 233, 150, 0, 2, 150, 0, 2.3, 3],
    [67, 1, 160, 286, 108, 1, 1, 108, 1, 1.5, 2],
    # Add more feature vectors for other patients...
]

# Example categorical features
X_test_categorical = [
    ['Adult', 'Normal', False, True, False, True, True, True, False, False, True, False, True, False, True, False],
    ['Senior', 'High', True, False, True, False, False, False, True, False, False, False, True, False, True, False],
    # Add more lists for other patients...
]

# Combine numerical and categorical features
X_test_combined = [num_features + cat_features for num_features, cat_features in zip(X_test_numerical, X_test_categorical)]

y_test = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]  # Corresponding labels

def predict_with_visualizations(model, input_features):
    
    # Assuming model.predict() is the method to make predictions
    predictions = model.predict(input_features)
    
    # Your code for creating visualizations using predictions and y_test
    # ...
    # Sample code for creating visualizations
    def create_visualizations(predictions, y_test):
        # Assuming predictions and y_test are numpy arrays or pandas series

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('static/confusion_matrix.png')
        plt.close()

        # Additional visualizations can be added based on your requirements
        # ...

        
    # Return the paths to the generated visualizations
    visualization_data = create_visualizations(predictions, y_test)
    return predictions, visualization_data
# Declare input_features and encoded_features as global variables
input_features = []
encoded_features = []

# Define the encode_categorical_features function outside of main_prediction
def encode_categorical_features(encoder, feature_name, feature_value):
    df = pd.DataFrame({feature_name: [feature_value]})
    encoded_feature = pd.get_dummies(df, columns=[feature_name], prefix=[feature_name])
    return encoded_feature.values

# Replace the placeholder route with your prediction logic
@app.route('/main_prediction', methods=['GET', 'POST'])
def main_prediction():
    global encoded_features
    if request.method == 'POST':
        # Load your feature engineered dataset (replace 'path/to/your/dataset.csv' with the actual path)
        your_data = pd.read_csv('C:\\Users\\User\\Desktop\\heart_disease_feature_engineering_intermediate_results.csv')

        # Extract feature values from the form
        age = int(request.form.get('age'))
        sex = request.form.get('sex')
        # Convert sex to an integer
        sex_mapping = {'M': 0, 'F': 1}
        sex = sex_mapping.get(sex, 0)  # Default to 0 if not found
        restingBP = int(request.form.get('restingBP'))
        cholesterol = int(request.form.get('cholesterol'))
        fastingBS = int(request.form.get('fastingBS'))
        restingECG = request.form.get('restingECG')
        # Convert restingECG to an integer
        resting_ecg_mapping = {'Normal': 0, 'ST': 1, 'LVH': 2}
        restingECG = resting_ecg_mapping.get(restingECG, 0)  # Default to 0 if not found

        maxHR = int(request.form.get('maxHR'))
        exerciseAngina = request.form.get('exerciseAngina')
        # Convert exerciseAngina to an integer
        exercise_angina_mapping = {'N': 0, 'Y': 1}
        exerciseAngina = exercise_angina_mapping.get(exerciseAngina, 0)  # Default to 0 if not found

        oldPeak_str = request.form.get('oldPeak')
        # Apply handle_missing_values
        oldPeak = handle_missing_values(oldPeak_str)

        stSlope_str = request.form.get('stSlope')
        stSlope_cleaned = clean_float(stSlope_str)
        st_slope_mapping = {'Flat': 0, 'Up': 1, 'Down': 2}  # Adjust as needed
        stSlope = st_slope_mapping.get(stSlope_str, 0)  # Default to 0 if not found

        chestPainType = request.form.get('chestPainType')
        chest_pain_type_mapping = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
        chestPainType = chest_pain_type_mapping.get(chestPainType, 0)  # Default to 0 if not found

        AgeGroup = get_age_group(age)
        bloodPressureCategory = get_blood_pressure_category(restingBP)
        ageMaxHrInteract = calculate_age_max_hr_interaction(age, maxHR)
        avgExerciseHr = calculate_avg_exercise_hr(exerciseAngina, maxHR)
        chestPainASY = is_chest_pain_asy(chestPainType)
        chestPainATA = is_chest_pain_ata(chestPainType)
        chestPainNAP = is_chest_pain_nap(chestPainType)
        chestPainTA = is_chest_pain_ta(chestPainType)
        fastingBS1 = is_fasting_bs_1(fastingBS)
        restingECGLVH = is_resting_ecg_lv_h(restingECG)
        restingECGNormal = is_resting_ecg_normal(restingECG)
        restingECGST = is_resting_ecg_st(restingECG)
        exerciseAnginaN = is_exercise_angina_n(exerciseAngina)
        exerciseAnginaY = is_exercise_angina_y(exerciseAngina)

        # Load the encoder
        encoder = load_encoder()
        
        # Encode categorical features
        age_group_encoded = encode_categorical_features(encoder, 'AgeGroup', AgeGroup)
        blood_pressure_category_encoded = encode_categorical_features(encoder, 'BloodPressureCategory', bloodPressureCategory)
        chest_pain_type_encoded = encode_categorical_features(encoder, 'ChestPainType', chestPainType)
        exercise_angina_N_encoded = encode_categorical_features(encoder, 'ExerciseAnginaN', exerciseAnginaN)
        exercise_angina_Y_encoded = encode_categorical_features(encoder, 'ExerciseAnginaY', exerciseAnginaY)
        resting_ecg_encoded = encode_categorical_features(encoder, 'RestingECG', restingECG)
        fasting_bs_encoded = encode_categorical_features(encoder, 'FastingBS', fastingBS)
        
        # Combine numerical and encoded categorical features
        input_features = [
            age, sex, restingBP, cholesterol, maxHR, oldPeak, stSlope_cleaned,  # Numerical features
            age_group_encoded, blood_pressure_category_encoded,  # Encoded categorical features
            chest_pain_type_encoded, exercise_angina_N_encoded, exercise_angina_Y_encoded, resting_ecg_encoded, fasting_bs_encoded  # Additional encoded categorical features
        ]
        
        # Debugging: Check the shape and type of each element in input_features
        for feature in input_features:
            feature_shape = np.array(feature).shape
            feature_type = type(feature)
            print("Shape:", feature_shape, "Type:", feature_type)

        # Perform model prediction using input_features
        print(type(model))
        predictions = model.predict(np.array([input_features]))

        # Print the number of features expected by the model
        num_features_expected = len(model.feature_importances_)
        print("Number of Features Expected by the Model:", num_features_expected)

        # Identify missing features
        missing_features = set(['AgeGroup', 'BloodPressureCategory', 'Age_MaxHR_Interact', 'AvgExerciseHR', 'ChestPain_ASY', 'ChestPain_ATA', 'ChestPain_NAP', 'ChestPain_TA', 'FastingBS_0', 'FastingBS_1', 'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N', 'ExerciseAngina_Y']) - set(input_features)

        # Add missing features to input_features
        input_features += [your_data[feature] for feature in missing_features]

        # Add the encoded features to input_features
        input_features += encoded_features

        # Check the number of features in the input_features
        num_features_input = len(input_features)
        print("Number of Features in input_features:", num_features_input)

        # Verify if the number of features matches the model's expectation
        if num_features_input != num_features_expected:
            print("Number of features in input_features does not match the model's expectation. Please check your feature encoding.")

        # Print the shapes for debugging
        print("Shape of input_features before:", input_features.shape)

        # Convert input_features to a 2D array (reshape)
        input_features = np.array([input_features])  # Ensure it's a 2D array
        
        # Reshape input_features if necessary
        if len(input_features.shape) == 3:
            input_features = input_features.squeeze()


        # Print the shapes for debugging
        print("Shape of input_features before:", input_features.shape)

        # Make predictions
        prediction, visualization_data = predict_with_visualizations(model, input_features)

        # Include any further processing or response generation here

    # Return the response
    return response

    # Define the function to generate visualization data
def generate_visualization_data():
    # Example visualization_data with extended 'Age' categories and 'Sex' values
    visualization_data = {
        'Age': ['25-29', '25-29', '30-34', '30-34', '35-39', '35-39', '40-44', '40-44', 
                '45-49', '45-49', '50-54', '50-54', '55-59', '55-59', '60-64', '60-64', 
                '65-69', '65-69', '70-74', '70-74', '75-79', '75-79'],  
        'Sex': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female',
                'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 
                'male', 'female', 'male', 'female', 'male', 'female'], 
        'ChestPainType': ['NAP', 'ATA', 'ASY', 'TA', 'NAP', 'ATA', 'ASY', 'TA', 'NAP', 
                          'ATA', 'ASY'],
        'RestingBP': [],
        'Cholesterol': [],
        'FastingBS': [],
        'RestingECG': [],
        'MaxHR': [],
        'ExerciseAngina': [],
        'Oldpeak': [],
        'ST_Slope': [],
        'AgeGroup': [],
        'BloodPressureCategory': [],
        'AgeMaxHrInteract': [],
        'AvgExerciseHR': [],
        'ChestPainASY': [],
        'ChestPainATA': [],
        'ChestPainNAP': [],
        'ChestPainTA': [],
        'FastingBS0': [],
        'FastingBS1': [],
        'RestingECGLVH': [],
        'RestingECGNormal': [],
        'RestingECGST': [],
        'ExerciseAnginaN': [],
        'ExerciseAnginaY': []
        # Add more visualization data as needed
    }
    return visualization_data

# Call the function to generate visualization data
visualization_data = generate_visualization_data()

# Now you can use the visualization data in your code
chart_path = generate_prediction_chart(visualization_data)

# Load your testing data from a CSV file (replace 'path/to/your/testing_data.csv' with the actual path)
testing_data = pd.read_csv('C:\\Users\\User\\Desktop\\heart_disease_feature_engineering_intermediate_results.csv')

# Extract features (X_test) and labels (y_test) from the testing data
X_test = testing_data.drop('HeartDisease', axis=1)  # Replace 'target_column' with the actual label column name
y_test = testing_data['HeartDisease']  # Replace 'target_column' with the actual label column name

# Convert X_test and y_test to numpy arrays
X_test = X_test.values
y_test = y_test.values

# Pass X_test and y_test to the predict_with_visualizations function
prediction, visualization_data = predict_with_visualizations(X_test, y_test)

# Generate prediction chart
chart_path = generate_prediction_chart(visualization_data)

# Render the results page with the prediction and chart path
@app.route('/main_prediction', methods=['GET', 'POST'])
def main_prediction():
    # Your prediction logic here
    return render_template('results.html', prediction=prediction, chart_path=chart_path)

# Return the heart_disease_prediction.html page if the request method is GET
@app.route('/')
def index():
    return render_template('heart_disease_prediction.html')


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)









