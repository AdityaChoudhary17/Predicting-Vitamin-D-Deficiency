import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

class CustomNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_hidden = np.random.randn(input_size, hidden_size) * 0.01 
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid_derivative(self, a): 
        return a * (1 - a)

    def relu_derivative(self, a): 
        return (a > 0).astype(float)

    def forward(self, X):
    
        self.hidden_layer_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.hidden_layer_output = self.relu(self.hidden_layer_input) 

        # Output layer
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_output) + self.bias_output
        self.output_layer_output = self.sigmoid(self.output_layer_input) 

        return self.output_layer_output

    def backward(self, X, y, output, learning_rate):
    
        output_error = output - y 
        output_delta = output_error * self.sigmoid_derivative(output) 

        hidden_layer_error = np.dot(output_delta, self.weights_output.T)
        hidden_layer_delta = hidden_layer_error * self.relu_derivative(self.hidden_layer_output)

        self.weights_output -= learning_rate * np.dot(self.hidden_layer_output.T, output_delta)
        self.bias_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_hidden -= learning_rate * np.dot(X.T, hidden_layer_delta)
        self.bias_hidden -= learning_rate * np.sum(hidden_layer_delta, axis=0, keepdims=True)

    def train(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
    
            output = self.forward(X_train)


            self.backward(X_train, y_train, output, learning_rate)


            loss = self.binary_cross_entropy_loss(output, y_train)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        probabilities = self.forward(X)
        predictions = (probabilities > 0.5).astype(int)
        return predictions

    def binary_cross_entropy_loss(self, y_predicted, y_true):
    
        m = len(y_true) 
        loss = -1/m * np.sum(y_true * np.log(y_predicted) + (1 - y_true) * np.log(1 - y_predicted + 1e-8))
        return loss
try:
    data = pd.read_csv("augmented_vitamin_d_dataset.csv")
except FileNotFoundError:
    st.error("Please upload the 'augmented_vitamin_d_dataset.csv' file.")
    st.stop()

# Data Preprocessing (same as before)
data.rename(columns={'Vitamin_D_Deficient': 'Vitamin_D_Status'}, inplace=True)
data['Sex'] = data['Sex'].astype('category')

X = data.drop('Vitamin_D_Status', axis=1)
y = data['Vitamin_D_Status']

X['BMI'] = pd.to_numeric(X['BMI'], errors='coerce')
X.dropna(subset=['BMI'], inplace=True)
y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['Sex']
numerical_features = ['Age', 'Weight', 'Height', 'BMI', 'WC', 'BF', 'BM', 'Exercise', 'Sunlight']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

X_train_processed = preprocessor.fit_transform(X_train) 
X_test_processed = preprocessor.transform(X_test) 

# --- Train your Custom Model ---
input_size = X_train_processed.shape[1] 
hidden_size = 10 
output_size = 1 

custom_model = CustomNeuralNetwork(input_size, hidden_size, output_size)
custom_model.train(X_train_processed, y_train.values.reshape(-1, 1), epochs=1000, learning_rate=0.01) 



y_pred_custom_nn = custom_model.predict(X_test_processed)
accuracy_custom_nn = accuracy_score(y_test, y_pred_custom_nn)
classification_rep_custom_nn = classification_report(y_test, y_pred_custom_nn, zero_division=0, output_dict=True)
conf_matrix_custom_nn = confusion_matrix(y_test, y_pred_custom_nn)



st.set_page_config(page_title="Vitamin D Predictor App", page_icon="☀️")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap');

    body {
        font-family: 'Poppins', sans-serif;
        color: #333;
        background-color: #f0f8ff; /* Light background color - Alice Blue */
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif;
        color: #4a7fbb; /* A nice heading color */
    }

    .stButton>button {
        color: white;
        background-color: #4a7fbb;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        cursor: pointer;
    }

    .stButton>button:hover {
        background-color: #5e95d4;
    }

    .sidebar .stNumberInput>label,
    .sidebar .stSelectbox>label,
    .sidebar .stTextInput>label {
        color: #333; /* Keep sidebar labels readable */
    }

    .streamlit-expanderHeader {
        font-weight: bold;
        color: #4a7fbb;
    }

    .streamlit-expanderContent {
        color: #555;
    }

    .report-table th {
        background-color: #e9ecef;
        font-weight: bold;
        padding: 0.75rem;
        text-align: left;
    }

    .report-table td {
        padding: 0.75rem;
        border-top: 1px solid #dee2e6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title('☀️ Vitamin D Deficiency Predictor')
st.markdown("Is sunshine enough? 🧐 Find out your potential Vitamin D deficiency risk with this predictor, powered by a Neural Network.")
st.write("---") # Separator line

with st.expander("ℹ️ How to use this app"): # Guidance in an expander
    st.write("""
        **Follow these steps to get your Vitamin D status prediction:**

        1.  **Enter Your Information:** Use the sidebar to the left to input your personal details.
        2.  **Click 'Predict':** Once you've filled in the details, click the 'Predict Vitamin D Status (Custom Model)' button in the sidebar.
        3.  **View Results:** The prediction result and model evaluation metrics will be displayed below.

        **Important Note:** This tool provides a prediction and is for informational purposes only. It is not a substitute for professional medical advice.
    """)
st.write("---") # Separator line

# --- Improved Sidebar with Information and Styling ---
st.sidebar.header("🧍 Enter Your Information")
st.sidebar.markdown("Provide your details to get a Vitamin D deficiency prediction.") # Added subtitle
age = st.sidebar.number_input("Age", min_value=18, max_value=100, help="Your age in years.") # Removed value=30
sex_option = st.sidebar.selectbox("Sex", options=['M', 'F'], help="Your biological sex.") # Removed default 'M'
weight = st.sidebar.number_input("Weight (kg)", min_value=40.0, max_value=150.0, step=0.5, format="%.1f", help="Your weight in kilograms.") # Removed value=70.0
height = st.sidebar.number_input("Height (cm)", min_value=140, max_value=210, help="Your height in centimeters.") # Removed value=170
bmi = st.sidebar.number_input("BMI", min_value=18.5, max_value=50.0, step=0.1, format="%.1f", help="Your Body Mass Index (BMI). You can calculate it online if you don't know it.") # Removed value=25.0
wc = st.sidebar.number_input("Waist Circumference (cm)", min_value=50, max_value=150, help="Your waist circumference measured at the narrowest point, in centimeters.") # Removed value=80
bf = st.sidebar.number_input("Body Fat (%)", min_value=5.0, max_value=50.0, step=0.1, format="%.1f", help="Your body fat percentage.  Estimate or use a body fat scale.") # Removed value=25.0
bm = st.sidebar.number_input("Bone Mass (kg)", min_value=1.0, max_value=40.0, step=0.1, format="%.1f", help="Your bone mass in kilograms (if known, often from body composition analysis).") # Removed value=25.0
exercise = st.sidebar.number_input("Exercise (hours/week)", min_value=0.0, max_value=20.0, step=0.5, format="%.1f", help="Average hours of moderate to vigorous exercise per week.") # Removed value=2.0
sunlight = st.sidebar.number_input("☀️ Sunlight Exposure (hours/day)", min_value=0.0, max_value=10.0, step=0.1, format="%.1f", help="Average hours of direct sunlight exposure per day (without sunscreen, face and arms exposed).") # Removed value=1.0
st.sidebar.write("---") 

feature_ranges = {
    'Age': {'Very Good': (18, 40), 'Good': (41, 60), 'Less Focus': (61, 75), 'Immediate Focus': (76, 100)},
    'BMI': {'Very Good': (18.5, 24.9), 'Good': (25, 29.9), 'Less Focus': (30, 34.9), 'Immediate Focus': (35, 50)},
    'WC': {'Very Good': (50, 94), 'Good': (95, 102), 'Less Focus': (103, 110), 'Immediate Focus': (111, 150)}, 
    'BF': {'Very Good': (5, 18), 'Good': (19, 25), 'Less Focus': (26, 30), 'Immediate Focus': (31, 50)}, 
    'BM': {'Very Good': (20, 40), 'Good': (15, 19.9), 'Less Focus': (10, 14.9), 'Immediate Focus': (1, 9.9)}, 
    'Exercise': {'Very Good': (5, 20), 'Good': (3, 4.9), 'Less Focus': (1, 2.9), 'Immediate Focus': (0, 0.9)},
    'Sunlight': {'Very Good': (2, 10), 'Good': (1, 1.9), 'Less Focus': (0.5, 0.9), 'Immediate Focus': (0, 0.4)}
    
}

range_categories_ordered = ['Immediate Focus', 'Less Focus', 'Good', 'Very Good'] # Order for progress bar color mapping


def get_feature_category(feature_name, value):
    ranges = feature_ranges.get(feature_name)
    if ranges:
        for category, (min_val, max_val) in ranges.items():
            if min_val <= value <= max_val:
                return category
    return 'Unknown' # Or handle default category if needed


# Prediction button
if st.sidebar.button('✨ Predict Vitamin D Status (Custom Model) ✨'):
    input_data_streamlit = pd.DataFrame([{
        'Age': age,
        'Sex': sex_option,
        'Weight': weight,
        'Height': height,
        'BMI': bmi,
        'WC': wc,
        'BF': bf,
        'BM': bm,
        'Exercise': exercise,
        'Sunlight': sunlight
    }])

    input_data_processed = preprocessor.transform(input_data_streamlit) 

    prediction_custom_nn = custom_model.predict(input_data_processed)[0][0] 

    status_label = {0: "Not Likely Deficient", 1: "Likely Deficient", 2: "Potentially Deficient", 3: "Unknown Status" } # Assuming 0, 1, 2 are your status labels
    status_labels_list = list(status_label.values()) 

    st.header("🔮 Prediction Result (Custom Model)") 
    predicted_status = status_label.get(prediction_custom_nn, 'Unknown')

    if predicted_status == "Likely Deficient" or predicted_status == "Potentially Deficient":
        st.error(f"⚠️ Predicted Vitamin D Status: **{predicted_status}** ⚠️", icon="⚠️")    
        st.write("It's recommended to consult with a healthcare professional to check your Vitamin D levels and discuss appropriate next steps.")
        st.markdown("""
            **What to do if you are potentially Vitamin D deficient:**
            *   **Consult a Doctor:** The most important step is to get a blood test to accurately measure your Vitamin D levels.
            *   **Discuss Supplementation:** If confirmed deficient, your doctor can recommend the correct Vitamin D dosage for supplementation.
            *   **Dietary Sources:** Include Vitamin D rich foods in your diet such as fatty fish (salmon, mackerel, tuna), egg yolks, and fortified foods (milk, cereals).
            *   **Safe Sunlight Exposure:**  When appropriate and safe for your skin type and health conditions, get some sunlight exposure.  Remember to avoid sunburn.
            *   **Re-test:** After implementing changes, follow up with your doctor for re-testing to ensure your Vitamin D levels are improving.
        """)

    else:
        st.success(f"✅ Predicted Vitamin D Status: **{predicted_status}** ✅", icon="✅") 
        st.write("Continue maintaining a healthy lifestyle with balanced diet and adequate sunlight exposure.")
        st.balloons() 

    st.write("---") 


    st.subheader("📊 Your Input Feature Ranges")
    st.markdown("Below is a visualization of where your entered values fall within defined health ranges. This gives you a quick overview of areas that might need attention.")

    feature_values = {
        'Age': age,
        'BMI': bmi,
        'WC': wc,
        'BF': bf,
        'BM': bm,
        'Exercise': exercise,
        'Sunlight': sunlight
    }

    for feature_name, value in feature_values.items():
        category = get_feature_category(feature_name, value)
        category_index = range_categories_ordered.index(category) if category in range_categories_ordered else -1 
        progress_color = ['red', 'orange', 'blue', 'green'][category_index] if 0 <= category_index < 4 else 'gray' 
        percentage = (category_index + 1) / len(range_categories_ordered) if 0 <= category_index < 4 else 0 

        col1, col2 = st.columns([1, 3]) 
        with col1:
            st.metric(label=feature_name, value=value) 
        with col2:
            st.progress(percentage, text=f"{feature_name} Range: {category}") 

    st.write("---") 


    st.subheader("📊 Model Evaluation Metrics (Custom NN on Test Data)")

   
    st.markdown("#### Classification Report Explained")
    st.write(
        """
        The classification report provides a detailed look at the model's performance for each class (Vitamin D status). Here's a quick explanation of the metrics:

        *   **Precision:**  Out of all the instances the model predicted as a certain class (e.g., 'Likely Deficient'), what proportion was actually correct? High precision means fewer false positives.
        *   **Recall:** Out of all the actual instances of a certain class, what proportion did the model correctly identify? High recall means fewer false negatives.
        *   **F1-score:** The harmonic mean of precision and recall, providing a balanced measure of accuracy for each class.
        *   **Support:** The number of actual instances of each class in the test dataset.

        The 'accuracy' is the overall accuracy of the model across all classes. 'macro avg' is the average of the metrics across all classes (unweighted), and 'weighted avg' is the average weighted by the number of instances in each class.
        """
    )
    st.write("**Detailed Classification Report (Custom NN):**")
    report_df = pd.DataFrame(classification_rep_custom_nn).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Blues'), width=800) 
   


    
    st.markdown("#### Confusion Matrix")
    st.write("This table shows the counts of:")
    st.write("- **True Positives (TP):** Model correctly predicted the class.")
    st.write("- **True Negatives (TN):** (Not directly shown in this matrix for multi-class, but implied in correctly classified other classes)")
    st.write("- **False Positives (FP):** Model incorrectly predicted this class (Type I error).")
    st.write("- **False Negatives (FN):** Model incorrectly predicted another class when it should have been this class (Type II error).")

    
    st.write("--- **Debugging Confusion Matrix Table Creation** ---")
    st.write(f"**Type of conf_matrix_custom_nn:** {type(conf_matrix_custom_nn)}")
    st.write(f"**Content of conf_matrix_custom_nn:**")
    st.write(conf_matrix_custom_nn)
    st.write(f"**Status Labels List (status_labels_list):** {status_labels_list}")

    
    cm_df = pd.DataFrame(conf_matrix_custom_nn,
                         index=[f"Actual {l}" for l in status_labels_list],
                         columns=[f"Predicted {l}" for l in status_labels_list]) 

    st.write(f"**Type of cm_df:** {type(cm_df)}") 
    st.write("**Content of cm_df (DataFrame):**")
    st.write(cm_df) 

    st.dataframe(cm_df.style.background_gradient(cmap='Blues'), width=800) 
    st.write(f"**Overall Accuracy (Custom NN):** {accuracy_custom_nn:.2f}")


    st.warning("⚠️ Disclaimer: This prediction is for informational purposes only and is NOT medical advice. Always consult with a healthcare professional for diagnosis and treatment.") # More prominent disclaimer
else:
    st.info("⬅️ Enter your details in the sidebar and click the 'Predict Vitamin D Status (Custom Model)' button to get your personalized prediction.") # Guidance to user
