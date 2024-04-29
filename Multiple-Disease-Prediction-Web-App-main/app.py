# -*- coding: utf-8 -*-
"""

@author: AK
"""
import pickle
import streamlit as st
from streamlit_option_menu import option_menu 
from sklearn.preprocessing import LabelEncoder




# Loading the saved models
diabetes_model = pickle.load(open("./models/diabetes_model_new.sav",'rb'))
heart_model = pickle.load(open("./models/heart_disease_model.sav",'rb'))
parkinsons_model = pickle.load(open("./models/parkinsons_model.sav",'rb'))
breast_model = pickle.load(open("./models/breast_cancer_model.sav",'rb'))

# Sidebar navigation
with st.sidebar:   
    selected = option_menu('Multiple Disease Prediction System', 
                           ['Heart Disease Prediction',
                            'Diabetes Prediction',
                            "Parkinson's Prediction",
                            'Breast Cancer Prediction'],
                           icons=['heart','activity','person','gender-female'],
                           default_index=0)





# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction': 
    # Page title
    st.title('Heart Disease Prediction using ML')

    # Define a label encoder for 'Sex'
    sex_label_encoder = LabelEncoder()
    sex_label_encoder.fit(['Male', 'Female'])

    # Define help information for each feature
    help_info = {
        'Age': 'The age of the patient.',
        'Sex': 'The gender of the patient.',
        'Chest Pain types': 'The type of chest pain experienced by the patient.',
        'Resting Blood Pressure': 'The resting blood pressure of the patient.',
        'Serum Cholestoral in mg/dl': 'The serum cholesterol level of the patient.',
        'Fasting Blood Sugar > 120 mg/dl': 'Whether the fasting blood sugar is greater than 120 mg/dl.',
        'Resting Electrocardiographic results': 'The results of resting electrocardiographic measurements.',
        'Maximum Heart Rate achieved': 'The maximum heart rate achieved by the patient.',
        'Exercise Induced Angina': 'Whether exercise induced angina is present.',
        'ST depression induced by exercise': 'The ST depression induced by exercise relative to rest.',
        'Slope of the peak exercise ST segment': 'The slope of the peak exercise ST segment.',
        'Major vessels colored by flourosopy': 'The number of major vessels colored by flourosopy.',
        'thal: 1 = normal; 2 = fixed defect; 3 = reversible defect': 'The thalassemia type.'
    }

    # Define default values for each feature
    default_values = {
        'Age': 40,
        'Sex': 'Male',
        'Chest Pain types': 1,
        'Resting Blood Pressure': 120,
        'Serum Cholestoral in mg/dl': 200,
        'Fasting Blood Sugar > 120 mg/dl': 0,
        'Resting Electrocardiographic results': 0,
        'Maximum Heart Rate achieved': 150,
        'Exercise Induced Angina': 0,
        'ST depression induced by exercise': 0.0,
        'Slope of the peak exercise ST segment': 1,
        'Major vessels colored by flourosopy': 0,
        'thal: 1 = normal; 2 = fixed defect; 3 = reversible defect': 2
    }

    # Define options for Sex
    sex_options = ['Male', 'Female']

    # Map default value to index for Sex
    default_sex_index = sex_options.index(default_values['Sex'])

    # Page layout
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Age', min_value=20, max_value=100, value=default_values['Age'], help=help_info['Age'])

    with col2:
        sex = st.selectbox('Sex', options=sex_options, index=default_sex_index, help=help_info['Sex'])

    with col3:
        cp = st.slider('Chest Pain types', min_value=1, max_value=4, value=default_values['Chest Pain types'], help=help_info['Chest Pain types'])

    with col1:
        trestbps = st.slider('Resting Blood Pressure', min_value=90, max_value=200, value=default_values['Resting Blood Pressure'], help=help_info['Resting Blood Pressure'])

    with col2:
        chol = st.slider('Serum Cholestoral in mg/dl', min_value=100, max_value=600, value=default_values['Serum Cholestoral in mg/dl'], help=help_info['Serum Cholestoral in mg/dl'])

    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], index=default_values['Fasting Blood Sugar > 120 mg/dl'], help=help_info['Fasting Blood Sugar > 120 mg/dl'])

    with col1:
        restecg = st.slider('Resting Electrocardiographic results', min_value=0, max_value=2, value=default_values['Resting Electrocardiographic results'], help=help_info['Resting Electrocardiographic results'])

    with col2:
        thalach = st.slider('Maximum Heart Rate achieved', min_value=60, max_value=220, value=default_values['Maximum Heart Rate achieved'], help=help_info['Maximum Heart Rate achieved'])

    with col3:
        exang = st.selectbox('Exercise Induced Angina', options=[0, 1], index=default_values['Exercise Induced Angina'], help=help_info['Exercise Induced Angina'])

    with col1:
        oldpeak = st.slider('ST depression induced by exercise', min_value=0.0, max_value=6.0, value=default_values['ST depression induced by exercise'], step=0.1, help=help_info['ST depression induced by exercise'])

    with col2:
        slope = st.slider('Slope of the peak exercise ST segment', min_value=0, max_value=2, value=default_values['Slope of the peak exercise ST segment'], step=1, help=help_info['Slope of the peak exercise ST segment'])

    with col3:
        ca = st.slider('Major vessels colored by flourosopy', min_value=0, max_value=4, value=default_values['Major vessels colored by flourosopy'], step=1, help=help_info['Major vessels colored by flourosopy'])

    with col1:
        thal = st.slider('thal: 1 = normal; 2 = fixed defect; 3 = reversible defect', min_value=0, max_value=3, value=default_values['thal: 1 = normal; 2 = fixed defect; 3 = reversible defect'], step=1, help=help_info['thal: 1 = normal; 2 = fixed defect; 3 = reversible defect'])

    # Code for Prediction
    heart_diagnosis = ''

    # Button for Prediction
    if st.button('Heart Disease Test Result'):
        input_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        # Convert 'Sex' to numeric format
        sex_numeric = sex_label_encoder.transform([sex])[0]
        input_values[1] = sex_numeric  # Replace 'Sex' with its numeric value

        # Convert input values to numeric format
        input_values_numeric = [float(val) if isinstance(val, str) else val for val in input_values]

        # Predict using the model
        heart_prediction = heart_model.predict([input_values_numeric])

        # Interpret the prediction
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person has a heart disease.'
        else:
            heart_diagnosis = 'The person does not have any heart disease.'

    st.success(heart_diagnosis)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # Page title
    st.title('Diabetes Prediction using ML')
    
    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Input field for number of pregnancies
        pregnancies = st.slider('Pregnancies', min_value=0, max_value=17, step=1, value=6,
                                help="Number of times pregnant. Higher number of pregnancies may indicate a higher risk of diabetes.")
        
        # Input field for glucose level
        glucose = st.slider('Glucose Level', min_value=0, max_value=200, step=1, value=150,
                            help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test. Elevated glucose levels may indicate diabetes.")
        
        # Input field for skin thickness
        skin_thickness = st.slider('Skin Thickness value', min_value=0, max_value=100, step=1, value=35,
                                   help="Triceps skin fold thickness in mm. Thicker skin folds may indicate a higher risk of diabetes.")
        
    with col2:
        # Input field for blood pressure
        blood_pressure = st.slider('Blood Pressure value', min_value=0, max_value=200, step=1, value=80,
                                   help="Diastolic blood pressure (mm Hg). Higher blood pressure levels may indicate a higher risk of diabetes.")
        
        # Input field for insulin level
        insulin = st.slider('Insulin Level', min_value=0, max_value=1000, step=1, value=150,
                            help="2-Hour serum insulin (mu U/ml). Elevated insulin levels may indicate insulin resistance and a higher risk of diabetes.")
        
        # Input field for age
        age = st.slider('Age of the Person', min_value=0, max_value=120, step=1, value=40,
                        help="Age in years. Older age may indicate a higher risk of diabetes.")
        
    with col3:
        # Input field for diabetes pedigree function
        diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function value(0-3)', value='0.6',
                                                   help="A function which scores likelihood of diabetes based on family history. Higher values indicate a higher risk of diabetes.")
        
        # Input field for BMI (Body Mass Index)
        bmi = st.text_input('BMI value(0-50)', value='30.0',
                            help="Body Mass Index (weight in kg / (height in m)^2). Higher BMI values may indicate obesity, which is associated with a higher risk of diabetes.")
        
    # Code for prediction
    diab_diagnosis = ''
    
    # Button for prediction
    if st.button('Diabetes Test Result'):
        if not all([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]):
            st.warning("Please fill in all the fields.")
        else:
            diab_prediction = diabetes_model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
            
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is predicted to have diabetes.'
            else:
                diab_diagnosis = 'The person is predicted to be non-diabetic.'
        
    st.success(diab_diagnosis)

# Parkinson's Prediction Page  
# Parkinson's Prediction Page  
if selected == "Parkinson's Prediction":    
    # Page title
    st.title("Parkinson's Disease Prediction using ML")
    
    # Icon for Help Section
    with st.expander("ℹ️ Click here for Help"):
        st.write("## Input Parameters:")
        st.write("- **MDVP: Fo(Hz):** Average vocal fundamental frequency. Higher values may indicate Parkinson's disease.")
        st.write("- **MDVP: Fhi(Hz):** Maximum vocal fundamental frequency. Higher values may indicate Parkinson's disease.")
        st.write("- **MDVP: Flo(Hz):** Minimum vocal fundamental frequency. Lower values may indicate Parkinson's disease.")
        st.write("- **MDVP: Jitter(%):** Variation in fundamental frequency. Higher values may indicate Parkinson's disease.")
        st.write("- **MDVP: Jitter(Abs):** Absolute variation in fundamental frequency. Higher values may indicate Parkinson's disease.")
        st.write("- **MDVP: RAP:** Relative amplitude perturbation. Higher values may indicate Parkinson's disease.")
        st.write("- **MDVP: PPQ:** Five-point period perturbation quotient. Higher values may indicate Parkinson's disease.")
        st.write("- **Jitter: DDP:** Average absolute difference of differences between consecutive periods. Higher values may indicate Parkinson's disease.")
        st.write("- **MDVP: Shimmer:** Vocal shimmer. Higher values may indicate Parkinson's disease.")
        st.write("- **MDVP: Shimmer(dB):** Decibel level of shimmer. Higher values may indicate Parkinson's disease.")
        st.write("- **Shimmer: APQ3:** Three-point amplitude perturbation quotient for shimmer.")
        st.write("- **Shimmer: APQ5:** Five-point amplitude perturbation quotient for shimmer.")
        st.write("- **MDVP: APQ:** Amplitude perturbation quotient.")
        st.write("- **Shimmer: DDA:** Average absolute differences between amplitudes of consecutive periods.")
        st.write("- **NHR:** Noise-to-harmonics ratio. Higher values may indicate Parkinson's disease.")
        st.write("- **HNR:** Harmonics-to-noise ratio. Lower values may indicate Parkinson's disease.")
        st.write("- **RPDE:** Recurrence period density entropy. Higher values may indicate Parkinson's disease.")
        st.write("- **DFA:** Detrended fluctuation analysis. Higher values may indicate Parkinson's disease.")
        st.write("- **spread1:** Nonlinear measure of fundamental frequency variation. Higher values may indicate Parkinson's disease.")
        st.write("- **spread2:** Nonlinear measure of fundamental frequency variation. Higher values may indicate Parkinson's disease.")
        st.write("- **D2:** Correlation dimension. Higher values may indicate Parkinson's disease.")
        st.write("- **PPE:** Pitch period entropy. Higher values may indicate Parkinson's disease.")
        
        
        
    
    # Sliders for input parameters
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.slider('MDVP: Fo(Hz)', min_value=0.0, max_value=200.0, step=0.01, value=100.0, help="Average vocal fundamental frequency. Higher values may indicate Parkinson's disease.")
        
    with col2:
        fhi = st.slider('MDVP: Fhi(Hz)', min_value=0.0, max_value=400.0, step=0.01, value=200.0, help="Maximum vocal fundamental frequency. Higher values may indicate Parkinson's disease.")
        
    with col3:
        flo = st.slider('MDVP: Flo(Hz)', min_value=0.0, max_value=200.0, step=0.01, value=100.0, help="Minimum vocal fundamental frequency. Lower values may indicate Parkinson's disease.")
        
    with col4:
        jitter_percent = st.slider('MDVP: Jitter(%)', min_value=0.0, max_value=1.0, step=0.01, value=0.5, help="Variation in fundamental frequency. Higher values may indicate Parkinson's disease.")
        
    with col5:
        jitter_abs = st.slider('MDVP: Jitter(Abs)', min_value=0.0, max_value=0.1, step=0.001, value=0.05, help="Absolute variation in fundamental frequency. Higher values may indicate Parkinson's disease.")
        
    with col1:
        rap = st.slider('MDVP: RAP', min_value=0.0, max_value=0.1, step=0.001, value=0.05, help="Relative amplitude perturbation. Higher values may indicate Parkinson's disease.")
        
    with col2:
        ppq = st.slider('MDVP: PPQ', min_value=0.0, max_value=0.1, step=0.001, value=0.05, help="Five-point period perturbation quotient. Higher values may indicate Parkinson's disease.")
        
    with col3:
        ddp = st.slider('Jitter: DDP', min_value=0.0, max_value=0.2, step=0.001, value=0.1, help="Average absolute difference of differences between consecutive periods. Higher values may indicate Parkinson's disease.")
        
    with col4:
        shimmer = st.slider('MDVP: Shimmer', min_value=0.0, max_value=1.0, step=0.01, value=0.5, help="Vocal shimmer. Higher values may indicate Parkinson's disease.")
        
    with col5:
        shimmer_db = st.slider('MDVP: Shimmer(dB)', min_value=0.0, max_value=1.0, step=0.01, value=0.5, help="Decibel level of shimmer. Higher values may indicate Parkinson's disease.")
        
    with col1:
        apq3 = st.slider('Shimmer: APQ3', min_value=0.0, max_value=1.0, step=0.01, value=0.5, help="Three-point amplitude perturbation quotient for shimmer.")
        
    with col2:
        apq5 = st.slider('Shimmer: APQ5', min_value=0.0, max_value=1.0, step=0.01, value=0.5, help="Five-point amplitude perturbation quotient for shimmer.")
        
    with col3:
        apq = st.slider('MDVP: APQ', min_value=0.0, max_value=1.0, step=0.01, value=0.5, help="Amplitude perturbation quotient.")
        
    with col4:
        dda = st.slider('Shimmer: DDA', min_value=0.0, max_value=1.0, step=0.01, value=0.5, help="Average absolute differences between amplitudes of consecutive periods.")
        
    with col5:
        nhr = st.slider('NHR', min_value=0.0, max_value=1.0, step=0.01, value=0.5, help="Noise-to-harmonics ratio. Higher values may indicate Parkinson's disease.")
        
    with col1:
        hnr = st.slider('HNR', min_value=0.0, max_value=30.0, step=0.01, value=15.0, help="Harmonics-to-noise ratio. Lower values may indicate Parkinson's disease.")
        
    with col2:
        rpde = st.slider('RPDE', min_value=0.0, max_value=1.0, step=0.01, value=0.5, help="Recurrence period density entropy. Higher values may indicate Parkinson's disease.")
        
    with col3:
        dfa = st.slider('DFA', min_value=0.0, max_value=1.0, step=0.01, value=0.5, help="Detrended fluctuation analysis. Higher values may indicate Parkinson's disease.")
        
    with col4:
        spread1 = st.slider('spread1', min_value=-10.0, max_value=10.0, step=0.01, value=0.0, help="Nonlinear measure of fundamental frequency variation. Higher values may indicate Parkinson's disease.")
    with col5:
        spread2 = st.slider('spread2', min_value=-10.0, max_value=10.0, step=0.01, value=0.0, help="Nonlinear measure of fundamental frequency variation. Higher values may indicate Parkinson's disease.")
        
    with col1:
        d2 = st.slider('D2', min_value=0.0, max_value=10.0, step=0.01, value=5.0, help="Correlation dimension. Higher values may indicate Parkinson's disease.")
        
    with col2:
        ppe = st.slider('PPE', min_value=0.0, max_value=1.0, step=0.01, value=0.5, help="Pitch period entropy. Higher values may indicate Parkinson's disease.")
    # Code for Prediction
    parkinsons_diagnosis = ''
    
    # Button for Prediction    
    if st.button("Parkinson's Test Result"):
        if not all([fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]):
            st.warning("Please fill in all the fields.")
        else:
            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq,ddp,shimmer,shimmer_db,apq3,apq5,apq,dda,nhr,hnr,rpde,dfa,spread1,spread2,d2,ppe]])                          
            
            if parkinsons_prediction[0] == 1:
              parkinsons_diagnosis = "The person has Parkinson's disease."
            else:
              parkinsons_diagnosis = "The person does not have Parkinson's disease."
        
    st.success(parkinsons_diagnosis)

# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    # Page title
    st.title('Breast Cancer Prediction using ML')

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mean_radius = st.slider('Mean Radius', min_value=0.0, max_value=40.0, step=0.1, value=15.0, 
                                help="Mean radius of the cancerous cells.")
        
        mean_texture = st.slider('Mean Texture', min_value=0.0, max_value=30.0, step=0.1, value=15.0, 
                                 help="Mean texture of the cancerous cells.")
        
        mean_perimeter = st.slider('Mean Perimeter', min_value=0.0, max_value=250.0, step=1.0, value=100.0, 
                                    help="Mean perimeter of the cancerous cells.")
        
        mean_area = st.slider('Mean Area', min_value=0.0, max_value=3000.0, step=10.0, value=500.0, 
                               help="Mean area of the cancerous cells.")
        
        mean_smoothness = st.slider('Mean Smoothness', min_value=0.0, max_value=0.3, step=0.01, value=0.1, 
                                    help="Mean smoothness of the cancerous cells.")
        
        mean_compactness = st.slider('Mean Compactness', min_value=0.0, max_value=0.3, step=0.01, value=0.1, 
                                      help="Mean compactness of the cancerous cells.")
        
        mean_concavity = st.slider('Mean Concavity', min_value=0.0, max_value=0.3, step=0.01, value=0.1, 
                                    help="Mean concavity of the cancerous cells.")
        
        mean_concave_points = st.slider('Mean Concave Points', min_value=0.0, max_value=0.3, step=0.01, value=0.1, 
                                         help="Mean number of concave portions of the contour for the cancerous cells.")
        
        mean_symmetry = st.slider('Mean Symmetry', min_value=0.0, max_value=1.0, step=0.01, value=0.5, 
                                   help="Mean symmetry of the cancerous cells.")
        
        mean_fractal_dimension = st.slider('Mean Fractal Dimension', min_value=0.0, max_value=0.1, step=0.001, value=0.05, 
                                            help="Mean fractal dimension of the cancerous cells.")
        
    with col2:
        radius_error = st.slider('Radius Error', min_value=0.0, max_value=2.0, step=0.01, value=0.5, 
                                  help="Error in radius measurement for the cancerous cells.")
        
        texture_error = st.slider('Texture Error', min_value=0.0, max_value=5.0, step=0.01, value=1.0, 
                                   help="Error in texture measurement for the cancerous cells.")
        
        perimeter_error = st.slider('Perimeter Error', min_value=0.0, max_value=20.0, step=0.1, value=5.0, 
                                     help="Error in perimeter measurement for the cancerous cells.")
        
        area_error = st.slider('Area Error', min_value=0.0, max_value=200.0, step=1.0, value=50.0, 
                                help="Error in area measurement for the cancerous cells.")
        
        smoothness_error = st.slider('Smoothness Error', min_value=0.0, max_value=0.5, step=0.01, value=0.1, 
                                      help="Error in smoothness measurement for the cancerous cells.")
        
        compactness_error = st.slider('Compactness Error', min_value=0.0, max_value=1.0, step=0.01, value=0.2, 
                                       help="Error in compactness measurement for the cancerous cells.")
        
        concavity_error = st.slider('Concavity Error', min_value=0.0, max_value=1.0, step=0.01, value=0.2, 
                                     help="Error in concavity measurement for the cancerous cells.")
        
        concave_points_error = st.slider('Concave Points Error', min_value=0.0, max_value=1.0, step=0.01, value=0.2, 
                                          help="Error in number of concave portions of the contour measurement for the cancerous cells.")
        
        symmetry_error = st.slider('Symmetry Error', min_value=0.0, max_value=0.5, step=0.01, value=0.1, 
                                    help="Error in symmetry measurement for the cancerous cells.")
        
        fractal_dimension_error = st.slider('Fractal Dimension Error', min_value=0.0, max_value=0.1, step=0.001, value=0.05, 
                                             help="Error in fractal dimension measurement for the cancerous cells.")
    
    with col3:
        worst_radius = st.slider('Worst Radius', min_value=0.0, max_value=40.0, step=0.1, value=15.0, 
                                help="Worst radius of the cancerous cells.")
        
        worst_texture = st.slider('Worst Texture', min_value=0.0, max_value=30.0, step=0.1, value=15.0, 
                                 help="Worst texture of the cancerous cells.")
        
        worst_perimeter = st.slider('Worst Perimeter', min_value=0.0, max_value=250.0, step=1.0, value=100.0, 
                                    help="Worst perimeter of the cancerous cells.")
        
        worst_area = st.slider('Worst Area', min_value=0.0, max_value=3000.0, step=10.0, value=500.0, 
                               help="Worst area of the cancerous cells.")
        
        worst_smoothness = st.slider('Worst Smoothness', min_value=0.0, max_value=0.3, step=0.01, value=0.1, 
                                    help="Worst smoothness of the cancerous cells.")
        
        worst_compactness = st.slider('Worst Compactness', min_value=0.0, max_value=0.3, step=0.01, value=0.1, 
                                      help="Worst compactness of the cancerous cells.")
        
        worst_concavity = st.slider('Worst Concavity', min_value=0.0, max_value=0.3, step=0.01, value=0.1, 
                                    help="Worst concavity of the cancerous cells.")
        
        worst_concave_points = st.slider('Worst Concave Points', min_value=0.0, max_value=0.3, step=0.01, value=0.1, 
                                         help="Worst number of concave portions of the contour for the cancerous cells.")
        
        worst_symmetry = st.slider('Worst Symmetry', min_value=0.0, max_value=1.0, step=0.01, value=0.5, 
                                   help="Worst symmetry of the cancerous cells.")
        
        worst_fractal_dimension = st.slider('Worst Fractal Dimension', min_value=0.0, max_value=0.1, step=0.001, value=0.05, 
                                            help="Worst fractal dimension of the cancerous cells.")
        
    # Code for prediction
    cancer_diagnosis = ''
    
    # Button for prediction
    if st.button('Breast Cancer Test Result'):
        if not all([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness,
                    mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error,
                    texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error,
                    concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture,
                    worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity,
                    worst_concave_points, worst_symmetry, worst_fractal_dimension]):
            st.warning("Please fill in all the fields.")
        else:
            cancer_prediction = breast_model.predict([[mean_radius, mean_texture, mean_perimeter, mean_area,
                                                       mean_smoothness, mean_compactness, mean_concavity,
                                                       mean_concave_points, mean_symmetry, mean_fractal_dimension,
                                                       radius_error, texture_error, perimeter_error, area_error,
                                                       smoothness_error, compactness_error, concavity_error,
                                                       concave_points_error, symmetry_error, fractal_dimension_error,
                                                       worst_radius, worst_texture, worst_perimeter, worst_area,
                                                       worst_smoothness, worst_compactness, worst_concavity,
                                                       worst_concave_points, worst_symmetry, worst_fractal_dimension]])
            
            if cancer_prediction[0] == 1:
                cancer_diagnosis = 'The person is diagnosed with breast cancer.'
            else:
                cancer_diagnosis = 'The person is not diagnosed with breast cancer.'
        
    st.success(cancer_diagnosis)
