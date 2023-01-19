import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import difflib

model = joblib.load('model.pkl')
# Load the dataset
dataset = pd.read_csv('Training.csv')
doc_dataset = pd.read_csv('doctors_dataset.csv')

# Extract the features and labels
X = dataset.iloc[:, :132].values
y = dataset.iloc[:, -1].values


# Dimensionality Reduction for removing redundancies
dimensionality_reduction = dataset.groupby(dataset['prognosis']).max()
#print(dimensionality_reduction)


# Encode the categorical labels
labelencoder = LabelEncoder()

y = labelencoder.fit_transform(y)
cols = dataset.columns
cols = cols[:-1]


for i, col in enumerate(cols):
    print(f'{i+1}. {col}')


symptoms_str = input("Enter the symptoms separated by commas: ")

symptoms_list = symptoms_str.split(',')

symptoms_list2 = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']



similar_symptoms = []
for i in symptoms_list:
    matches = difflib.get_close_matches(i,cols,n=1,cutoff=0.8)
    if matches:
        similar_symptoms.append(matches[0])
    else:
        similar_symptoms.append(i)


print("Your symptoms: ",similar_symptoms)
# Create a new list of binary values corresponding to the presence or absence of each symptom
symptoms_binary = []
for col in cols:
    if col in similar_symptoms:
        symptoms_binary.append(1)
    else:
        symptoms_binary.append(0)

#print(symptoms_binary)
# Define a function to diagnose a disease based on symptoms
def diagnose(symptoms_binary):
    symptoms_binary = np.array(symptoms_binary).reshape(1, -1)
    prediction = model.predict(symptoms_binary)
    disease = labelencoder.inverse_transform(prediction)[0]
    return disease

def get_disease_info(disease):
    des = doc_dataset[doc_dataset['Prognosis'] == disease]['Description'].values[0]
    doc = doc_dataset[doc_dataset['Prognosis'] == disease]['Name'].values[0]
    med = doc_dataset[doc_dataset['Prognosis'] == disease]['Medicine'].values[0]
    return des,doc,med
def run_chatbot():
    disease = diagnose(symptoms_binary)
    print()
    print(f'You may have {disease}.')
    print()
    info=get_disease_info(disease)
    print("Medicine: ", info[2])
    print("Doctor: ",info[1])
    print("Link: ", info[0])




run_chatbot()
