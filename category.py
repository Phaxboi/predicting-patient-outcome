import argparse
import os
from unicodedata import category
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

def add_category_capillary(episode, category):
    category['cat_crt'] = [np.nan if pd.isna(ele) else 'normal' if ele <= 2 else 'abnormal' for ele in episode['Capillary refill rate']]
    category['num_crt'] = [0 if ele == 'normal' else 1 if ele == 'abnormal' else np.nan for ele in category['cat_crt']]
    return category

def add_category_diastolic_blood_pressure(episode, category):
    categories_dbp = [
        (pd.isna(episode['Diastolic blood pressure']) == True),
        (episode['Diastolic blood pressure'] < 60),
        (episode['Diastolic blood pressure'] >= 60) & (episode['Diastolic blood pressure'] < 90),
        (episode['Diastolic blood pressure'] >= 90)
        ]
    category_diastolic = [np.nan,'low', 'normal', 'high']
    category['cat_diastolic_blood_pressure'] = np.select(categories_dbp, category_diastolic).astype('str')
    category['cat_diastolic_blood_pressure'] = [np.nan if ele == 'nan' else ele for ele in category['cat_diastolic_blood_pressure']]
    category['num_dbp'] = [0 if ele == 'low' else 1 if ele == 'normal' else 2 if ele == 'high' else np.nan for ele in category['cat_diastolic_blood_pressure']]
    return category

def add_category_fraction_inspired_oxygen(episode, category):
    categories_fio = [
        (pd.isna(episode['Fraction inspired oxygen']) == True),
        (episode['Fraction inspired oxygen'] < 21),
        (episode['Fraction inspired oxygen'] == 21),
        (episode['Fraction inspired oxygen'] > 21)
        ]
    category_fraction_inspired_oxygen = [np.nan, 'low', 'normal', 'high']
    category['cat_fraction_inspired_oxygen'] = np.select(categories_fio, category_fraction_inspired_oxygen)
    category['cat_fraction_inspired_oxygen'] = [np.nan if ele == 'nan' else ele for ele in category['cat_fraction_inspired_oxygen']]
    category['num_fio'] = [0 if ele == 'low' else 1 if ele == 'normal' else 2 if ele == 'high' else np.nan for ele in category['cat_fraction_inspired_oxygen']]
    return category

def add_category_glasgow_coma_scale_eye_opening(episode, category):
    categories_gcs_eye = [
        (episode['Glasgow coma scale eye opening'] == 0),
        (episode['Glasgow coma scale eye opening'] == 1),
        (episode['Glasgow coma scale eye opening'] == 2),
        (episode['Glasgow coma scale eye opening'] == 3),
        (episode['Glasgow coma scale eye opening'] == 4)
        ]
    category_eye = ['Not Testable', 'Does not open eyes', 'Opens eyes in response to pain', 'Opens eyes in response to voice', 'Opens eyes spontaneously']
    category['cat_gcs_eye'] = np.select(categories_gcs_eye, category_eye)
    category['cat_gcs_eye'] = [np.nan if ele == 'nan' else ele for ele in category['cat_gcs_eye']]
    category['num_gcs_eye'] = [0 if ele == 'Not Testable' else 1 if ele == 'Does not open eyes' else 2 if ele == 'Opens eyes in response to pain' else 3 if ele == 'Opens eyes in response to voice' else 4 if ele == 'Opens eyes spontaneously' else np.nan for ele in category['cat_gcs_eye']]
    return category

def add_category_glasgow_coma_scale_motor_response(episode, category):
    categories_gcs_motor = [
        (episode['Glasgow coma scale motor response'] == 0),
        (episode['Glasgow coma scale motor response'] == 1),
        (episode['Glasgow coma scale motor response'] == 2),
        (episode['Glasgow coma scale motor response'] == 3),
        (episode['Glasgow coma scale motor response'] == 4),
        (episode['Glasgow coma scale motor response'] == 5),
        (episode['Glasgow coma scale motor response'] == 6)
        ]
    category_motor = ['Not Testable', 'Makes no movements', 'Extension to painful stimuli', 'Abnormal flexion to painful stimuli', 'Flexion / Withdrawal to painful stimuli', 'Localizes to painful stimuli', 'Obeys commands']
    category['cat_gcs_motor'] = np.select(categories_gcs_motor, category_motor)
    category['cat_gcs_motor'] = [np.nan if ele == 'nan' else ele for ele in category['cat_gcs_motor']]
    category['num_gcs_motor'] = [0 if ele == 'Not Testable' else 1 if ele == 'Makes no movements' else 2 if ele == 'Extension to painful stimuli' else 3 if ele == 'Abnormal flexion to painful stimuli' else 4 if ele == 'Flexion / Withdrawal to painful stimuli' else 5 if ele == 'Localizes to painful stimuli' else 6 if ele == 'Obeys commands' else np.nan for ele in category['cat_gcs_motor']]
    return category

def add_category_glasgow_coma_scale_verbal_response(episode, category):
    categories_gcs_verbal = [
        (episode['Glasgow coma scale verbal response'] == 0),
        (episode['Glasgow coma scale verbal response'] == 1),
        (episode['Glasgow coma scale verbal response'] == 2),
        (episode['Glasgow coma scale verbal response'] == 3),
        (episode['Glasgow coma scale verbal response'] == 4),
        (episode['Glasgow coma scale verbal response'] == 5)
        ]
    category_verbal = ['Not Testable', 'Makes no sounds', 'Makes sounds', 'Words', 'Confused, disoriented', 'Oriented, converses normally']
    category['cat_gcs_verbal'] = np.select(categories_gcs_verbal, category_verbal)
    category['cat_gcs_verbal'] = [np.nan if ele == 'nan' else ele for ele in category['cat_gcs_verbal']]
    category['num_gcs_verbal'] = [0 if ele == 'Not Testable' else 1 if ele == 'Makes no sounds' else 2 if ele == 'Makes sounds' else 3 if ele == 'Words' else 4 if ele == 'Confused, disoriented' else 5 if ele == 'Oriented, converses normally' else np.nan for ele in category['cat_gcs_verbal']]
    return category

def add_category_glucose(episode, category):
    # 1 mmol/L = 18 mg/dL
    categories_glucose = [
        (pd.isna(episode['Glucose']) == True),
        (episode['Glucose'] < (18*4)),
        (episode['Glucose'] >= (18*4)) & (episode['Glucose'] < (18*8.7)),
        (episode['Glucose'] >= (18*8.7)) & (episode['Glucose'] < (18*12.2)),
        (episode['Glucose'] >= (18*12.2))
        ]
    category_glu = [np.nan, 'low', 'normal', 'elevated', 'high']
    category['cat_glucose'] = np.select(categories_glucose, category_glu)
    category['cat_glucose'] = [np.nan if ele == 'nan' else ele for ele in category['cat_glucose']]
    category['num_glucose'] = [0 if ele == 'low' else 1 if ele == 'normal' else 2 if ele == 'elevated' else 3 if ele == 'high' else np.nan for ele in category['cat_glucose']]
    return category

def add_category_heart_rate(episode, category):
    categories_heart_rate = [
        (pd.isna(episode['Heart rate']) == True),
        (episode['Heart rate'] < 50),
        (episode['Heart rate'] >= 50) & (episode['Heart rate'] < 100),
        (episode['Heart rate'] >= 100)
        ]
    category_hr = [np.nan, 'low', 'normal', 'high']
    category['cat_heart_rate'] = np.select(categories_heart_rate, category_hr)
    category['cat_heart_rate'] = [np.nan if ele == 'nan' else ele for ele in category['cat_heart_rate']]
    category['num_heart_rate'] = [0 if ele == 'low' else 1 if ele == 'normal' else 2 if ele == 'high' else np.nan for ele in category['cat_heart_rate']]
    return category

def add_category_height(episode, category):
    # Mean height in Sweden, Women: 166 cm and Men: 180 cm
    categories_height = [
        (pd.isna(episode['Height']) == True),
        (episode['Height'] < 166),
        (episode['Height'] >= 166) & (episode['Height'] < 180),
        (episode['Height'] >= 180)
        ]
    category_height = [np.nan, 'low', 'normal', 'high']
    category['cat_height'] = np.select(categories_height, category_height)
    category['cat_height'] = [np.nan if ele == 'nan' else ele for ele in category['cat_height']]
    category['num_height'] = [0 if ele == 'low' else 1 if ele == 'normal' else 2 if ele == 'high' else np.nan for ele in category['cat_height']]
    return category

def add_category_mean_blood_pressure(episode, category):
    #Min för diastolic and max for systolic
    categories_mbp = [
        (pd.isna(episode['Mean blood pressure']) == True),
        (episode['Mean blood pressure'] < 60),
        (episode['Mean blood pressure'] >= 60) & (episode['Mean blood pressure'] < 140),
        (episode['Mean blood pressure'] >= 140)
        ]
    category_mbp = [np.nan, 'low', 'normal', 'high']
    category['cat_mean_blood_pressure'] = np.select(categories_mbp, category_mbp)
    category['cat_mean_blood_pressure'] = [np.nan if ele == 'nan' else ele for ele in category['cat_mean_blood_pressure']]
    category['num_mean_blood_pressure'] = [0 if ele == 'low' else 1 if ele == 'normal' else 2 if ele == 'high' else np.nan for ele in category['cat_mean_blood_pressure']]
    return category

def add_category_oxygen_saturation(episode, category):
    categories_os = [
        (pd.isna(episode['Oxygen saturation']) == True),
        (episode['Oxygen saturation'] < 94),
        (episode['Oxygen saturation'] >= 94) & (episode['Oxygen saturation'] < 96),
        (episode['Oxygen saturation'] >= 96) & (episode['Oxygen saturation'] < 100)
        ]
    category_os = [np.nan, 'critical', 'low', 'normal']
    category['cat_oxygen_saturation'] = np.select(categories_os, category_os)
    category['cat_oxygen_saturation'] = [np.nan if ele == 'nan' else ele for ele in category['cat_oxygen_saturation']]
    category['num_oxygen_saturation'] = [0 if ele == 'critical' else 1 if ele == 'low' else 2 if ele == 'normal' else np.nan for ele in category['cat_oxygen_saturation']]
    return category

def add_category_respiratory_rate(episode, category):
    #the number of breaths a person takes per minute.
    categories_rr = [
        (pd.isna(episode['Respiratory Rate']) == True),
        (episode['Respiratory Rate'] < 12),
        (episode['Respiratory Rate'] >= 12) & (episode['Respiratory Rate'] <= 16),
        (episode['Respiratory Rate'] > 16)
        ]
    category_respiratory_rate = [np.nan, 'low', 'normal', 'high']
    category['cat_respiratory_rate'] = np.select(categories_rr, category_respiratory_rate)
    category['cat_respiratory_rate'] = [np.nan if ele == 'nan' else ele for ele in category['cat_respiratory_rate']]
    category['num_respiratory_rate'] = [0 if ele == 'low' else 1 if ele == 'normal' else 2 if ele == 'high' else np.nan for ele in category['cat_respiratory_rate']]
    return category

def add_systolic_blood_pressure(episode, category):
    categories_sbp = [
        (pd.isna(episode['Systolic blood pressure']) == True),
        (episode['Systolic blood pressure'] < 90),
        (episode['Systolic blood pressure'] >= 90) & (episode['Systolic blood pressure'] < 140),
        (episode['Systolic blood pressure'] >= 140)
        ]
    category_systolic = [np.nan, 'low', 'normal', 'high']
    category['cat_systolic_blood_pressure'] = np.select(categories_sbp, category_systolic)
    category['cat_systolic_blood_pressure'] = [np.nan if ele == 'nan' else ele for ele in category['cat_systolic_blood_pressure']]
    category['num_systolic_blood_pressure'] = [0 if ele == 'low' else 1 if ele == 'normal' else 2 if ele == 'high' else np.nan for ele in category['cat_systolic_blood_pressure']]
    return category

def add_category_temperature(episode, category):
    categories_temp = [
        (pd.isna(episode['Temperature']) == True),
        (episode['Temperature'] < 36.5),
        (episode['Temperature'] >= 36.5) & (episode['Temperature'] <= 37.2),
        (episode['Temperature'] > 37.2)
        ]
    category_temperature = [np.nan, 'low', 'normal', 'high']
    category['cat_temperature'] = np.select(categories_temp, category_temperature)
    category['cat_temperature'] = [np.nan if ele == 'nan' else ele for ele in category['cat_temperature']]
    category['num_temperature'] = [0 if ele == 'low' else 1 if ele == 'normal' else 2 if ele == 'high' else np.nan for ele in category['cat_temperature']]
    return category

def add_category_weight(episode, category):
    height = episode['Height'].unique()
    height = round(sorted(height)[-1]*0.01,2)

    bmi = episode['Weight'] / height ** 2
    categories_bmi = [
        (pd.isna(bmi) == True),
        (bmi < 18.5),
        (bmi >= 18.5) & (bmi < 25),
        (bmi >= 25) & (bmi < 30),
        (bmi >= 30)
        ]
    category_bmi = [np.nan, 'underweight', 'normal', 'overweight', 'obesity']
    category['cat_bmi'] = np.select(categories_bmi, category_bmi)
    category['cat_bmi'] = [np.nan if ele == 'nan' else ele for ele in category['cat_bmi']]
    category['num_bmi'] = [0 if ele == 'underweight' else 1 if ele == 'normal' else 2 if ele == 'overweight' else 3 if ele == 'obesity' else np.nan for ele in category['cat_bmi']]
    return category

def add_category_ph(episode, category):
    categories_ph = [
        (pd.isna(episode['pH']) == True),
        (episode['pH'] < 7.35),
        (episode['pH'] >= 7.35) & (episode['pH'] <= 7.45),
        (episode['pH'] > 7.45)
        ]
    category_ph = [np.nan, 'acidic', 'normal', 'alkaline']
    category['cat_ph'] = np.select(categories_ph, category_ph)
    category['cat_ph'] = [np.nan if ele == 'nan' else ele for ele in category['cat_ph']]
    category['num_ph'] = [0 if ele == 'acidic' else 1 if ele == 'normal' else 2 if ele == 'alkaline' else np.nan for ele in category['cat_ph']]
    return category

def remove_cat_values(category):
    category = category.drop(columns=['cat_crt', 'cat_diastolic_blood_pressure', 'cat_fraction_inspired_oxygen', 'cat_gcs_eye', 'cat_gcs_motor', 'cat_gcs_verbal', 'cat_glucose', 'cat_heart_rate', 'cat_height', 'cat_mean_blood_pressure', 'cat_oxygen_saturation', 'cat_respiratory_rate', 'cat_systolic_blood_pressure', 'cat_temperature', 'cat_bmi', 'cat_ph'])
    return category

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
    args = parser.parse_args()
    subjects_root_path = args.subjects_root_path

    for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='Category'):
        category_counter = 0
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))
                category = pd.DataFrame()
                category['hours'] = episode['hours']
                category = add_category_capillary(episode, category)
                category = add_category_diastolic_blood_pressure(episode, category)
                category = add_category_fraction_inspired_oxygen(episode, category)
                category = add_category_glasgow_coma_scale_eye_opening(episode, category)
                category = add_category_glasgow_coma_scale_motor_response(episode, category)
                category = add_category_glasgow_coma_scale_verbal_response(episode, category)
                category = add_category_glucose(episode, category)
                category = add_category_heart_rate(episode, category)
                category = add_category_height(episode, category)
                category = add_category_mean_blood_pressure(episode, category)
                category = add_category_oxygen_saturation(episode, category)
                category = add_category_respiratory_rate(episode, category)
                category = add_systolic_blood_pressure(episode, category)
                category = add_category_temperature(episode, category)
                category = add_category_weight(episode, category)
                category = add_category_ph(episode, category)

                subj_id = re.search('.*_(\d*)_.*', file_name).group(1)
                file_name = 'category' + str(category_counter) + '_' + str(subj_id) + '.csv'
                category.to_csv(os.path.join(root, file_name), index=False)

                category = remove_cat_values(category)
                file_name = 'num_category' + str(category_counter) + '_' + str(subj_id) + '.csv'
                category.to_csv(os.path.join(root, file_name), index=False)

if __name__ == '__main__':
    main()