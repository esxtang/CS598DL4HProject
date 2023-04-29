### Start tracking time and memory usage
# Import time & memory modules
import time
import tracemalloc
 
# Starting the monitoring
start = time.time()
tracemalloc.start()


### Import libraries
import numpy as np
import pandas as pd
from IPython.display import display


### Path for file access
MIMIC_DATA_PATH = './MimicData/'
OUTPUT_PATH = './PreprocessedData/'


### Get MIMIC-III data
# Read itemid_to_variable_map from MIMIC-Extract 
# https://github.com/MLforHealth/MIMIC_Extract/blob/master/README.md
MIMIC_Extract_itemid_to_variable_map = pd.read_csv(
    MIMIC_DATA_PATH + 'itemid_to_variable_map.csv', 
    index_col = 'ITEMID', 
    dtype = {'ITEMID': int})
# Read LABEVENTS data from MIMIC III
MIMIC_III_lab_events = pd.read_csv(
    MIMIC_DATA_PATH + 'LABEVENTS.csv', 
    index_col = 'SUBJECT_ID', 
    parse_dates = ['CHARTTIME'])
# Read PATIENTS data from MIMIC III
MIMIC_III_patients = pd.read_csv(
    MIMIC_DATA_PATH + 'PATIENTS.csv', 
    usecols = ['SUBJECT_ID', 'GENDER', 'DOB'], 
    index_col = 'SUBJECT_ID', 
    parse_dates = ['DOB'])
# Read ADMISSIONS data from MIMIC III
MIMIC_III_admissions = pd.read_csv(
    MIMIC_DATA_PATH + 'ADMISSIONS.csv',  
    usecols = ['SUBJECT_ID', 'HADM_ID', 'ADMISSION_TYPE', 'HOSPITAL_EXPIRE_FLAG', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'INSURANCE', 'DIAGNOSIS'],
    index_col = 'SUBJECT_ID',
    parse_dates = ['ADMITTIME', 'DISCHTIME', 'DEATHTIME'])


### Clean Patient and Admission data
# Combine PATIENTS and ADMISSIONS data
admissions = MIMIC_III_admissions.merge(MIMIC_III_patients, on = 'SUBJECT_ID')

print(admissions.shape)
admissions.head()

zero_days = pd.to_timedelta('0 days')
one_day = pd.to_timedelta('1 day')

# Add Length of Stay between Admission and Death (unit is fraction of a day)
admissions['LOS_DEATH'] = (admissions['DEATHTIME'] - admissions['ADMITTIME']) / one_day

# Add Length of Stay between Admission and Discharge (unit is days)
admissions['LOS_DISCHARGE'] = admissions['DISCHTIME'] - admissions['ADMITTIME']

# Add Age (length of time between Admission and DOB, unit is years)
admissions['AGE'] = (admissions['ADMITTIME'].dt.year 
                            - admissions['DOB'].dt.year 
                            - ((admissions['ADMITTIME'].dt.month < admissions['DOB'].dt.month)
                            | ((admissions['ADMITTIME'].dt.month == admissions['DOB'].dt.month) 
                            & (admissions['ADMITTIME'].dt.day < admissions['DOB'].dt.day))))

# Code to identify patients who are in the original paper's data set but removed from my data set (1 patient with SUBJECT_ID = 45184)
excluded_patients = admissions.copy()
excluded_patients['AGE_ORIGINAL'] = [date.days for date in excluded_patients['ADMITTIME'].dt.to_pydatetime() - excluded_patients['DOB'].dt.to_pydatetime()]
excluded_patients['AGE_ORIGINAL'] /= 365
excluded_patients = excluded_patients[(excluded_patients['AGE_ORIGINAL'] > 18) & (excluded_patients['AGE'] < 18)]

print('Number of patients excluded:', excluded_patients.shape[0])
display(excluded_patients)
if excluded_patients.index.isin(MIMIC_III_lab_events.index):
  excluded_labs = MIMIC_III_lab_events.loc[excluded_patients.index]
  print('Number of labs excluded:', excluded_labs.shape[0])

# Keep only adult patients
adult_admissions = admissions[admissions['AGE'] >= 18]

# Keep only the last record for each SUBJECT ID
adult_admissions = adult_admissions[~adult_admissions.index.duplicated(keep = 'last')]

# Keep only patients whose length of stay before discharge is >=1 day
LOSgte1_adult_admissions = adult_admissions[adult_admissions['LOS_DISCHARGE'] >= one_day]

print(LOSgte1_adult_admissions.shape)
LOSgte1_adult_admissions.head()


### Clean Item Id to Variable Map
# Remove rows with blank LEVEL2, COUNT <= 0, or STATUS != 'ready'
itemid_to_variable_map = MIMIC_Extract_itemid_to_variable_map.dropna(subset = ['LEVEL2'])
itemid_to_variable_map = itemid_to_variable_map[(itemid_to_variable_map['COUNT'] > 0) & (itemid_to_variable_map['STATUS'] == 'ready')]

print(itemid_to_variable_map.shape)
itemid_to_variable_map.head()


### Clean Lab data
# Select only necessary columns
lab_events = MIMIC_III_lab_events[['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']]

# Select only rows with ITEMIDs in the itemid_to_variable_map & with HADM_IDs in LOSgte1_adult_admissions
subset_lab_events = lab_events[((lab_events['ITEMID'].isin(itemid_to_variable_map.index))
                              & (lab_events['HADM_ID'].isin(LOSgte1_adult_admissions['HADM_ID'])))]

# Add a name of Lab using the itemid_to_variable_map
subset_lab_events['Lab'] = itemid_to_variable_map['LEVEL1'].loc[subset_lab_events['ITEMID']].values

# Add Length of Stay between Admission and Lab Event
subset_lab_events['LOS_LAB_EVENT'] = subset_lab_events['CHARTTIME'] - LOSgte1_adult_admissions['ADMITTIME'].loc[subset_lab_events.index]

# Keep only labs from within the first day & who are in the patient list & that don't have multiple values for the same Patient & LOS_LAB_EVENT timestamp
day1_lab_events = subset_lab_events[((subset_lab_events['LOS_LAB_EVENT'] < one_day) 
                                   & (subset_lab_events['LOS_LAB_EVENT'] > zero_days)
                                   & (subset_lab_events.index.isin(LOSgte1_adult_admissions.index))
                                   & (~subset_lab_events.reset_index().duplicated(subset = ['SUBJECT_ID', 'LOS_LAB_EVENT', 'Lab'], keep = False).values))]

# Change LOS_LAB_EVENT from days into fractions of a day
day1_lab_events['LOS_LAB_EVENT'] = day1_lab_events['LOS_LAB_EVENT'] / one_day

print(day1_lab_events.shape)
day1_lab_events.head()


### Reformat Lab data
# Select certain columns
day1_lab_events = day1_lab_events[['LOS_LAB_EVENT', 'Lab', 'VALUENUM']]

# Pivot labs into columns
day1_lab_events = day1_lab_events.reset_index().pivot(index = ['SUBJECT_ID', 'LOS_LAB_EVENT'], columns = 'Lab')

# Remove if all lab values are NA
day1_lab_events = day1_lab_events.dropna(how = 'all')

print(day1_lab_events.shape)
day1_lab_events.head()


### Filter Patient Admissions set to only include those with Lab records
LOSgte1_adult_admissions = LOSgte1_adult_admissions.loc[day1_lab_events.index.get_level_values(0).unique()]

print(LOSgte1_adult_admissions.shape)
LOSgte1_adult_admissions.head()


### Convert values to binary labels
labeled_LOSgte1_adult_admissions = LOSgte1_adult_admissions.copy(deep = True)

labeled_LOSgte1_adult_admissions['Outcome'] = (labeled_LOSgte1_adult_admissions['LOS_DEATH'] < 8).replace({True: 'Death', False: 'Alive'})
labeled_LOSgte1_adult_admissions['ETHNICITY'] = (labeled_LOSgte1_adult_admissions['ETHNICITY'].str.contains('BLACK')).replace({True: 'Black', False: 'Non Black'})
labeled_LOSgte1_adult_admissions['GENDER'] = (labeled_LOSgte1_adult_admissions['GENDER'] == 'F').replace({True: 'Female', False: 'Male'})
labeled_LOSgte1_adult_admissions['INSURANCE'] = (labeled_LOSgte1_adult_admissions['INSURANCE'] == 'Private').replace({True: 'Private', False: 'Public'})

print(labeled_LOSgte1_adult_admissions.shape)
labeled_LOSgte1_adult_admissions.head()


### Save files
labeled_LOSgte1_adult_admissions.to_csv(OUTPUT_PATH + 'preprocessed_labeled_outcomes.csv')
day1_lab_events.to_csv(OUTPUT_PATH + 'preprocessed_labs.csv')


### Report time and memory usage
# Record end time
end = time.time()

display(pd.DataFrame(
    data = {'Preprocessing': [(end - start)/60, tracemalloc.get_traced_memory()[1]/10**9]},
    index = ['Time (min)', 'Memory Usage (GB)']))
 
# Stopping the memory trace
tracemalloc.stop()
