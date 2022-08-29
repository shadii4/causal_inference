import pandas as pd
import os
import numpy as np
from mimic_prepocessing import *
from transformers import AutoTokenizer, AutoModel
import yaml
from box import Box
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def prepare_features(args):
    ## Transformer Model
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    ### Diagnosis Table
    diagnoses = pd.read_csv(os.path.join(args.MIMIC_PATH, 'DIAGNOSES_ICD.csv'), index_col=0)
    diagnoses.columns = [c.upper() for c in diagnoses.columns]
    diagnoses_grpdBy = diagnoses.sort_values(by=['HADM_ID', 'SEQ_NUM']).groupby(by='HADM_ID')['ICD9_CODE'].apply(
        list).reset_index()
    ###-------GroupBy Admission ID------###

    ### Admissions Table
    admis = pd.read_csv(os.path.join(args.MIMIC_PATH, 'ADMISSIONS.csv'), index_col=0)
    admis.columns = [c.upper() for c in admis.columns]
    admis.DISCHTIME = pd.to_datetime(admis.DISCHTIME)
    admis.ADMITTIME = pd.to_datetime(admis.ADMITTIME)
    diag_adm_fields = ['HADM_ID', 'SUBJECT_ID', 'ETHNICITY', 'LOS', 'ADMITTIME', 'DISCHTIME',
                       'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION',
                       # 'MARITAL_STATUS',
                       'DIAGNOSIS']
    # could take LOS from ICUSTAYS.csv
    admis['LOS'] = (admis.DISCHTIME - admis.ADMITTIME).apply(lambda s: s / np.timedelta64(1, 'D'))# / 60. / 60 / 24
    admission_diags = diagnoses_grpdBy.merge(admis[diag_adm_fields], on='HADM_ID')
    ###ICUSTAYS TABLE
    stays = pd.read_csv(os.path.join(args.MIMIC_PATH, 'ICUSTAYS.csv'), index_col=0)
    stays.columns = [c.upper() for c in stays.columns]
    stays_adm_fields = ['HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ADMISSION_TYPE', 'ADMISSION_LOCATION',
                        'DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION', 'ETHNICITY']
    stays = stays.merge(admis[stays_adm_fields], on='HADM_ID', how='inner')
    ### Procedures Table
    procs = pd.read_csv(os.path.join(args.MIMIC_PATH, 'PROCEDURES_ICD.csv'), index_col=0)
    procs.columns = [c.upper() for c in procs.columns]
    admission_procs = procs.sort_values(by=['HADM_ID', 'SEQ_NUM']).groupby(by='HADM_ID')['ICD9_CODE'].apply(
        list).reset_index()
    admission_procs = admission_procs.merge(admis[['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME']], on='HADM_ID')
    ### D_Diagnosis Table
    d_diagnoses = pd.read_csv(os.path.join(args.MIMIC_PATH, 'D_ICD_DIAGNOSES.csv'), index_col=0)
    d_diagnoses.columns = [c.upper() for c in d_diagnoses.columns]
    import re
    def preprocess1(x):
        y = re.sub('\\[(.*?)\\]', '', x)  # remove de-identified brackets
        y = re.sub('[0-9]+\.', '', y)  # remove 1.2. since the segmenter segments based on this
        y = re.sub('dr\.', 'doctor', y)
        y = re.sub('m\.d\.', 'md', y)
        y = re.sub('admission date:', '', y)
        y = re.sub('discharge date:', '', y)
        y = re.sub('--|__|==', '', y)
        return y

    def preprocessing(df_less_n):
        df_less_n['TEXT'] = df_less_n['TEXT'].fillna(' ')
        df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\n', ' ')
        df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\r', ' ')
        df_less_n['TEXT'] = df_less_n['TEXT'].apply(str.strip)
        df_less_n['TEXT'] = df_less_n['TEXT'].str.lower()
        df_less_n['TEXT'] = df_less_n['TEXT'].apply(lambda x: preprocess1(x))

        # to get 318 words chunks for readmission tasks
        if False:
            from tqdm import tqdm
            df_len = len(df_less_n)
            want = pd.DataFrame({'HADM_ID': [], 'TEXT': [], 'Label': []})
            for i in tqdm(range(df_len)):
                x = df_less_n.TEXT.iloc[i].split()
                n = int(len(x) / 318)
                for j in range(n):
                    want = want.append(
                        {'TEXT': ' '.join(x[j * 318:(j + 1) * 318]), 'Label': df_less_n.OUTPUT_LABEL.iloc[i],
                         'HADM_ID': df_less_n.HADM_ID.iloc[i]}, ignore_index=True)
                if len(x) % 318 > 10:
                    want = want.append({'TEXT': ' '.join(x[-(len(x) % 318):]), 'Label': df_less_n.OUTPUT_LABEL.iloc[i],
                                        'HADM_ID': df_less_n.HADM_ID.iloc[i]}, ignore_index=True)
            return want
        return df_less_n

    df_notes = pd.read_csv(os.path.join(args.MIMIC_PATH, 'NOTEEVENTS.csv'), index_col=0)
    df_notes.columns = [c.upper() for c in df_notes.columns]
    ### Take only Discharge Summary
    df_notes = df_notes[df_notes['CATEGORY'] == 'Discharge summary']
    df_notes = df_notes.sort_values(by=['HADM_ID', 'SUBJECT_ID', 'CHARTDATE'])
    df_adm_notes = pd.merge(admis[['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'ADMISSION_TYPE', 'DEATHTIME']],
                            df_notes[['HADM_ID', 'SUBJECT_ID', 'CHARTDATE', 'TEXT', 'CATEGORY']],
                            on=['SUBJECT_ID', 'HADM_ID'],
                            how='left')

    df_adm_notes.ADMITTIME_C = df_adm_notes.ADMITTIME.apply(lambda x: str(x).split(' ')[0])
    df_adm_notes['ADMITTIME_C'] = pd.to_datetime(df_adm_notes.ADMITTIME_C, format='%Y-%m-%d', errors='coerce')
    df_adm_notes['CHARTDATE'] = pd.to_datetime(df_adm_notes.CHARTDATE)  # , format = '%Y-%m-%d', errors = 'coerce')
    df_adm_notes = (df_adm_notes.sort_values(by='CHARTDATE').groupby(['SUBJECT_ID', 'HADM_ID']).nth(0)).reset_index()
    df_adm_notes['NOTE_PERIOD'] = (df_adm_notes['CHARTDATE'] - df_adm_notes['ADMITTIME']) / np.timedelta64(1, 'D')
    n = 1
    df_less_n = df_adm_notes#[df_adm_notes['NOTE_PERIOD'] < n]
    df_less_n = df_less_n[df_less_n['TEXT'].notnull()]
    print(f"number of admissions before filtering null notes days={len(df_adm_notes)}, after= {len(df_less_n)}")

    df_discharge = preprocessing(df_less_n)
    admission_diags = admission_diags.merge(df_discharge[['HADM_ID', 'TEXT']], how='inner',
                                            on="HADM_ID")  # --///%%%&&&--inner not outer

    # Patients Table - get demographics from ADMISSIONS and gender from PATIENTS
    patients = pd.read_csv(os.path.join(args.MIMIC_PATH, 'PATIENTS.csv'), index_col=0).reset_index()
    patients.columns = [c.upper() for c in patients.columns]
    patients.DOB = pd.to_datetime(patients.DOB)
    stays = stays.merge(patients[['SUBJECT_ID', 'DOB', 'DOD', 'GENDER']], on='SUBJECT_ID')
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.ADMITTIME = pd.to_datetime(stays.ADMITTIME)
    stays.DISCHTIME = pd.to_datetime(stays.DISCHTIME)
    stays.DOD = pd.to_datetime(stays.DOD)
    stays.DEATHTIME = pd.to_datetime(stays.DEATHTIME)
    stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)
    stays = add_inhospital_mortality_to_icustays(stays)
    stays = add_inunit_mortality_to_icustays(stays)
    # how many icu stays per hospital admission
    counts = stays.groupby(['HADM_ID']).size().reset_index(name='COUNTS')
    stays = merge_stays_counts(stays, counts)
    # binary column: is this stay the last one in the hospital admission?
    max_outtime = stays.groupby(['HADM_ID'])['OUTTIME'].transform(max) == stays['OUTTIME']
    stays['MAX_OUTTIME'] = max_outtime.astype(int)

    # was the patient transferred back to the icu, during this admission, after this stay?
    transferback = (stays.COUNTS > 1) & (stays.MAX_OUTTIME == 0)
    stays['TRANSFERBACK'] = transferback.astype(int)

    # Did the patient die in the hospital but out of the icu?
    dieinward = (stays.MORTALITY == 1) & (stays.MORTALITY_INHOSPITAL == 1) & (stays.MORTALITY_INUNIT == 0)
    stays['DIEINWARD'] = dieinward.astype(int)


    stays['NEXT_INTIME'] = stays.apply(lambda row: get_next_intime(row, stays), axis=1)
    stays['DIFF'] = stays['NEXT_INTIME'] - stays['OUTTIME']

    less_than_30days = stays.DIFF.notnull() & (stays.DIFF < '30 days 00:00:00')
    # less_than_30days = stays.DIFF.notnull() & (stays.DIFF < 30)
    stays['LESS_THAN_30DAYS'] = less_than_30days.astype(int)

    # did the patient die after being discharged? (from the hospital, not from the ICU)
    # stays['DISCHARGE_DIE'] = (stays.DOD - stays.DISCHTIME).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24
    stays['DISCHARGE_DIE'] = stays.DOD - stays.DISCHTIME
    stays['DIE_LESS_THAN_30DAYS'] = (stays.MORTALITY == 1) & (stays.MORTALITY_INHOSPITAL == 0) & (
            stays.MORTALITY_INUNIT == 0) & (stays.DISCHARGE_DIE < '30 days 00:00:00')
    stays['DIE_LESS_THAN_30DAYS'] = stays['DIE_LESS_THAN_30DAYS'].astype(int)

    # final label calculation

    stays['READMISSION'] = ((stays.TRANSFERBACK == 1) | (stays.DIEINWARD == 1) | (stays.LESS_THAN_30DAYS == 1) | (
            stays.DIE_LESS_THAN_30DAYS == 1)).astype(int)
    # Add previous diagnoses embeddings + number of previous procedurs to each admission
    samples = []
    X_diag = np.zeros((len(admission_diags), 768))
    X_notes = np.zeros((len(admission_diags), 768))
    prev_diag_emb = np.zeros((len(admission_diags), 768))
    admission_diags['DIAGNOSIS'] = admission_diags['DIAGNOSIS'].astype(str)

    c = 0
    for i, r in admission_diags.iterrows():
        c+=1
        if c%1000==0:
            print(c)
        subj_id = r['SUBJECT_ID']
        in_time = r['ADMITTIME']
        prime_diag = r['DIAGNOSIS'].replace('S/P', '').replace('/', ';').strip().split(';')[0]
        prev_stays = admission_diags[(admission_diags.SUBJECT_ID == subj_id) & (admission_diags.DISCHTIME < in_time)]
        diag_input = tokenizer(prime_diag, return_tensors="pt")
        X_diag[i] = model(**diag_input)['last_hidden_state'][0, 0, :].detach().numpy()
        note_input = tokenizer(r['TEXT'], return_tensors="pt",truncation=True,max_length=500)
        X_notes[i] = model(**note_input)['last_hidden_state'][0, 0, :].detach().numpy()
        #icd_codes = admission_diags.loc[(admission_diags.SUBJECT_ID == subj_id), 'ICD9_CODE'].values[0]
        #prime_diag_code = icd_codes[0]
        #prime_diag_title = d_diagnoses.loc[d_diagnoses.ICD9_CODE == prime_diag_code, 'LONG_TITLE'].values[0]
        new_row = r
        #new_row['DIAGNOSIS_TITLE'] = prime_diag_title
        new_row['NUM_PREV_ADMIS'] = len(prev_stays)
        if len(prev_stays) == 0:
            new_row['PREV_DIAGS'] = {}
            new_row['DAYS_SINCE_LAST_ADMIS'] = 0
        #else:
            #for icd_code in prev_stays['ICD9_CODE'].values[0]:
                #code_title = d_diagnoses.loc[d_diagnoses.ICD9_CODE == icd_code, 'LONG_TITLE'].values
                # if len(code_title) != 0 :
                # input = tokenizer(code_title[0], return_tensors="pt")
                # prev_diag_emb[i] += model(**input)['last_hidden_state'][0, 0, :].detach().numpy()
            #new_row['PREV_DIAGS'] = set(combine(prev_stays['ICD9_CODE'].values))
        else:
            last_discharge = prev_stays['DISCHTIME'].max()
            new_row['DAYS_SINCE_LAST_ADMIS'] = (in_time - last_discharge) / np.timedelta64(1, 's') / 60. / 60 / 24
        prev_procs = admission_procs[(admission_procs.SUBJECT_ID == subj_id) & (admission_procs.DISCHTIME < in_time)]
        new_row['NUM_PREV_PROCS'] = len(set(combine(prev_procs.ICD9_CODE.values)))
        samples.append(new_row)
    print('hi')
    samples = pd.DataFrame.from_records(samples)
    print('bye')
    # get demographics from ADMISSIONS and gender+birth date from PATIENTS
    # convert to numerical features
    samples = samples.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB']], on='SUBJECT_ID')
    samples['AGE'] = samples.apply(safe_age, axis=1).apply(lambda s: s / np.timedelta64(1, 's')) / 60. / 60 / 24 / 365
    samples.loc[samples.AGE.isna(), 'AGE'] = 90
    samples['AGE'] = samples['AGE'].astype(int)
    samples['LOS'] = samples['LOS'].astype(int)
    samples['GENDER'] = samples['GENDER'].map({'M': -1, 'F': 1})
    # samples['INSURANCE'] = samples['INSURANCE'].map(ins_map)
    # samples['MARITAL_STATUS'] = samples['MARITAL_STATUS'].map(status_map)
    # samples.loc[samples.MARITAL_STATUS.isna(), 'MARITAL_STATUS'] = 0
    samples['DAYS_SINCE_LAST_ADMIS'] = samples['DAYS_SINCE_LAST_ADMIS'].astype(int)
    # samples['ETHNICITY'] = samples.ETHNICITY.str.find("WHITE") == -1  # WHITE = 0 , OTHER = 1
    samples['RACE'] = samples['ETHNICITY']  # .astype('category').cat.codes  # its written : BLACK/AFRICAN AMERICAN
    #  HISPANIC/LATINO - PUERTO RICAN
    samples['READMISSION'] = stays['READMISSION']
    samples['MORTALITY_IN_HOSPITAL'] = stays['MORTALITY_INHOSPITAL']
    features = samples[
        ['HADM_ID', 'AGE', 'GENDER', 'NUM_PREV_ADMIS', 'DAYS_SINCE_LAST_ADMIS', 'NUM_PREV_PROCS', 'INSURANCE',
         'RACE','LOS', 'READMISSION','MORTALITY_IN_HOSPITAL']]
    #features = pd.get_dummies(features, columns=['INSURANCE'])
    print(len(features),len(X_diag))
    features.to_csv(os.path.join(args.FEATURES_PATH,r'disc_features.csv'))
    pd.DataFrame(X_diag).to_csv(os.path.join(args.FEATURES_PATH,r'diagnoses_embeddings.csv'))
    pd.DataFrame(X_notes).to_csv(os.path.join(args.FEATURES_PATH,r'notes_embeddings.csv'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running Experiment')
    parser.add_argument('--config', default='config.yaml', type=str,
                            help='Path to YAML config file. Defualt: config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        args = Box(yaml.load(f, Loader=yaml.FullLoader))

    prepare_features(args)

