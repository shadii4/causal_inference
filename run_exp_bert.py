import pandas as pd
import os
import numpy as np
from mimic_prepocessing import *
from transformers import AutoTokenizer, AutoModel
import yaml
from box import Box
import torch
import torch.nn as nn
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from modeling import ClinicalCausalBert
from trainer import ModelTrainer
from torch.utils.data import SequentialSampler, DataLoader, Dataset, TensorDataset
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import seaborn as sns


def prepare_features(args):

    ### Diagnosis Table
    parser = argparse.ArgumentParser(description='Running Experiment')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to YAML config file. Defualt: config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        args = Box(yaml.load(f, Loader=yaml.FullLoader))

    disc_features = pd.read_csv(os.path.join(args.FEATURES_PATH, 'disc_features.csv'), index_col=0)
    if args.treatment_label == 'RACE':
        disc_features = pd.get_dummies(disc_features, columns=['INSURANCE'])
        if args.treatment_division == 'white_vs_black':
            # filt = contains WHITE or BLACK
            filt = (disc_features['RACE'].str.find("WHITE") != -1) | (disc_features['RACE'].str.find("BLACK") != -1)
            disc_features = disc_features[filt]
            disc_features['RACE'] = disc_features['RACE'].str.find("WHITE") == -1  # WHITE = 0 vs BLACK = 1
            disc_features['RACE'] = disc_features['RACE'].astype('category').cat.codes
        if args.treatment_division == 'white_vs_other':
            disc_features['RACE'] = disc_features['RACE'].str.find("WHITE") == -1  # WHITE = 0 vs OTHER = 1
            disc_features['RACE'] = disc_features['RACE'].astype('category').cat.codes
            print(f" 0:WHITE , 1:Other {disc_features['RACE'].value_counts()}")
        if args.treatment_division == 'white_vs_spanish':
            # filt = contains WHITE or BLACK
            filt = (disc_features['RACE'].str.find("WHITE") != -1) | (disc_features['RACE'].str.find("LATINO") != -1)
            disc_features = disc_features[filt]
            disc_features['RACE'] = disc_features['RACE'].str.find("WHITE") == -1  # WHITE = 0 vs BLACK = 1
            disc_features['RACE'] = disc_features['RACE'].astype('category').cat.codes
            print(f" 0:WHITE , 1:Spanish {disc_features['RACE'].value_counts()}")
        if args.treatment_division == 'white_vs_asian':
            # filt = contains WHITE or BLACK
            filt = (disc_features['RACE'].str.find("WHITE") != -1) | (disc_features['RACE'].str.find("ASIAN") != -1)
            disc_features = disc_features[filt]
            disc_features['RACE'] = disc_features['RACE'].str.find("WHITE") == -1  # WHITE = 0 vs BLACK = 1
            disc_features['RACE'] = disc_features['RACE'].astype('category').cat.codes
            print(f" 0:WHITE , 1:Asian {disc_features['RACE'].value_counts()}")
    y_propensity = disc_features['RACE']
    y_outcome = disc_features[args.outcome_label]

    ### Admissions Table
    admis = pd.read_csv(os.path.join(args.MIMIC_PATH, 'ADMISSIONS.csv'), index_col=0)
    admis.columns = [c.upper() for c in admis.columns]
    admis.DISCHTIME = pd.to_datetime(admis.DISCHTIME)
    admis.ADMITTIME = pd.to_datetime(admis.ADMITTIME)
    diag_adm_fields = ['HADM_ID', 'SUBJECT_ID', 'ETHNICITY', 'LOS', 'ADMITTIME', 'DISCHTIME',
                       'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION',
                       # 'MARITAL_STATUS',
                       'DIAGNOSIS']
    admis = admis.merge(disc_features[['HADM_ID']], how='inner',
                                            on="HADM_ID")  # --///%%%&&&--inner not outer)
    # could take LOS from ICUSTAYS.csv
    #admis['LOS'] = (admis.DISCHTIME - admis.ADMITTIME).apply(lambda s: s / np.timedelta64(1, 'D'))# / 60. / 60 / 24
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

    y_propensity.to_csv(os.path.join(args.FEATURES_PATH, 'race_labels_'+args.treatment_division+'.csv'))
    y_outcome.to_csv(os.path.join(args.FEATURES_PATH, 'outcome_labels_'+args.treatment_division+'_'+args.outcome_label+'.csv'))
    X = df_discharge['TEXT']
    X.to_csv(os.path.join(args.FEATURES_PATH, 'text_notes_'+args.treatment_division+'_'+args.outcome_label+'.csv'))

    """checkpoint or new model"""

    model = ClinicalCausalBert()
    # model = torch.load(os.path.join(args.FEATURES_PATH, 'clinical-bert-two-layers'))

    y_propensity, y_outcome = pd.read_csv(os.path.join(args.FEATURES_PATH, 'race_labels_'+args.treatment_division+'.csv')).loc[:,'RACE'].to_numpy(),pd.read_csv(os.path.join(args.FEATURES_PATH, 'outcome_labels_'+args.treatment_division+'_'+args.outcome_label+'.csv')).loc[:,args.outcome_label].to_numpy()
    y_propensity, y_outcome = y_propensity, y_outcome
    indices = np.where(y_propensity == 0)[0][:34000] #subsampling white patients
    indices_to_keep = np.array([i for i in range(len(y_propensity)) if i not in indices])
    X = pd.read_csv(os.path.join(args.FEATURES_PATH, 'text_notes.csv')).loc[indices_to_keep, 'TEXT'].astype(str)

    y_propensity = np.delete(y_propensity,indices)
    y_outcome = np.delete(y_outcome,indices)
    #print(len(X), len(y_propensity))
    print(f" 0:WHITE {(y_propensity==0).sum()}, 1:Other {(y_propensity==1).sum()}")


    

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.outcome_type == 'discrete':
        loss_fn = nn.CrossEntropyLoss()
        y_propensity, y_outcome = torch.LongTensor(y_propensity), torch.LongTensor(y_outcome)
    else:
        loss_fn = nn.MSELoss()
        y_propensity, y_outcome = torch.LongTensor(y_propensity), torch.FloatTensor(y_outcome)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs = args.num_epochs
    trainer = ModelTrainer(model,optimizer,loss_fn,device,n_epochs,X,y_propensity, y_outcome,args.FEATURES_PATH)
    trainer.train()
    t1_num = (y_propensity == 1).sum()
    ATE, propensity, outcomesT0, outcomesT1 = trainer.estimate(t1_num)
    #propensity, outcomesT0, outcomesT1 = propensity.detach().numpy(), outcomesT0.detach().numpy(), outcomesT1.detach().numpy()
    pd.DataFrame(propensity).to_csv(os.path.join(args.FEATURES_PATH, r'propensity_estimates_'+args.treatment_division+'_'+args.outcome_label+'.csv'))
    pd.DataFrame(outcomesT0).to_csv(os.path.join(args.FEATURES_PATH, r'outcome_estimatesT0_'+args.treatment_division+'_'+args.outcome_label+'.csv'))
    pd.DataFrame(outcomesT1).to_csv(os.path.join(args.FEATURES_PATH, r'outcome_estimatesT1_'+args.treatment_division+'_'+args.outcome_label+'.csv'))

    experiment_name =  args.treatment_division + '_' + args.outcome_label
    result_path = os.path.join(args.result_folder, experiment_name)

    isExist = os.path.exists(result_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(result_path)
        print("The new exp folder is created!")

    fig, ax = plt.subplots()

    # evaluation of regression model
    #print(f"LR roc auc score = {roc_auc_score(y_propensity.numpy, propensity)}")
    # --print(f"LR roc auc score = {roc_auc_score(labels_binary, DNNpropensity)}")
    # the histogram of the data
    plt.hist(propensity, 50, density=True, facecolor='g', alpha=0.75)
    plt.title('Histogram of Propensity')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    my_file = 'propensityHist.png'
    plt.savefig(os.path.join(result_path, my_file), format='png')
    #plt.show()
    plt.close()

    fig, ax = plt.subplots()
    plt.title('Histogram of Propensity')
    plt.xlim(0, 1)
    plt.grid(True)
    df = pd.DataFrame()
    df['prop'] = propensity
    df['y'] = y_propensity.numpy()
    df.groupby('y')['prop'].plot(kind='hist',sharex=True,range=(0,1),bins=20,alpha=0.7)
    my_file = 'propensityHistGroupBy.png'
    plt.legend()
    plt.savefig(os.path.join(result_path, my_file), format='png')
    plt.close()

    # check calibration:
    fig, ax = plt.subplots()
    prob_true, prob_pred = calibration_curve(y_propensity.numpy(), df['prop'].to_numpy(), n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1)
    # --plt.plot(DNNy, DNNx, marker='o', linewidth=1, label='DNN')
    # reference line, legends, and axis labels
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle('Calibration plot')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability in each bin')
    plt.xlim(0, 1)
    plt.legend()
    my_file = 'calibration.png'

    plt.savefig(os.path.join(result_path, my_file), format='png')
    #plt.show()
    plt.close()
    fig, ax = plt.subplots()
    print('ATE = ', ATE.item())
    plt.text(0.28, 0.1,
             'ATE ='+ str(round(ATE.item(),4)),
             style='italic',
             fontsize=20,
             color="red")
    my_file = 'ATE.png'
    plt.savefig(os.path.join(result_path, my_file), format='png')
    plt.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running Experiment')
    parser.add_argument('--config', default='config.yaml', type=str,
                            help='Path to YAML config file. Defualt: config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        args = Box(yaml.load(f, Loader=yaml.FullLoader))

    prepare_features(args)

