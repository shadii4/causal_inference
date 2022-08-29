
ins_map = {'Government': 0,
           'Self Pay': 1,
           'Medicare': 2,
           'Private': 3,
           'Medicaid': 4}
status_map = {'': 0,
              'UNKNOWN (DEFAULT)': 0,
              'SINGLE': 1,
              'SEPARATED': 2,
              'DIVORCED': 3,
              'MARRIED': 4,
              'WIDOWED': 5,
              'NaN':0
              }

def get_next_intime(row, stays):
    subj_id = row['SUBJECT_ID']
    outtime = row['OUTTIME']
    later_stays = stays[(stays.SUBJECT_ID == subj_id) & (stays.INTIME > outtime)]
    if len(later_stays) == 0:
        return None
    return later_stays.sort_values(by='INTIME', ascending=True).iloc[0].INTIME
def combine(list_of_sets):
    s = []
    for s1 in list_of_sets:
        s.extend(list(s1))
    return s


def safe_age(row):
    try:
        return row['ADMITTIME'] - row['DOB']
    except:
        return None


def merge_stays_counts(table1, table2):
    return table1.merge(table2, how='inner', left_on=['HADM_ID'], right_on=['HADM_ID'])

def add_inhospital_mortality_to_icustays(stays):
    mortality_all = stays.DOD.notnull() | stays.DEATHTIME.notnull()
    stays['MORTALITY'] = mortality_all.astype(int)

    # in hospital mortality
    mortality = stays.DEATHTIME.notnull() & ((stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME))

    stays['MORTALITY0'] = mortality.astype(int)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY0']
    return stays


def add_inunit_mortality_to_icustays(stays):
    mortality = stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME))

    stays['MORTALITY_INUNIT'] = mortality.astype(int)
    return stays
