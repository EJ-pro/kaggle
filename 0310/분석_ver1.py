import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def reset_seeds(seed=42):
    import random
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

ori_train = pd.read_csv("/kaggle/input/competitions/skn-27-ml/train.csv")
ori_te = pd.read_csv("/kaggle/input/competitions/skn-27-ml/test.csv")
ori_train.columns = ori_train.columns.str.lower()
ori_te.columns = ori_te.columns.str.lower()

train, test = train_test_split(ori_train, test_size=0.2, random_state=42, stratify=ori_train['survived'])
y_tr = train['survived']
y_te = test['survived']
train = train.drop('survived', axis=1)
test = test.drop('survived', axis=1)

all_tickets = pd.concat([train['ticket'], test['ticket'], ori_te['ticket']])
ticket_counts = all_tickets.value_counts()

fare_map = train.groupby('pclass')['fare'].median()
temp_title = train['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
age_map = train.assign(title=temp_title).groupby(['title', 'pclass'])['age'].median()
global_age_median = train['age'].median()

def process_features(df):
    df = df.copy()
    df['fare'] = df.apply(lambda x: fare_map[x['pclass']] if pd.isna(x['fare']) else x['fare'], axis=1)
    
    temp_title_df = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    def get_age(row, title):
        if pd.isna(row['age']):
            try: return age_map[title, row['pclass']]
            except: return global_age_median
        return row['age']
    df['age'] = [get_age(row, t) for (_, row), t in zip(df.iterrows(), temp_title_df)]
    
    df['gender_code'] = df['gender'].map({"male": 0, "female": 1})
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    
    df['title'] = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3} 
    df['title_code'] = df['title'].map(title_map).fillna(4)
    
    df['pclass_sex'] = df['pclass'].astype(str) + '_' + df['gender_code'].astype(str)
    df['pclass_sex'] = df['pclass_sex'].map({'1_0': 0, '1_1': 1, '2_0': 2, '2_1': 3, '3_0': 4, '3_1': 5})
    df['ticket_count'] = df['ticket'].map(ticket_counts)
    
    return df

def apply_te_vectorized(df_tr, df_te, target_s, col='surname', smoothing=10):
    global_mean = target_s.mean()
    tmp_tr = df_tr.copy()
    tmp_tr['target'] = target_s
    stats = tmp_tr.groupby(col)['target'].agg(['sum', 'count'])
    
    tmp_tr = tmp_tr.join(stats, on=col)
    tmp_tr['loo_sum'] = tmp_tr['sum'] - tmp_tr['target']
    tmp_tr['loo_count'] = tmp_tr['count'] - 1
    
    tr_te = (tmp_tr['loo_sum'] + smoothing * global_mean) / (tmp_tr['loo_count'] + smoothing)
    tr_te.loc[tmp_tr['loo_count'] == 0] = global_mean 
    
    tmp_te = df_te.copy()
    tmp_te = tmp_te.join(stats, on=col)
    te_te = (tmp_te['sum'] + smoothing * global_mean) / (tmp_te['count'] + smoothing)
    te_te = te_te.fillna(global_mean) 
    
    return tr_te.values, te_te.values

def add_te_features(df_tr, df_te, target_s):
    df_tr, df_te = df_tr.copy(), df_te.copy()
    for df in [df_tr, df_te]:
        df['surname'] = df['name'].apply(lambda x: x.split(',')[0].strip())
    
    tr_surv, te_surv = apply_te_vectorized(df_tr, df_te, target_s, col='surname', smoothing=10)
    df_tr['surname_surv'] = tr_surv
    df_te['surname_surv'] = te_surv
    return df_tr, df_te

features = ['pclass', 'age', 'fare', 'family_size', 'is_alone', 'title_code', 'pclass_sex', 'ticket_count', 'surname_surv']
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)

tr_processed = process_features(train)
te_processed = process_features(test)

print("--- [Experiment Ver 1: 불필요한 중복 피처 제거] ---")
tr_final, te_final = add_te_features(tr_processed, te_processed, y_tr)
X_tr, X_te = tr_final[features], te_final[features]

reset_seeds()
rf_model.fit(X_tr, y_tr)
print(f"Shape: {X_tr.shape} / {y_tr.shape}")
print(f"Ver 1 Train Score : {rf_model.score(X_tr, y_tr):.4f}")
print(f"Ver 1 Test Score  : {rf_model.score(X_te, y_te):.4f}\n")

importances = pd.Series(rf_model.feature_importances_, index=features)
print("[Ver 1 모델의 피처 중요도 (100점 만점)]")
print((importances * 100).sort_values(ascending=False).head(8))
print("\n" + "="*60 + "\n")

print("--- [Final Check: Stratified 5-Fold CV on Ver 1] ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
X_full = pd.concat([tr_processed, te_processed]).reset_index(drop=True)
y_full = pd.concat([y_tr, y_te]).reset_index(drop=True)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
    X_f_tr, y_f_tr = X_full.iloc[train_idx].copy(), y_full.iloc[train_idx].copy()
    X_f_val, y_f_val = X_full.iloc[val_idx].copy(), y_full.iloc[val_idx].copy()
    
    X_f_tr_te, X_f_val_te = add_te_features(X_f_tr, X_f_val, y_f_tr)
    
    rf_model.fit(X_f_tr_te[features], y_f_tr)
    score = rf_model.score(X_f_val_te[features], y_f_val)
    cv_scores.append(score)
    print(f"Fold {fold+1} Score: {score:.4f}")

print(f"✅ Final CV Score Mean: {np.mean(cv_scores):.4f} (Variance: {np.var(cv_scores):.6f})")
