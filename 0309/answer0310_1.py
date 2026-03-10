import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# ⚙️ 1. 시드 고정 및 데이터 준비
# ==============================================================================
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

# 검증용 데이터 분리
train, test = train_test_split(ori_train, test_size=0.2, random_state=42, stratify=ori_train['survived'])

y_tr = train['survived']
y_te = test['survived']
train = train.drop('survived', axis=1)
test = test.drop('survived', axis=1)

# 전체 데이터 (티켓 카운트용)
all_tickets = pd.concat([train['ticket'], test['ticket'], ori_te['ticket']])
ticket_counts = all_tickets.value_counts()

# [사각지대 보완] 결측치 맵핑을 위한 통계값 (Data Leakage 방지를 위해 Train에서만 추출)
fare_map = train.groupby('pclass')['fare'].median()

# Age 보간을 위해 임시로 호칭 추출
temp_title = train['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
age_map = train.assign(title=temp_title).groupby(['title', 'pclass'])['age'].median()
global_age_median = train['age'].median()


# ==============================================================================
# 💡 2. [모듈화] 피처 생성 함수들 (당신의 정교한 통찰력 반영)
# ==============================================================================
def process_baseline(df):
    """V0: 지능적인 결측치 처리 및 기본 인코딩"""
    df = df.copy()
    
    # 1. Embarked: 1등석 80달러 탑승객의 패턴을 반영하여 'C'로 고정
    df['embarked'] = df['embarked'].fillna('C')
    
    # 2. Fare: 전체 중앙값이 아닌, 해당 승객의 객실 등급(Pclass) 중앙값으로 처리
    df['fare'] = df.apply(lambda x: fare_map[x['pclass']] if pd.isna(x['fare']) else x['fare'], axis=1)
    
    # 3. Age: Title과 Pclass를 교차하여 정교하게 채움
    temp_title_df = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    def get_age(row, title):
        if pd.isna(row['age']):
            try: return age_map[title, row['pclass']]
            except: return global_age_median
        return row['age']
    
    df['age'] = [get_age(row, t) for (_, row), t in zip(df.iterrows(), temp_title_df)]
    
    df['gender_code'] = df['gender'].map({"male": 0, "female": 1})
    df['embarked_code'] = df['embarked'].map({"S": 0, "C": 1, "Q": 2})
    return df

def add_power_features(df):
    """V1: 가족 규모 및 호칭"""
    df = df.copy()
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    
    df['title'] = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3} 
    df['title_code'] = df['title'].map(title_map).fillna(4)
    return df

def add_ultimate_features(df):
    """V2: 객실등급+성별 콤보 및 티켓 동행자 수"""
    df = df.copy()
    df['pclass_sex'] = df['pclass'].astype(str) + '_' + df['gender_code'].astype(str)
    df['pclass_sex'] = df['pclass_sex'].map({'1_0': 0, '1_1': 1, '2_0': 2, '2_1': 3, '3_0': 4, '3_1': 5})
    df['ticket_count'] = df['ticket'].map(ticket_counts)
    return df

def add_grandmaster_features(df, train_df_for_encoding, y_train_for_encoding):
    """V3: [Kaggle GM] Cabin Deck 추출 및 Surname 타겟 인코딩"""
    df = df.copy()
    
    # 1. Deck 추출 (버려졌던 정보 복구)
    df['deck'] = df['cabin'].astype(str).str[0]
    df['deck'] = df['deck'].replace('n', 'U')
    deck_map = {'U':0, 'C':1, 'B':2, 'D':3, 'E':4, 'A':5, 'F':6, 'G':7, 'T':8}
    df['deck_code'] = df['deck'].map(deck_map).fillna(0)
    
    # 2. Surname 추출 및 Target Encoding (운명 공동체 파악)
    df['surname'] = df['name'].apply(lambda x: x.split(',')[0].strip())
    
    # Data Leakage 방지: 무조건 훈련 데이터(train_df)의 생존율만 매핑
    train_surname = train_df_for_encoding['name'].apply(lambda x: x.split(',')[0].strip())
    surname_surv_map = y_train_for_encoding.groupby(train_surname).mean()
    global_mean = y_train_for_encoding.mean()
    
    df['surname_surv'] = df['surname'].map(surname_surv_map).fillna(global_mean)
    return df


# ==============================================================================
# 🚀 3. [실험 진행: V0 -> V1 -> V2 -> V3]
# ==============================================================================
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)

# ---------------------------------------------------------
print("--- [Experiment V0: Baseline Features (Refined Imputation)] ---")
tr_v0 = process_baseline(train)
te_v0 = process_baseline(test)

features_v0 = ['pclass', 'gender_code', 'age', 'fare', 'embarked_code', 'sibsp', 'parch']
X_tr_v0, X_te_v0 = tr_v0[features_v0], te_v0[features_v0]

reset_seeds()
rf_model.fit(X_tr_v0, y_tr)
print(f"Shape: {X_tr_v0.shape} / {y_tr.shape}")
print(f"V0 Train Score : {rf_model.score(X_tr_v0, y_tr):.4f}")
print(f"V0 Test Score  : {rf_model.score(X_te_v0, y_te):.4f}\n")

# ---------------------------------------------------------
print("--- [Experiment V1: Power Features] ---")
tr_v1 = add_power_features(tr_v0)
te_v1 = add_power_features(te_v0)

features_v1 = features_v0 + ['family_size', 'is_alone', 'title_code']
X_tr_v1, X_te_v1 = tr_v1[features_v1], te_v1[features_v1]

reset_seeds()
rf_model.fit(X_tr_v1, y_tr)
print(f"Shape: {X_tr_v1.shape} / {y_tr.shape}")
print(f"V1 Train Score : {rf_model.score(X_tr_v1, y_tr):.4f}")
print(f"V1 Test Score  : {rf_model.score(X_te_v1, y_te):.4f}\n")

# ---------------------------------------------------------
print("--- [Experiment V2: Ultimate Features] ---")
tr_v2 = add_ultimate_features(tr_v1)
te_v2 = add_ultimate_features(te_v1)

features_v2 = features_v1 + ['pclass_sex', 'ticket_count']
X_tr_v2, X_te_v2 = tr_v2[features_v2], te_v2[features_v2]

reset_seeds()
rf_model.fit(X_tr_v2, y_tr)
print(f"Shape: {X_tr_v2.shape} / {y_tr.shape}")
print(f"V2 Train Score : {rf_model.score(X_tr_v2, y_tr):.4f}")
print(f"V2 Test Score  : {rf_model.score(X_te_v2, y_te):.4f}\n")

# ---------------------------------------------------------
print("--- [Experiment V3: Grandmaster Features (Deck & Surname)] ---")
tr_v3 = add_grandmaster_features(tr_v2, train, y_tr)
te_v3 = add_grandmaster_features(te_v2, train, y_tr)

features_v3 = features_v2 + ['deck_code', 'surname_surv']
X_tr_v3, X_te_v3 = tr_v3[features_v3], te_v3[features_v3]

reset_seeds()
rf_model.fit(X_tr_v3, y_tr)
print(f"Shape: {X_tr_v3.shape} / {y_tr.shape}")
print(f"V3 Train Score : {rf_model.score(X_tr_v3, y_tr):.4f}")
print(f"V3 Test Score  : {rf_model.score(X_te_v3, y_te):.4f}\n")

importances = pd.Series(rf_model.feature_importances_, index=features_v3)
print("[V3 모델의 피처 중요도 (100점 만점)]")
print((importances * 100).sort_values(ascending=False).head(8))
print("\n" + "="*60 + "\n")

# ==============================================================================
# 🎯 [최종 점검] 왜 단일 분리(train_test_split)를 믿으면 안 되는가? (K-Fold 증명)
# ==============================================================================
print("--- [Kaggle GM's Final Check: Stratified 5-Fold CV on V3] ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

X_full = pd.concat([X_tr_v3, X_te_v3]).reset_index(drop=True)
y_full = pd.concat([y_tr, y_te]).reset_index(drop=True)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
    X_f_tr, y_f_tr = X_full.iloc[train_idx], y_full.iloc[train_idx]
    X_f_val, y_f_val = X_full.iloc[val_idx], y_full.iloc[val_idx]
    
    rf_model.fit(X_f_tr, y_f_tr)
    score = rf_model.score(X_f_val, y_f_val)
    cv_scores.append(score)
    print(f"Fold {fold+1} Score: {score:.4f}")

print(f"✅ Final CV Score Mean: {np.mean(cv_scores):.4f} (Variance: {np.var(cv_scores):.6f})")