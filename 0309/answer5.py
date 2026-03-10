import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터 로드
# Kaggle 환경에 따라 경로를 수정하십시오 (예: /kaggle/input/...)
train = pd.read_csv("/kaggle/input/competitions/skn-27-ml/train.csv")
test = pd.read_csv("/kaggle/input/competitions/skn-27-ml/test.csv")
submission = pd.read_csv("/kaggle/input/competitions/skn-27-ml/submission.csv")

# 2. 전처리 및 피처 엔지니어링 함수
def preprocess_data(df):
    # Title extraction
    df['title'] = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['title'] = df['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['title'] = df['title'].replace(['Mlle', 'Ms'], 'Miss')
    df['title'] = df['title'].replace('Mme', 'Mrs')
    title_map = {"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Rare": 4}
    df['title_code'] = df['title'].map(title_map)
    
    # Embarked
    df['embarked'] = df['embarked'].fillna('C')
    df['embarked_code'] = df['embarked'].map({"S": 0, "C": 1, "Q": 2})
    
    # Fare
    df['fare'] = df['fare'].fillna(train['fare'].median())
    df['fare_log'] = np.log1p(df['fare'])
    
    # Gender
    df['gender_code'] = df['gender'].map({"male": 0, "female": 1})
    
    # Cabin & Deck (사각지대 보정 로직)
    def get_deck(row):
        if pd.notna(row['cabin']) and row['cabin'] != "":
            c = str(row['cabin']).split()[0]
            deck = c[0]
            num = "".join(filter(str.isdigit, c))
            return deck, (int(num) if num else 0)
        if row['pclass'] == 1:
            if row['fare'] > 100: return 'B', 0
            elif row['fare'] > 50: return 'C', 0
            else: return 'D', 0
        elif row['pclass'] == 2: return 'E', 0
        else: return 'F', 0

    df['deck'], df['room_num'] = zip(*df.apply(get_deck, axis=1))
    deck_list = sorted(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
    deck_map = {k: i for i, k in enumerate(deck_list)}
    df['deck_code'] = df['deck'].map(deck_map).fillna(-1)
    df['room_odd'] = df['room_num'].apply(lambda x: 1 if x % 2 != 0 else 0)
    
    # Family & Surname
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['surname'] = df['name'].apply(lambda x: x.split(',')[0].strip())
    
    return df

# 전처리 적용 및 나이 결측치 보간
train_proc = preprocess_data(train.copy())
test_proc = preprocess_data(test.copy())
train_proc['age'] = train_proc.groupby(['title_code', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))
test_proc['age'] = test_proc.groupby(['title_code', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))

# 3. 5-Fold Stratified CV (WCG 피처 엔지니어링 - Leakage Free)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_proc))
final_test_preds = np.zeros(len(test_proc))

features = ['pclass', 'age', 'title_code', 'embarked_code', 'fare_log', 
            'gender_code', 'deck_code', 'room_odd', 'family_size',
            'wcg_dead', 'wcg_survived']

print("--- Starting Hybrid Master CV (5-Fold) ---")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_proc, train_proc['survived'])):
    X_train_fold = train_proc.iloc[train_idx].copy()
    X_val_fold = train_proc.iloc[val_idx].copy()
    X_test_fold = test_proc.copy()
    
    # [WCG Logic: Train 폴드 정보만 사용하여 그룹 생존/사망 그룹 식별]
    dead_groups = X_train_fold[(X_train_fold['survived'] == 0) & ((X_train_fold['gender_code'] == 1) | (X_train_fold['age'] < 14))]
    dead_tickets = set(dead_groups['ticket'].unique())
    dead_surnames = set(dead_groups['surname'].unique())
    
    surv_groups = X_train_fold[(X_train_fold['survived'] == 1) & (X_train_fold['gender_code'] == 0) & (X_train_fold['age'] >= 14)]
    surv_tickets = set(surv_groups['ticket'].unique())
    surv_surnames = set(surv_groups['surname'].unique())
    
    for df in [X_train_fold, X_val_fold, X_test_fold]:
        df['wcg_dead'] = df.apply(lambda r: 1 if (r['ticket'] in dead_tickets or r['surname'] in dead_surnames) else 0, axis=1)
        df['wcg_survived'] = df.apply(lambda r: 1 if (r['ticket'] in surv_tickets or r['surname'] in surv_surnames) else 0, axis=1)
    
    X_tr, y_tr = X_train_fold[features], X_train_fold['survived']
    X_va = X_val_fold[features]
    X_te = X_test_fold[features]
    
    # 모델: RandomForest (하이퍼파라미터 튜닝 버전)
    model = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    
    oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
    final_test_preds += model.predict_proba(X_te)[:, 1] / 5

print(f"\n✅ Total Real CV Accuracy: {accuracy_score(train_proc['survived'], (oof_preds > 0.5).astype(int)):.4f}")

# 4. 제출 파일 생성
submission['survived'] = (final_test_preds > 0.5).astype(int)
submission.to_csv("submission_hybrid_master.csv", index=False)