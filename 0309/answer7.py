import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import warnings

warnings.filterwarnings('ignore')

# [1] 데이터 로드
# 환경에 맞춰 경로를 수정하세요. (예: pd.read_csv("train.csv"))
train = pd.read_csv("/kaggle/input/competitions/skn-27-ml/train.csv")
test = pd.read_csv("/kaggle/input/competitions/skn-27-ml/test.csv")
submission = pd.read_csv("/kaggle/input/competitions/skn-27-ml/submission.csv")

# [2] 통합 전처리 함수 (Feature Engineering)
def preprocess_all(df):
    # 1. Title 추출 및 그룹화
    df['title'] = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['title'] = df['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['title'] = df['title'].replace(['Mlle', 'Ms'], 'Miss')
    df['title'] = df['title'].replace('Mme', 'Mrs')
    title_map = {"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Rare": 4}
    df['title_code'] = df['title'].map(title_map)
    
    # 2. Gender & Embarked 인코딩
    df['gender_code'] = df['gender'].map({"male": 0, "female": 1})
    df['embarked_code'] = df['embarked'].fillna('C').map({"S": 0, "C": 1, "Q": 2})
    
    # 3. Fare 보정 및 파생 변수
    df['fare'] = df['fare'].fillna(train['fare'].median())
    df['fare_log'] = np.log1p(df['fare'])
    df['fare_per_pclass'] = df['fare'] / df['pclass'] # 객실 등급 대비 지불 가치
    
    # 4. 선실(Cabin) 사각지대 복구 로직
    def get_deck_info(row):
        if pd.notna(row['cabin']) and row['cabin'] != "":
            c = str(row['cabin']).split()[0]
            deck, num = c[0], "".join(filter(str.isdigit, c))
            return deck, (int(num) if num else 0)
        # 역사적/통계적 근거에 기반한 결측치 추론
        if row['pclass'] == 1:
            return ('B', 0) if row['fare'] > 100 else ('C', 0)
        return ('E' if row['pclass'] == 2 else 'F', 0)

    df['deck'], df['room_num'] = zip(*df.apply(get_deck_info, axis=1))
    deck_map = {k: i for i, k in enumerate(sorted(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']))}
    df['deck_code'] = df['deck'].map(deck_map).fillna(-1)
    df['room_odd'] = df['room_num'].apply(lambda x: 1 if x % 2 != 0 else 0) # 좌우현 위치
    
    # 5. 가족/그룹 식별 정보
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['surname'] = df['name'].apply(lambda x: x.split(',')[0].strip())
    
    return df

# 전처리 적용
train_proc = preprocess_all(train.copy())
test_proc = preprocess_all(test.copy())

# 나이(Age) 정밀 보간 (Title + Pclass 조합 중앙값)
train_proc['age'] = train_proc.groupby(['title_code', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))
test_proc['age'] = test_proc.groupby(['title_code', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))

# [3] 5-Fold CV + Ensemble 전략 (Leakage Zero)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
features = ['pclass', 'age', 'title_code', 'embarked_code', 'fare_log', 'fare_per_pclass',
            'gender_code', 'deck_code', 'room_odd', 'family_size',
            'wcg_dead', 'wcg_survived']

oof_preds = np.zeros(len(train_proc))
final_test_preds = np.zeros(len(test_proc))

print("--- [Titanic Master Pipeline] CV & Ensemble Start ---")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_proc, train_proc['survived'])):
    X_tr_f = train_proc.iloc[train_idx].copy()
    X_va_f = train_proc.iloc[val_idx].copy()
    X_te_f = test_proc.copy()
    
    # [WCG Feature Engineering: Fold 내부의 Train 정보만 활용]
    # 1) 사망 여성/아이 그룹 식별
    dead_group = X_tr_f[(X_tr_f['survived'] == 0) & ((X_tr_f['gender_code'] == 1) | (X_tr_f['age'] < 14))]
    dead_tix, dead_names = set(dead_group['ticket']), set(dead_group['surname'])
    
    # 2) 생존 성인 남성 그룹 식별
    surv_group = X_tr_f[(X_tr_f['survived'] == 1) & (X_tr_f['gender_code'] == 0) & (X_tr_f['age'] >= 14)]
    surv_tix, surv_names = set(surv_group['ticket']), set(surv_group['surname'])
    
    # 피처 생성
    for df in [X_tr_f, X_va_f, X_te_f]:
        df['wcg_dead'] = df.apply(lambda r: 1 if (r['ticket'] in dead_tix or r['surname'] in dead_names) else 0, axis=1)
        df['wcg_survived'] = df.apply(lambda r: 1 if (r['ticket'] in surv_tix or r['surname'] in surv_names) else 0, axis=1)
    
    # 모델 1: Tuned Random Forest
    rf = RandomForestClassifier(n_estimators=1000, max_depth=5, min_samples_split=3, min_samples_leaf=2, random_state=42, n_jobs=-1)
    
    # 모델 2: Tuned Extra Trees (다양성 확보)
    et = ExtraTreesClassifier(n_estimators=1000, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
    
    # 학습
    rf.fit(X_tr_f[features], X_tr_f['survived'])
    et.fit(X_tr_f[features], X_tr_f['survived'])
    
    # 블렌딩 예측 (RF 70% + ET 30%)
    fold_prob = (rf.predict_proba(X_va_f[features])[:, 1] * 0.8) + (et.predict_proba(X_va_f[features])[:, 1] * 0.3)
    oof_preds[val_idx] = fold_prob
    
    # 테스트 데이터 예측 누적
    test_prob = (rf.predict_proba(X_te_f[features])[:, 1] * 0.8) + (et.predict_proba(X_te_f[features])[:, 1] * 0.3)
    final_test_preds += test_prob / 5
    
    fold_acc = accuracy_score(train_proc.iloc[val_idx]['survived'], (fold_prob > 0.5).astype(int))
    print(f"Fold {fold+1} Accuracy: {fold_acc:.4f}")

# 최종 검증 성능 보고
total_cv = accuracy_score(train_proc['survived'], (oof_preds > 0.5).astype(int))
print(f"\n✅ Total Cross-Validation Accuracy: {total_cv:.4f}")

# [4] 최종 제출 파일 저장
submission['survived'] = (final_test_preds > 0.5).astype(int)
submission.to_csv("submission_master_final_v2.csv", index=False)
print("🚀 [Master Strategy] Submission file is ready. LB 0.90+ Challenge!")