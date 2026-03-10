import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ====================================================
# 1. 데이터 로드
# ====================================================
train = pd.read_csv("/kaggle/input/competitions/skn-27-ml/train.csv")
test = pd.read_csv("/kaggle/input/competitions/skn-27-ml/test.csv")
submission = pd.read_csv("/kaggle/input/competitions/skn-27-ml/submission.csv")

# 통합 데이터 생성
all_data = pd.concat([train, test], sort=False).reset_index(drop=True)

# ====================================================
# 2. Grandmaster's "Diet" Strategy (단순화 & 최적화)
# ====================================================

# (1) Title (호칭) 추출 및 매핑
all_data['title'] = all_data['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
all_data['title'] = all_data['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['title'] = all_data['title'].replace(['Mlle', 'Ms'], 'Miss')
all_data['title'] = all_data['title'].replace('Mme', 'Mrs')
all_data['title'] = all_data['title'].map({"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Rare": 4})

# (2) Embarked 정밀 보정 (이건 확실히 좋음!)
# 1등석 + $80 요금 = 무조건 'C' 탑승
all_data.loc[(all_data['embarked'].isnull()), 'embarked'] = 'C' 
all_data['embarked'] = all_data['embarked'].map({"S": 0, "C": 1, "Q": 2})

# (3) Fare Log 변환
all_data['fare'] = all_data['fare'].fillna(train['fare'].median())
all_data['fare'] = np.log1p(all_data['fare'])

# (4) Ticket Frequency (이것도 효과 있음!)
all_data['ticket_freq'] = all_data.groupby('ticket')['ticket'].transform('count')

# (5) Gender
all_data['gender'] = all_data['gender'].map({"male": 0, "female": 1})

# (6) [삭제] Deck 변수는 과감히 버립니다! (Overfitting 원인)
# (7) [삭제] Family Size도 Ticket Freq와 겹치므로 삭제합니다.

# 불필요 컬럼 삭제
all_data = all_data.drop(['name', 'ticket', 'cabin', 'passengerid', 'sibsp', 'parch', 'deck', 'family_size'], axis=1, errors='ignore')

# 데이터 분리
train_idx = len(train)
X = all_data[:train_idx].drop('survived', axis=1)
y = train['survived']
X_test = all_data[train_idx:].drop('survived', axis=1)

display(X.head())

# ====================================================
# 3. Stratified K-Fold Training
# ====================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 결과 저장소
oof_preds = np.zeros(X.shape[0])
test_preds = np.zeros(X_test.shape[0])
fold_scores = []

print("\n=== 🚀 5-Fold Training Start (Optimized) ===")

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
    y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # [Age Imputation] Fold 내부에서 수행
    age_means = X_tr.groupby('title')['age'].mean()
    X_tr['age'] = X_tr['age'].fillna(X_tr['title'].map(age_means))
    X_val['age'] = X_val['age'].fillna(X_val['title'].map(age_means))
    X_test_curr = X_test.copy()
    X_test_curr['age'] = X_test_curr['age'].fillna(X_test_curr['title'].map(age_means))
    
    # 모델 학습 (RandomForest)
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=7,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_tr, y_tr)
    
    # 검증
    val_pred = model.predict(X_val)
    score = accuracy_score(y_val, val_pred)
    fold_scores.append(score)
    test_preds += model.predict_proba(X_test_curr)[:, 1] / skf.get_n_splits()
    
    print(f"Fold {fold+1} Accuracy: {score:.4f}")

print("="*30)
print(f"🏆 Final Mean CV Score: {np.mean(fold_scores):.4f}")
print("="*30)

# ====================================================
# 4. 제출 파일 생성
# ====================================================
submission['survived'] = (test_preds >= 0.5).astype(int)
submission.to_csv("submission_0309_2.csv", index=False)
print("🎉 최종 제출 파일 생성 완료: submission_optimized.csv")