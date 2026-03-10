import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터 로드
train = pd.read_csv("/kaggle/input/competitions/skn-27-ml/train.csv")
test = pd.read_csv("/kaggle/input/competitions/skn-27-ml/test.csv")
all_data = pd.concat([train, test], sort=False).reset_index(drop=True)

# 2. Cabin & Deck 정밀 복구 (역사적 가중치 반영)
def repair_cabin(row):
    if pd.notna(row['cabin']) and row['cabin'] != "":
        parts = str(row['cabin']).split()
        first = parts[0]
        deck = first[0]
        num = "".join(filter(str.isdigit, first))
        return deck, (int(num) if num else 0)
    
    # 역사적 사실 기반: Pclass와 Fare에 따른 데크 추론
    if row['pclass'] == 1:
        if row['fare'] > 80: return 'B', 0
        elif row['fare'] > 50: return 'C', 0
        else: return 'D', 0
    elif row['pclass'] == 2: return 'E', 0
    else: return 'F', 0

all_data['deck'], all_data['room_num'] = zip(*all_data.apply(repair_cabin, axis=1))
all_data['room_odd'] = all_data['room_num'].apply(lambda x: 1 if x % 2 != 0 else 0)

# 3. Title & Age (Title + Pclass 세분화 보간)
all_data['title'] = all_data['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
all_data['title'] = all_data['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['title'] = all_data['title'].replace(['Mlle', 'Ms'], 'Miss')
all_data['title'] = all_data['title'].replace('Mme', 'Mrs')
all_data['age'] = all_data.groupby(['title', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))

# 4. Family Survival Grouping (Leakage 방지)
all_data['surname'] = all_data['name'].apply(lambda x: x.split(',')[0].strip())
train_df = all_data[:len(train)].copy()
train_df['survived'] = train['survived']

# 성씨와 티켓 기반 생존율 맵핑
surname_surv = train_df.groupby('surname')['survived'].mean()
ticket_surv = train_df.groupby('ticket')['survived'].mean()

all_data['surname_surv_rate'] = all_data['surname'].map(surname_surv).fillna(0.5)
all_data['ticket_surv_rate'] = all_data['ticket'].map(ticket_surv).fillna(0.5)

# 5. 인코딩 및 전처리
all_data['gender'] = all_data['gender'].map({"male": 0, "female": 1})
all_data['embarked'] = all_data['embarked'].fillna('S').map({"S": 0, "C": 1, "Q": 2})
all_data['deck'] = all_data['deck'].map({k: i for i, k in enumerate(sorted(all_data['deck'].unique()))})
all_data['title'] = all_data['title'].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4})

# 불필요 변수 제거
X = all_data[:len(train)].drop(['survived', 'name', 'cabin', 'ticket', 'surname', 'passengerid'], axis=1)
y = train['survived']
X_test = all_data[len(train):].drop(['survived', 'name', 'cabin', 'ticket', 'surname', 'passengerid'], axis=1)

# 6. 5-Fold Stratified CV & Ensemble
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
final_test_preds = np.zeros(len(X_test))

print(f"--- Starting {n_splits}-Fold Cross Validation ---")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # XGBoost Hyperparameters (Kaggle 최적화 세팅)
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=1,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=100
    )
    
    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
    
    # 검증 데이터 예측 (OOF)
    oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
    
    # 테스트 데이터 예측 (각 폴드의 모델 결과 누적)
    final_test_preds += model.predict_proba(X_test)[:, 1] / n_splits
    
    fold_acc = accuracy_score(y_val_fold, (oof_preds[val_idx] > 0.5).astype(int))
    print(f"Fold {fold+1} Accuracy: {fold_acc:.4f}")

# 전체 검증 성능
overall_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
print(f"\n✅ Total CV Accuracy: {overall_acc:.4f}")

# 7. 제출 파일 생성
submission = pd.read_csv("/kaggle/input/competitions/skn-27-ml/submission.csv")
submission['survived'] = (final_test_preds > 0.5).astype(int)
submission.to_csv("submission_0309_04.csv", index=False)
print("\n🚀 5-Fold Ensemble Submission File Saved!")