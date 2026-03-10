import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터 로드
train = pd.read_csv("/kaggle/input/competitions/skn-27-ml/train.csv")
test = pd.read_csv("/kaggle/input/competitions/skn-27-ml/test.csv")
submission = pd.read_csv("/kaggle/input/competitions/skn-27-ml/submission.csv")

# 2. 전처리 (가장 점수 좋았던 0.8686 버전)
all_data = pd.concat([train, test], sort=False).reset_index(drop=True)

# Title & Mapping
all_data['title'] = all_data['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
all_data['title'] = all_data['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['title'] = all_data['title'].replace(['Mlle', 'Ms'], 'Miss')
all_data['title'] = all_data['title'].replace('Mme', 'Mrs')
all_data['title'] = all_data['title'].map({"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Rare": 4})

# Embarked Correction
all_data.loc[(all_data['embarked'].isnull()), 'embarked'] = 'C'
all_data['embarked'] = all_data['embarked'].map({"S": 0, "C": 1, "Q": 2})

# Fare Log
all_data['fare'] = all_data['fare'].fillna(train['fare'].median())
all_data['fare'] = np.log1p(all_data['fare'])

# Ticket Frequency
all_data['ticket_freq'] = all_data.groupby('ticket')['ticket'].transform('count')

# Gender
all_data['gender'] = all_data['gender'].map({"male": 0, "female": 1})

# Drop unused
all_data = all_data.drop(['name', 'cabin', 'passengerid', 'sibsp', 'parch'], axis=1, errors='ignore')

# 3. 모델 학습 및 1차 예측 (RandomForest)
train_idx = len(train)
X = all_data[:train_idx].drop(['survived', 'ticket'], axis=1) # Ticket은 학습에 방해되니 일단 제외
y = train['survived']
X_test = all_data[train_idx:].drop(['survived', 'ticket'], axis=1)

model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
model.fit(X, y)
initial_preds = model.predict(X_test)

# 4. [핵심] 후처리 (Family/Group Correction)
# 다시 Ticket 정보를 가져옴
X_test_with_ticket = all_data[train_idx:].copy()
X_test_with_ticket['survived_pred'] = initial_preds

# 가족/그룹 정보 구축 (Train + Test 전체에서)
full_data = pd.concat([train, test], sort=False).reset_index(drop=True)
full_data['surname'] = full_data['name'].apply(lambda x: x.split(',')[0].strip())

# (1) 죽은 여성/아이 찾기 (Train에서)
# 여성(female)이거나 15세 미만 아이(master)인데 사망한 그룹 찾기
dead_women_children_groups = []
for idx, row in full_data[:train_idx].iterrows():
    if row['survived'] == 0:
        if (row['gender'] == 'female') or (row['age'] < 15):
             dead_women_children_groups.append(row['ticket'])
             dead_women_children_groups.append(row['surname'])

dead_women_children_groups = list(set(dead_women_children_groups))

# (2) 산 남성 찾기 (Train에서)
# 남성(male)이면서 성인(15세 이상)인데 생존한 그룹 찾기
survived_men_groups = []
for idx, row in full_data[:train_idx].iterrows():
    if row['survived'] == 1:
        if (row['gender'] == 'male') and (row['age'] >= 15):
             survived_men_groups.append(row['ticket'])
             survived_men_groups.append(row['surname'])

survived_men_groups = list(set(survived_men_groups))

# 5. 예측값 강제 수정 (Correction)
print("=== 🛠️ Post-Processing Correction Applied ===")
final_preds = X_test_with_ticket['survived_pred'].values
modified_count = 0

for idx, row in X_test_with_ticket.iterrows():
    passenger_idx = idx - train_idx # 0부터 시작하는 인덱스
    
    # 해당 승객의 Ticket이나 Surname
    p_ticket = full_data.loc[idx, 'ticket']
    p_surname = full_data.loc[idx, 'surname']
    
    # Rule 1: 이 승객이 '죽은 여성/아이 그룹'에 속한다면 -> 사망(0)으로 변경
    if (p_ticket in dead_women_children_groups) or (p_surname in dead_women_children_groups):
        if final_preds[passenger_idx] == 1:
            final_preds[passenger_idx] = 0
            modified_count += 1
            # print(f"ID {full_data.loc[idx, 'passengerid']}: Force Dead (Family Died)")

    # Rule 2: 이 승객이 '산 남성 그룹'에 속한다면 -> 생존(1)으로 변경
    if (p_ticket in survived_men_groups) or (p_surname in survived_men_groups):
        if final_preds[passenger_idx] == 0:
            final_preds[passenger_idx] = 1
            modified_count += 1
            # print(f"ID {full_data.loc[idx, 'passengerid']}: Force Survived (Family Lived)")

print(f"총 {modified_count}명의 예측이 가족 정보에 의해 수정되었습니다.")
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

# 6. 제출 파일 생성
submission['survived'] = final_preds.astype(int)
submission.to_csv("submission_0309_03.csv", index=False)
print("🎉 최종 수정본 생성 완료: submission_corrected_092.csv")