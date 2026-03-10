import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

# 1. 데이터 로드
train = pd.read_csv('/kaggle/input/competitions/skn-27-ml/train.csv')
test = pd.read_csv('/kaggle/input/competitions/skn-27-ml/test.csv')

def build_final_features(df_train, df_test):
    train_df = df_train.copy()
    test_df = df_test.copy()
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    # --- Phase 1: 결측치 처리 (인사이트 반영) ---
    df['embarked'] = df['embarked'].fillna('C') 
    # $80 지불 1등석 승객의 Fare 결측치 처리
    median_fare_3S = df[(df['pclass']==3) & (df['embarked']=='S') & (df['sibsp']==0)]['fare'].median()
    df['fare'] = df['fare'].fillna(median_fare_3S)
    
    # Title 추출
    df['Title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Don', 'Capt', 'Dona', 'Lady']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    
    # 나이 결측치를 호칭별 중앙값으로 채움
    age_by_title = df.groupby('Title')['age'].median()
    df['age'] = df.apply(lambda row: age_by_title[row['Title']] if pd.isna(row['age']) else row['age'], axis=1)
    
    # Cabin Deck (M: Missing)
    df['Deck'] = df['cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
    df['Deck'] = df['Deck'].replace('T', 'A')
    
    # --- Phase 2: 파생 변수 생성 (Group Survival 치트키) ---
    df['Family_Size'] = df['sibsp'] + df['parch'] + 1
    
    # Ticket Grouping
    ticket_counts = df['ticket'].value_counts()
    df['Group_Size'] = df['ticket'].map(ticket_counts)
    
    # Leave-One-Out Ticket Survival Rate (데이터 누수 방지 버전)
    ticket_surv_sum = train_df.groupby('ticket')['survived'].sum()
    ticket_surv_count = train_df.groupby('ticket')['survived'].count()
    df['t_surv_sum'] = df['ticket'].map(ticket_surv_sum).fillna(0)
    df['t_surv_count'] = df['ticket'].map(ticket_surv_count).fillna(0)

    def get_loo_survival(row):
        if row['t_surv_count'] == 0: return -1
        if pd.notnull(row['survived']): # Train 데이터
            if row['t_surv_count'] > 1:
                return (row['t_surv_sum'] - row['survived']) / (row['t_surv_count'] - 1)
            else:
                return -1
        else: # Test 데이터
            return row['t_surv_sum'] / row['t_surv_count'] if row['t_surv_count'] > 0 else -1

    df['Group_Survival'] = df.apply(get_loo_survival, axis=1)
    
    # --- Phase 3: 인코딩 및 정리 ---
    # gender_female을 명시적으로 생성 (0: 남성, 1: 여성)
    df['gender_female'] = (df['gender'] == 'female').astype(int)
    
    # 불필요한 원본 문자열 제거 (name은 나중에 후처리에 써야 하므로 test_df용으로 따로 보관)
    # 인코딩할 변수들 선택
    df_encoded = pd.get_dummies(df, columns=['embarked', 'Title', 'Deck'], drop_first=True)
    
    # 특수문자 제거 (Column 명 정리)
    df_encoded = df_encoded.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    
    # 원본 name과 passengerid는 학습에서 제외하되 데이터프레임에는 유지 (후처리용)
    train_clean = df_encoded.iloc[:len(train_df)].copy()
    test_clean = df_encoded.iloc[len(train_df):].copy()
    
    return train_clean, test_clean

# 2. 전처리 수행
train_clean, test_clean = build_final_features(train, test)

# 3. 모델 학습 준비
# 학습에 사용할 피처 선정 (name, ticket, cabin, gender, passengerid 등 제외)
features = [col for col in train_clean.columns if col not in ['survived', 'passengerid', 'name', 'ticket', 'cabin', 'gender', 't_surv_sum', 't_surv_count']]

X = train_clean[features]
y = train_clean['survived']
X_test = test_clean[features]

# 4. Cross Validation (Stratified 5-Fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gb_model = GradientBoostingClassifier(learning_rate=0.03, n_estimators=400, max_depth=4, max_features=0.6, subsample=0.8, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for train_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    gb_model.fit(X_tr, y_tr)
    oof_preds[val_idx] = gb_model.predict(X_va)
    test_preds += gb_model.predict_proba(X_test)[:, 1] / 5

print(f"Local CV Accuracy: {accuracy_score(y, oof_preds):.4f}")

# 5. 최종 결과 도출 및 후처리 (Post-Processing)
test_clean['survived'] = (test_preds > 0.5).astype(int)

# Rule 1: 1등석의 어린 소년(Master)은 생존으로 보정 (모델이 놓친 케이스)
test_clean.loc[(test_clean['name'].str.contains('Master.')) & (test_clean['pclass'] == 1), 'survived'] = 1

# Rule 2: 3등석 대가족 여성 중 일행이 전멸한 경우(Group_Survival == 0) 사망으로 보정
test_clean.loc[(test_clean['pclass'] == 3) & (test_clean['gender_female'] == 1) & (test_clean['Group_Survival'] == 0), 'survived'] = 0

# 6. 파일 저장
test_clean[['passengerid', 'survived']].to_csv('final_submission.csv', index=False)
print("성공! 'final_submission.csv' 파일이 생성되었습니다.")

