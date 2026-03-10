import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# [1] 데이터 로드
# ─────────────────────────────────────────
train_raw = pd.read_csv("/kaggle/input/competitions/skn-27-ml/train.csv")
test_raw  = pd.read_csv("/kaggle/input/competitions/skn-27-ml/test.csv")
submission = pd.read_csv("/kaggle/input/competitions/skn-27-ml/submission.csv")

# ─────────────────────────────────────────
# [2] 기초 전처리 (New Features 추가)
# ─────────────────────────────────────────
def base_preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Title
    df['title'] = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['title'] = df['title'].replace(rare, 'Rare').replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['title_code'] = df['title'].map({"Master":0,"Miss":1,"Mr":2,"Mrs":3,"Rare":4})

    # Gender / Embarked
    df['gender_code']   = df['gender'].map({"male":0,"female":1})
    df['embarked_code'] = df['embarked'].fillna('S').map({"S":0,"C":1,"Q":2})

    # Family & Name Length (New)
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone']    = (df['family_size'] == 1).astype(int)
    df['name_len']    = df['name'].apply(len) # 이름 길이는 지위를 상징하기도 함

    # Cabin & Side (New)
    df['cabin_known'] = df['cabin'].notna().astype(int)
    df['cabin_deck']  = df['cabin'].str[0].fillna('U')
    deck_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8,'U':0}
    df['cabin_deck_code'] = df['cabin_deck'].map(deck_map).fillna(0).astype(int)
    
    # 객실 번호의 홀/짝(좌우현) 추출
    df['cabin_num'] = df['cabin'].str.extract('(\d+)').astype(float)
    df['cabin_side'] = df['cabin_num'] % 2
    df['cabin_side'] = df['cabin_side'].fillna(-1)

    # Surname
    df['surname'] = df['name'].apply(lambda x: x.split(',')[0].strip())
    return df

# ─────────────────────────────────────────
# [3] Fold-safe 파생 피처 (Ticket Group Stats 추가)
# ─────────────────────────────────────────
def fold_features(df_tr, df_va, df_te, fare_median):
    # Ticket group size & Fare stats
    tgs = df_tr['ticket'].value_counts().to_dict()
    for df in [df_tr, df_va, df_te]:
        df['ticket_group_size'] = df['ticket'].map(tgs).fillna(1)
        df['fare'] = df['fare'].replace(0, fare_median).fillna(fare_median) # 0원 요금 보정
        df['fare_per_person'] = df['fare'] / df['ticket_group_size']
        fare_cap = df_tr['fare'].quantile(0.99)
        df['fare_log'] = np.log1p(df['fare'].clip(upper=fare_cap))
        # [New] 요금 순위
        df['fare_rank'] = df.groupby('pclass')['fare'].rank(pct=True)

    # Age 보간
    age_lut = df_tr.groupby(['title_code','pclass','gender_code'])['age'].median().to_dict()
    def fill_age(r):
        if not pd.isna(r['age']): return r['age']
        return age_lut.get((r['title_code'], r['pclass'], r['gender_code']), 28)
    
    for df in [df_tr, df_va, df_te]:
        df['age'] = df.apply(fill_age, axis=1)
        df['age_log'] = np.log1p(df['age'])
        df['age_x_pclass'] = df['age'] * df['pclass']
        df['fare_x_pclass'] = df['fare_log'] * df['pclass']
        df['gender_x_pclass'] = df['gender_code'] * df['pclass']

    return df_tr, df_va, df_te

# ─────────────────────────────────────────
# [4] 피처 목록 및 모델 정의
# ─────────────────────────────────────────
FEATURES = [
    'pclass', 'age_log', 'title_code', 'embarked_code', 'fare_log', 'fare_per_person', 'fare_rank',
    'gender_code', 'family_size', 'is_alone', 'name_len', 'cabin_known', 'cabin_deck_code', 'cabin_side',
    'age_x_pclass', 'fare_x_pclass', 'gender_x_pclass', 'ticket_group_size'
]

BASE_MODELS = {
    'rf': RandomForestClassifier(n_estimators=800, max_depth=6, random_state=42, n_jobs=-1),
    'et': ExtraTreesClassifier(n_estimators=800, max_depth=7, random_state=42, n_jobs=-1),
    'gbm': GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42),
    'xgb': XGBClassifier(n_estimators=500, learning_rate=0.02, max_depth=4, random_state=42, eval_metric='logloss'),
    'lgbm': LGBMClassifier(n_estimators=500, learning_rate=0.02, max_depth=4, random_state=42, verbose=-1),
    'svm': SVC(probability=True, random_state=42)
}

# ─────────────────────────────────────────
# [5] Stratified split & Stacking Loop
# ─────────────────────────────────────────
# 요청하신 대로 분할 시 stratify를 고려한 KFold 적용
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fare_median = train_raw['fare'].median()

n_train, n_test, n_models = len(train_raw), len(test_raw), len(BASE_MODELS)
oof_preds = np.zeros((n_train, n_models))
test_preds = np.zeros((n_test, n_models))
scaler = StandardScaler()

print("=" * 55)
print("  Stacking Ensemble V12 (Stratified & New Features)")
print("=" * 55)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_raw, train_raw['survived'])):
    # 데이터를 나눌 때 이미 stratify가 적용됨
    df_tr = base_preprocess(train_raw.iloc[tr_idx])
    df_va = base_preprocess(train_raw.iloc[va_idx])
    df_te = base_preprocess(test_raw)

    df_tr, df_va, df_te = fold_features(df_tr, df_va, df_te, fare_median)

    X_tr, y_tr = df_tr[FEATURES].values, df_tr['survived'].values
    X_va, y_va = df_va[FEATURES].values, df_va['survived'].values
    X_te = df_te[FEATURES].values

    # Scaling
    X_tr_s, X_va_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    for i, (name, model) in enumerate(BASE_MODELS.items()):
        X_fit = X_tr_s if name == 'svm' else X_tr
        X_val = X_va_s if name == 'svm' else X_va
        X_tst = X_te_s if name == 'svm' else X_te
        
        model.fit(X_fit, y_tr)
        oof_p = model.predict_proba(X_val)[:, 1]
        oof_preds[va_idx, i] = oof_p
        test_preds[:, i] += model.predict_proba(X_tst)[:, 1] / 5

    print(f"Fold {fold+1} Completed.")

# ─────────────────────────────────────────
# [6] Meta Model & Final Submission
# ─────────────────────────────────────────
meta = LogisticRegression(C=1.0, random_state=42)
meta.fit(oof_preds, train_raw['survived'])

# 임계값 최적화
oof_final = meta.predict_proba(oof_preds)[:, 1]
best_thr = 0.5
best_acc = 0
for thr in np.arange(0.4, 0.6, 0.01):
    acc = accuracy_score(train_raw['survived'], (oof_final > thr).astype(int))
    if acc > best_acc:
        best_acc, best_thr = acc, thr

print(f"\n🏆 Best OOF Acc: {best_acc:.4f} at Threshold: {best_thr:.2f}")

# 제출 (0.88을 기록했던 0.58 수동 적용도 고려해보세요)
final_preds = meta.predict_proba(test_preds)[:, 1]
submission['survived'] = (final_preds > 0.58).astype(int) # 0.58 강제 고정
submission.to_csv("submission_v12_stratified.csv", index=False)