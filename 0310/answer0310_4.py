import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# [1] 데이터 로드
# ─────────────────────────────────────────
train = pd.read_csv("/kaggle/input/competitions/skn-27-ml/train.csv")
test  = pd.read_csv("/kaggle/input/competitions/skn-27-ml/test.csv")
submission = pd.read_csv("/kaggle/input/competitions/skn-27-ml/submission.csv")

# ─────────────────────────────────────────
# [2] 기초 전처리 (타겟 누수 없는 정적 피처)
# ─────────────────────────────────────────
def base_preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.lower()

    # ── Title ──
    df['title'] = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['title'] = (df['title']
                   .replace(rare, 'Rare')
                   .replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'}))
    df['title_code'] = df['title'].map({"Master":0,"Miss":1,"Mr":2,"Mrs":3,"Rare":4})

    # ── Gender / Embarked ──
    df['gender_code']   = df['gender'].map({"male":0,"female":1})
    df['embarked_code'] = df['embarked'].fillna('S').map({"S":0,"C":1,"Q":2})

    # ── Family ──
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone']    = (df['family_size'] == 1).astype(int)
    df['family_type'] = pd.cut(df['family_size'],
                                bins=[0,1,4,20],
                                labels=[0,1,2]).astype(int)   # 혼자/소가족/대가족

    # ── Cabin ──
    df['cabin_known'] = df['cabin'].notna().astype(int)
    df['cabin_deck']  = df['cabin'].str[0].fillna('U')        # A~G or U(unknown)
    deck_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8,'U':0}
    df['cabin_deck_code'] = df['cabin_deck'].map(deck_map).fillna(0).astype(int)

    # ── Ticket prefix ──
    def ticket_prefix(t):
        t = str(t).upper().replace('.','').replace('/','').strip()
        parts = t.split()
        return parts[0] if len(parts) > 1 and not parts[0].isdigit() else 'NUM'
    df['ticket_prefix'] = df['ticket'].apply(ticket_prefix)

    # ── Age 구간 (결측 전, 보간 후 다시 계산) ──
    df['is_child']  = (df['age'] < 12).astype(int)   # 일단 NaN→ False, 보간 후 재계산
    df['is_senior'] = (df['age'] > 60).astype(int)

    # ── Surname (가족 그룹용) ──
    df['surname'] = df['name'].apply(lambda x: x.split(',')[0].strip())

    return df

# ─────────────────────────────────────────
# [3] Fold-safe 파생 피처 계산
# ─────────────────────────────────────────
def fold_features(df_tr, df_va, df_te, fare_median):
    """훈련 fold 통계만 사용해 val/test에 적용 → 누수 방지"""

    # ── Ticket group size ──
    tgs = df_tr['ticket'].value_counts().to_dict()
    for df in [df_tr, df_va, df_te]:
        df['ticket_group_size'] = df['ticket'].map(tgs).fillna(1)
        df['fare'] = df['fare'].fillna(fare_median)
        df['fare_per_person'] = df['fare'] / df['ticket_group_size']
        df['fare_log']        = np.log1p(df['fare'])
        # Fare 이상치 clip (99th percentile of train)
        fare_cap = df_tr['fare'].quantile(0.99)
        df['fare_log'] = np.log1p(df['fare'].clip(upper=fare_cap))

    # ── Age 보간 (title + pclass + gender 기준) ──
    age_lut = df_tr.groupby(['title_code','pclass','gender_code'])['age'].median().to_dict()
    age_lut2 = df_tr.groupby(['title_code','pclass'])['age'].median().to_dict()  # fallback
    def fill_age(r):
        if not pd.isna(r['age']): return r['age']
        return age_lut.get((r['title_code'], r['pclass'], r['gender_code']),
               age_lut2.get((r['title_code'], r['pclass']), 28))
    for df in [df_tr, df_va, df_te]:
        df['age'] = df.apply(fill_age, axis=1)
        df['age_log']   = np.log1p(df['age'])
        df['is_child']  = (df['age'] < 12).astype(int)
        df['is_senior'] = (df['age'] > 60).astype(int)
        df['age_bin']   = pd.cut(df['age'],
                                  bins=[0,12,18,35,60,100],
                                  labels=[0,1,2,3,4]).astype(int)

    # ── Age × Pclass 교호작용 ──
    for df in [df_tr, df_va, df_te]:
        df['age_x_pclass']    = df['age'] * df['pclass']
        df['fare_x_pclass']   = df['fare_log'] * df['pclass']
        df['gender_x_pclass'] = df['gender_code'] * df['pclass']

    # ── Ticket prefix 인코딩 (train fold 기준 빈도) ──
    tp_freq = df_tr['ticket_prefix'].value_counts(normalize=True).to_dict()
    for df in [df_tr, df_va, df_te]:
        df['ticket_prefix_freq'] = df['ticket_prefix'].map(tp_freq).fillna(0)

    return df_tr, df_va, df_te

# ─────────────────────────────────────────
# [4] 피처 목록
# ─────────────────────────────────────────
FEATURES = [
    'pclass', 'age_log', 'title_code', 'embarked_code',
    'fare_log', 'fare_per_person',
    'gender_code', 'family_size', 'is_alone', 'family_type',
    'cabin_known', 'cabin_deck_code',
    'is_child', 'is_senior', 'age_bin',
    'age_x_pclass', 'fare_x_pclass', 'gender_x_pclass',
    'ticket_prefix_freq', 'ticket_group_size'
]

# 필요한 라이브러리 추가 임포트
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ─────────────────────────────────────────
# [5] Base 모델 정의 (Heavier Stacking)
# ─────────────────────────────────────────
BASE_MODELS = {
    'rf': RandomForestClassifier(
        n_estimators=800, max_depth=6, min_samples_leaf=5, 
        max_features='sqrt', random_state=42, n_jobs=-1),
    
    'et': ExtraTreesClassifier(
        n_estimators=800, max_depth=7, min_samples_leaf=5, 
        max_features='sqrt', random_state=42, n_jobs=-1),
    
    'gbm': GradientBoostingClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4, 
        min_samples_leaf=5, subsample=0.8, random_state=42),
    
    # 🔥 [NEW] XGBoost 추가: 과적합 방지를 위해 depth를 낮추고 규제 적용
    'xgb': XGBClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, gamma=1, 
        eval_metric='logloss', random_state=42, use_label_encoder=False),
    
    # 🔥 [NEW] LightGBM 추가: 속도가 매우 빠르고 성능이 강력함
    'lgbm': LGBMClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=4, num_leaves=15,
        subsample=0.8, colsample_bytree=0.8, 
        random_state=42, verbose=-1),
    
    'svm': SVC(probability=True, C=1.0, kernel='rbf', random_state=42),
}
# ─────────────────────────────────────────
# [6] Stacking 훈련 루프
# ─────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fare_median = train['fare'].median()

n_train = len(train)
n_test  = len(test)
n_models = len(BASE_MODELS)

oof_preds  = np.zeros((n_train, n_models))
test_preds = np.zeros((n_test,  n_models))

print("=" * 55)
print("  Stacking Ensemble V7  (RF / ET / GBM / SVM)")
print("=" * 55)

scaler = StandardScaler()   # SVM용

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, train['survived'])):
    df_tr = base_preprocess(train.iloc[tr_idx])
    df_va = base_preprocess(train.iloc[va_idx])
    df_te = base_preprocess(test)

    df_tr, df_va, df_te = fold_features(df_tr, df_va, df_te, fare_median)

    X_tr = df_tr[FEATURES].values
    y_tr = df_tr['survived'].values
    X_va = df_va[FEATURES].values
    y_va = df_va['survived'].values
    X_te = df_te[FEATURES].values

    # SVM용 정규화 (train fold 기준 fit)
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    fold_accs = []
    for i, (name, model) in enumerate(BASE_MODELS.items()):
        X_fit = X_tr_s if name == 'svm' else X_tr
        X_val = X_va_s if name == 'svm' else X_va
        X_tst = X_te_s if name == 'svm' else X_te

        model.fit(X_fit, y_tr)

        oof_p = model.predict_proba(X_val)[:, 1]
        oof_preds[va_idx, i] = oof_p
        test_preds[:, i]    += model.predict_proba(X_tst)[:, 1] / 5

        acc = accuracy_score(y_va, (oof_p > 0.5).astype(int))
        fold_accs.append(acc)

    avg_acc = np.mean(fold_accs)
    print(f"Fold {fold+1} | RF:{fold_accs[0]:.4f}  ET:{fold_accs[1]:.4f}"
          f"  GBM:{fold_accs[2]:.4f}  SVM:{fold_accs[3]:.4f} | avg={avg_acc:.4f}")

# ─────────────────────────────────────────
# [7] Meta 모델 (Logistic Regression)
# ─────────────────────────────────────────
print("\n--- Meta Model Training ---")
meta = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
meta.fit(oof_preds, train['survived'])

oof_meta = meta.predict_proba(oof_preds)[:, 1]
cv_acc   = accuracy_score(train['survived'], (oof_meta > 0.5).astype(int))
print(f"Meta Model OOF CV Accuracy: {cv_acc:.4f}")

# ─────────────────────────────────────────
# [8] 임계값 최적화 (OOF 기준)
# ─────────────────────────────────────────
best_thr, best_acc = 0.5, 0.0
for thr in np.arange(0.40, 0.61, 0.01):
    acc = accuracy_score(train['survived'], (oof_meta > thr).astype(int))
    if acc > best_acc:
        best_acc, best_thr = acc, thr
print(f"Best threshold: {best_thr:.2f}  (OOF acc={best_acc:.4f})")

# ─────────────────────────────────────────
# [9] 제출
# ─────────────────────────────────────────
final_preds = meta.predict_proba(test_preds)[:, 1]
submission['survived'] = (final_preds > best_thr).astype(int)
submission.to_csv("submission_absolute_v5.csv", index=False)