import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# [1] 데이터 로드
# ─────────────────────────────────────────
train = pd.read_csv("/kaggle/input/competitions/skn-27-ml/train.csv")
test  = pd.read_csv("/kaggle/input/competitions/skn-27-ml/test.csv")
submission = pd.read_csv("/kaggle/input/competitions/skn-27-ml/submission.csv")

# ─────────────────────────────────────────
# [2] 기초 전처리
# ─────────────────────────────────────────
def base_preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.lower()
    df['title'] = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['title'] = df['title'].replace(rare, 'Rare').replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['title_code'] = df['title'].map({"Master":0,"Miss":1,"Mr":2,"Mrs":3,"Rare":4})
    df['gender_code'] = df['gender'].map({"male":0,"female":1})
    df['embarked_code'] = df['embarked'].fillna('C').map({"S":0,"C":1,"Q":2})
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['family_type'] = pd.cut(df['family_size'], bins=[0,1,4,20], labels=[0,1,2]).astype(int)
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    df['cabin_known'] = df['cabin'].notna().astype(int)
    df['cabin_deck'] = df['cabin'].str[0].fillna('U')
    deck_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8,'U':0}
    df['cabin_deck_code'] = df['cabin_deck'].map(deck_map).fillna(0).astype(int)
    df['surname'] = df['name'].apply(lambda x: x.split(',')[0].strip())
    return df

# ─────────────────────────────────────────
# [3] Fold-safe 파생 피처 (그룹 생존 힌트 추가)
# ─────────────────────────────────────────
def fold_features(df_tr, df_va, df_te, fare_median):
    # Ticket Group Size
    tgs = df_tr['ticket'].value_counts().to_dict()
    for df in [df_tr, df_va, df_te]:
        df['ticket_group_size'] = df['ticket'].map(tgs).fillna(1)
        df['fare'] = df['fare'].fillna(fare_median)
        df['fare_per_person'] = df['fare'] / df['ticket_group_size']
        fare_cap = df_tr['fare'].quantile(0.99)
        df['fare_log'] = np.log1p(df['fare'].clip(upper=fare_cap))

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
# [4] 피처 목록 (불필요 피처 제거 및 콤마 수정)
# ─────────────────────────────────────────
FEATURES = [
    'pclass', 'age_log', 'title_code', 'embarked_code',
    'fare_log', 'fare_per_person', 'gender_code', 
    'family_size', 'family_type', 'is_alone',
    'cabin_known', 'cabin_deck_code',
    'age_x_pclass', 'fare_x_pclass', 'gender_x_pclass',
    'ticket_group_size'
]

# ─────────────────────────────────────────
# [5] Base 모델 정의 (XGB, LGBM 포함)
# ─────────────────────────────────────────
BASE_MODELS = {
    'rf': RandomForestClassifier(n_estimators=800, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1),
    'et': ExtraTreesClassifier(n_estimators=800, max_depth=7, min_samples_leaf=5, random_state=42, n_jobs=-1),
    'gbm': GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42),
    'xgb': XGBClassifier(n_estimators=500, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.8, gamma=1, eval_metric='logloss', random_state=42),
    'lgbm': LGBMClassifier(n_estimators=208, learning_rate=0.01087, max_depth=3, num_leaves=14, subsample=0.59, colsample_bytree=0.74, min_child_samples=16, verbose=-1),
    'svm': SVC(probability=True, C=1.0, kernel='rbf', random_state=42)
}

# ─────────────────────────────────────────
# [6] Stacking 훈련 루프
# ─────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fare_median = train['fare'].median()
n_train, n_test, n_models = len(train), len(test), len(BASE_MODELS)
oof_preds = np.zeros((n_train, n_models))
test_preds = np.zeros((n_test, n_models))
scaler = StandardScaler()

print("=" * 55)
print("  Stacking Ensemble V9 (Advanced Grouping)")
print("=" * 55)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, train['survived'])):
    df_tr = base_preprocess(train.iloc[tr_idx])
    df_va = base_preprocess(train.iloc[va_idx])
    df_te = base_preprocess(test)
    df_tr, df_va, df_te = fold_features(df_tr, df_va, df_te, fare_median)

    X_tr, y_tr = df_tr[FEATURES].values, df_tr['survived'].values
    X_va, y_va = df_va[FEATURES].values, df_va['survived'].values
    X_te = df_te[FEATURES].values

    X_tr_s, X_va_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    fold_accs = []
    for i, (name, model) in enumerate(BASE_MODELS.items()):
        X_fit = X_tr_s if name == 'svm' else X_tr
        X_val = X_va_s if name == 'svm' else X_va
        X_tst = X_te_s if name == 'svm' else X_te
        
        model.fit(X_fit, y_tr)
        oof_p = model.predict_proba(X_val)[:, 1]
        oof_preds[va_idx, i] = oof_p
        test_preds[:, i] += model.predict_proba(X_tst)[:, 1] / 5
        fold_accs.append(accuracy_score(y_va, (oof_p > 0.5).astype(int)))

    print(f"Fold {fold+1} | Avg Acc: {np.mean(fold_accs):.4f}")

# ─────────────────────────────────────────
# [7-9] Meta 모델 및 제출 (임계값 0.58 고정)
# ─────────────────────────────────────────
meta = LogisticRegression(C=1.0, random_state=42)
meta.fit(oof_preds, train['survived'])
final_preds = meta.predict_proba(test_preds)[:, 1]
submission['survived'] = (final_preds > 0.58).astype(int)
submission.to_csv("submission_v9_final.csv", index=False)
print("\n✅ V9 Submission Saved! (Threshold 0.58 Applied)")

# ─────────────────────────────────────────
# [10] 시각화
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sns.barplot(x=meta.coef_[0], y=list(BASE_MODELS.keys()), ax=axes[0], palette='viridis')
axes[0].set_title('Meta Model Weights')

tree_importances = np.mean([BASE_MODELS[m].feature_importances_ for m in ['rf','et','gbm','xgb','lgbm']], axis=0)
sns.barplot(x=tree_importances, y=FEATURES, ax=axes[1], palette='mako')
axes[1].set_title('Average Feature Importance')
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# [11] Seed Averaging (10-Seeds)
# ─────────────────────────────────────────
seeds = [42, 100, 2026, 777, 123, 9, 88, 55, 1, 999]
final_probs_list = []

print(f"🚀 {len(seeds)}개 시드 에버리징 시작... (이 작업은 시간이 소요됩니다)")

for run_seed in seeds:
    print(f"\n--- Running with Seed: {run_seed} ---")
    
    # 1. K-Fold 정의 (시드 변경)
    skf_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=run_seed)
    
    # 2. 모델들의 시드값도 동기화
    for name, model in BASE_MODELS.items():
        if hasattr(model, 'random_state'):
            model.random_state = run_seed
            
    oof_preds_seed = np.zeros((n_train, n_models))
    test_preds_seed = np.zeros((n_test, n_models))
    
    # 3. 스태킹 루프 (동일한 로직)
    for fold, (tr_idx, va_idx) in enumerate(skf_seed.split(train, train['survived'])):
        df_tr = base_preprocess(train.iloc[tr_idx])
        df_va = base_preprocess(train.iloc[va_idx])
        df_te = base_preprocess(test)
        df_tr, df_va, df_te = fold_features(df_tr, df_va, df_te, fare_median)

        X_tr, y_tr = df_tr[FEATURES].values, df_tr['survived'].values
        X_va, y_va = df_va[FEATURES].values, df_va['survived'].values
        X_te = df_te[FEATURES].values

        X_tr_s, X_va_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

        for i, (name, model) in enumerate(BASE_MODELS.items()):
            X_fit = X_tr_s if name == 'svm' else X_tr
            X_val = X_va_s if name == 'svm' else X_va
            X_tst = X_te_s if name == 'svm' else X_te
            
            model.fit(X_fit, y_tr)
            oof_preds_seed[va_idx, i] = model.predict_proba(X_val)[:, 1]
            test_preds_seed[:, i] += model.predict_proba(X_tst)[:, 1] / 5
            
    # 4. Meta 모델 학습 및 예측 확률 저장
    meta_seed = LogisticRegression(C=1.0, random_state=run_seed)
    meta_seed.fit(oof_preds_seed, train['survived'])
    
    # 현재 시드에서의 최종 확률값
    current_run_probs = meta_seed.predict_proba(test_preds_seed)[:, 1]
    final_probs_list.append(current_run_probs)
    
    print(f"✅ Seed {run_seed} 완료.")

# ─────────────────────────────────────────
# [12] 결과 평균 및 최종 제출
# ─────────────────────────────────────────
# 모든 시드에서 나온 확률값의 평균 계산
averaged_probs = np.mean(final_probs_list, axis=0)

# 우리가 찾은 마법의 숫자 0.58 적용
submission['survived'] = (averaged_probs > 0.58).astype(int)
submission.to_csv("submission_v11_seed_avg.csv", index=False)

print("\n🏆 V11 Seed Averaging Final Submission Saved!")
print(f"최종 평균 생존율: {submission['survived'].mean():.3f}")