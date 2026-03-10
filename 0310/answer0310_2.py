import pandas as pd
import numpy as np
import warnings
import optuna

# 머신러닝 라이브러리
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 다양한 성격의 모델들
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# [!] 경고 무시 및 시드 고정
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING) # 로그 축소
SEED = 42

# --------------------------------------------------------------------------------
# [1] 데이터 로드 및 전처리 (안정적인 피처 12개 유지)
# --------------------------------------------------------------------------------
train = pd.read_csv("/kaggle/input/competitions/skn-27-ml/train.csv")
test = pd.read_csv("/kaggle/input/competitions/skn-27-ml/test.csv")
submission = pd.read_csv("/kaggle/input/competitions/skn-27-ml/submission.csv")

train.columns, test.columns = train.columns.str.lower(), test.columns.str.lower()

def create_stable_features(df_input):
    df = df_input.copy()
    df['title'] = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 1, "Countess": 2, "Ms": 1, "Lady": 2, "Jonkheer": 4, "Don": 4, "Dona" : 4, "Mme": 2,"Capt": 4,"Sir": 4 }
    df['title_code'] = df['title'].map(title_map).fillna(4)
    df['gender_code'] = df['gender'].map({"male": 0, "female": 1})
    df['embarked_code'] = df['embarked'].fillna('S').map({"S": 0, "C": 1, "Q": 2})
    df['fare'] = df['fare'].fillna(train['fare'].median())
    df['fare_log'] = np.log1p(df['fare'])
    df['pclass_Sex'] = df['pclass'].astype(str) + '_' + df['gender_code'].astype(str)
    df['pclass_Sex'] = df['pclass_Sex'].map({'1_0': 0, '1_1': 1, '2_0': 2, '2_1': 3, '3_0': 4, '3_1': 5})
    df['name_Len'] = df['name'].apply(len)
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    return df

train_proc, test_proc = create_stable_features(train), create_stable_features(test)

train_proc['age'] = train_proc.groupby(['title_code', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))
test_proc['age'] = test_proc.groupby(['title_code', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))

def get_age_cat(age):
    if age <= 10: return 0
    elif age <= 16: return 1
    elif age <= 30: return 2
    elif age <= 60: return 3
    else: return 4
train_proc['age_Bin'] = train_proc['age'].apply(get_age_cat)
test_proc['age_Bin'] = test_proc['age'].apply(get_age_cat)

all_tickets = pd.concat([train_proc['ticket'], test_proc['ticket']])
ticket_counts = all_tickets.value_counts()
train_proc['ticket_Count'] = train_proc['ticket'].map(ticket_counts)
test_proc['ticket_Count'] = test_proc['ticket'].map(ticket_counts)

full_fare = pd.concat([train_proc['fare'], test_proc['fare']])
train_proc['fare_Bin'] = pd.qcut(full_fare, 13, labels=False).iloc[:len(train_proc)]
test_proc['fare_Bin'] = pd.qcut(full_fare, 13, labels=False).iloc[len(train_proc):]

stable_features = [
    'pclass', 'gender_code', 'age_Bin', 'fare_log', 
    'ticket_Count', 'pclass_Sex', 'name_Len', 'fare_Bin', 
    'title_code', 'family_size', 'embarked_code', 'is_alone'
]

X, y, X_test = train_proc[stable_features], train_proc['survived'], test_proc[stable_features]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# --------------------------------------------------------------------------------
# [2] 🔥 방어형 Optuna 세팅 (과적합 원천 차단)
# --------------------------------------------------------------------------------
# AI가 '깊이 파는 것'을 막고, '안전한 범위' 안에서만 찾게 제한합니다.

def objective_xgb(trial):
    params = {
        'n_estimators': 700,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'max_depth': trial.suggest_int('max_depth', 3, 4), # 🔥 무조건 얕게 
        'min_child_weight': trial.suggest_int('min_child_weight', 4, 10), # 🔥 깐깐한 조건 강제
        'gamma': trial.suggest_float('gamma', 0.1, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'random_state': SEED, 'n_jobs': -1, 'enable_categorical': False
    }
    model = XGBClassifier(**params)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        aucs.append(roc_auc_score(y.iloc[va_idx], model.predict_proba(X.iloc[va_idx])[:, 1]))
    return np.mean(aucs)

def objective_lgbm(trial):
    params = {
        'n_estimators': 700,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'max_depth': trial.suggest_int('max_depth', 3, 4), # 🔥 무조건 얕게
        'num_leaves': trial.suggest_int('num_leaves', 7, 15), # 🔥 잎 개수 제한
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 40), # 🔥 노이즈 차단
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 5.0), # 🔥 강력한 정규화 강제
        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'random_state': SEED, 'n_jobs': -1, 'verbose': -1
    }
    model = LGBMClassifier(**params)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        aucs.append(roc_auc_score(y.iloc[va_idx], model.predict_proba(X.iloc[va_idx])[:, 1]))
    return np.mean(aucs)

def objective_cat(trial):
    params = {
        'iterations': 700,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'depth': trial.suggest_int('depth', 3, 4), # 🔥 무조건 얕게
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 5, 15), # 🔥 강력한 정규화 강제
        'random_state': SEED, 'verbose': 0
    }
    model = CatBoostClassifier(**params)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        aucs.append(roc_auc_score(y.iloc[va_idx], model.predict_proba(X.iloc[va_idx])[:, 1]))
    return np.mean(aucs)

# --------------------------------------------------------------------------------
# [3] Optuna 실행 (각 30회)
# --------------------------------------------------------------------------------
N_TRIALS = 30
print(f"--- 🤖 Safe Optuna Tuning Start (Trials: {N_TRIALS}) ---")

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS)
print(f"✅ Best XGB AUC: {study_xgb.best_value:.4f}")

study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS)
print(f"✅ Best LGBM AUC: {study_lgbm.best_value:.4f}")

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=N_TRIALS)
print(f"✅ Best CAT AUC: {study_cat.best_value:.4f}")

# --------------------------------------------------------------------------------
# [4] 6대장 모델 앙상블 (수동 고정 모델 + Optuna 튜닝 모델)
# --------------------------------------------------------------------------------
print("\n--- 🛡️ Final Anti-Overfitting Ensemble ---")

models = {
    # 1~3. 고정 모델 (안정적인 앵커 역할)
    "RF": RandomForestClassifier(n_estimators=700, max_depth=5, min_samples_split=8, min_samples_leaf=4, random_state=SEED, n_jobs=-1),
    "ET": ExtraTreesClassifier(n_estimators=700, max_depth=5, min_samples_split=8, min_samples_leaf=4, random_state=SEED, n_jobs=-1),
    "LR": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, random_state=SEED)),
    
    # 4~6. 방어형 Optuna가 튜닝한 부스팅 모델들
    "Optuna_XGB": XGBClassifier(**study_xgb.best_params, n_estimators=700, random_state=SEED, n_jobs=-1, enable_categorical=False),
    "Optuna_LGBM": LGBMClassifier(**study_lgbm.best_params, n_estimators=700, random_state=SEED, n_jobs=-1, verbose=-1),
    "Optuna_CAT": CatBoostClassifier(**study_cat.best_params, iterations=700, verbose=0, random_state=SEED)
}

ensemble_oof = np.zeros(len(train_proc))
ensemble_test = np.zeros(len(test_proc))

for name, model in models.items():
    oof_preds = np.zeros(len(train_proc))
    test_preds = np.zeros(len(test_proc))
    
    for tr_idx, va_idx in skf.split(X, y):
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_preds[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / 5
        
    auc = roc_auc_score(y, oof_preds)
    print(f"✅ {name:15s} | CV AUC: {auc:.4f}")
    
    ensemble_oof += oof_preds / len(models)
    ensemble_test += test_preds / len(models)

# 최종 결과 계산
final_auc = roc_auc_score(y, ensemble_oof)
final_acc = accuracy_score(y, (ensemble_oof > 0.5).astype(int))

print("\n🌟 [FINAL SAFE ENSEMBLE] 🌟")
print(f"Combined AUC: {final_auc:.4f} | Accuracy: {final_acc:.4f}")

submission['survived'] = (ensemble_test > 0.5).astype(int)
submission.to_csv("submission_safe_optuna.csv", index=False)
print("\n📂 'submission_safe_optuna.csv' 저장 완료! (가장 단단한 Private 방패가 될 것입니다)")