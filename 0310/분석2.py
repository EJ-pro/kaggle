import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
import os
import random

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)
warnings.filterwarnings('ignore')
print("🌱 Global Seed Fixed to 42")

# ─────────────────────────────────────────
# [1] 데이터 로드
# ─────────────────────────────────────────
train_raw = pd.read_csv("/kaggle/input/competitions/skn-27-ml/train.csv")
test_raw  = pd.read_csv("/kaggle/input/competitions/skn-27-ml/test.csv")
submission = pd.read_csv("/kaggle/input/competitions/skn-27-ml/submission.csv")

# ─────────────────────────────────────────
# [2] 기초 전처리 함수
# ─────────────────────────────────────────
def base_preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    df['title'] = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['title'] = df['title'].replace(rare, 'Rare').replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['title_code'] = df['title'].map({"Master":0,"Miss":1,"Mr":2,"Mrs":3,"Rare":4})
    
    df['gender_code']   = df['gender'].map({"male":0,"female":1})
    df['embarked_code'] = df['embarked'].fillna('S').map({"S":0,"C":1,"Q":2})
    
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone']    = (df['family_size'] == 1).astype(int)
    df['name_len']    = df['name'].apply(len)
    
    df['cabin_known'] = df['cabin'].notna().astype(int)
    df['cabin_deck']  = df['cabin'].str[0].fillna('U')
    deck_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8,'U':0}
    df['cabin_deck_code'] = df['cabin_deck'].map(deck_map).fillna(0).astype(int)
    
    df['cabin_num'] = df['cabin'].str.extract('(\d+)').astype(float)
    df['cabin_side'] = df['cabin_num'] % 2
    df['cabin_side'] = df['cabin_side'].fillna(-1)
    return df

# ─────────────────────────────────────────
# [3] 고급 전처리 함수 (WCG 포함)
# ─────────────────────────────────────────
def advanced_features(df_tr, df_va, df_te, fare_median):
    df_tr['is_wcg'] = ((df_tr['gender_code'] == 1) | (df_tr['title_code'] == 0)).astype(int)
    wcg_train = df_tr[df_tr['is_wcg'] == 1]
    
    ticket_wcg_survival = wcg_train.groupby('ticket')['survived'].mean().to_dict()
    global_wcg_mean = wcg_train['survived'].mean() if not wcg_train.empty else 0.5

    tgs = df_tr['ticket'].value_counts().to_dict()
    age_lut = df_tr.groupby(['title_code','pclass','gender_code'])['age'].median().to_dict()
    
    def fill_age(r):
        if not pd.isna(r['age']): return r['age']
        return age_lut.get((r['title_code'], r['pclass'], r['gender_code']), 28)

    for df in [df_tr, df_va, df_te]:
        df['wcg_survival_index'] = df['ticket'].map(ticket_wcg_survival).fillna(global_wcg_mean)
        df['ticket_group_size'] = df['ticket'].map(tgs).fillna(1)
        df['fare'] = df['fare'].replace(0, fare_median).fillna(fare_median)
        df['fare_per_person'] = df['fare'] / df['ticket_group_size']
        fare_cap = df_tr['fare'].quantile(0.99)
        df['fare_log'] = np.log1p(df['fare'].clip(upper=fare_cap))
        df['fare_rank'] = df.groupby('pclass')['fare'].rank(pct=True)
        df['age'] = df.apply(fill_age, axis=1)
        df['age_log'] = np.log1p(df['age'])
        df['age_x_pclass'] = df['age'] * df['pclass']
        df['fare_x_pclass'] = df['fare_log'] * df['pclass']
        df['gender_x_pclass'] = df['gender_code'] * df['pclass']

    return df_tr, df_va, df_te

# 요청하신 train_test_split 코드 적용 (stratify=target)
train_split, val_split = train_test_split(
    train_raw,
    test_size=0.2,
    random_state=42,
    stratify=train_raw['survived']
)

FEATURES = [
    'pclass', 'age_log', 'title_code', 'embarked_code', 'fare_log', 'fare_per_person', 'fare_rank',
    'gender_code', 'family_size', 'is_alone', 'name_len', 'cabin_known', 'cabin_deck_code', 'cabin_side',
    'age_x_pclass', 'fare_x_pclass', 'gender_x_pclass', 'ticket_group_size', 'wcg_survival_index'
]

# 모델 학습 헬퍼 함수
def train_stacking(df_train, df_test, prefix="Stage 1"):
    print(f"\n{'='*50}\n ⚙️ {prefix} 모델 학습 진행 중...\n{'='*50}")
    train_split, val_split = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['survived'])
    
    df_tr = base_preprocess(train_split)
    df_va = base_preprocess(val_split)
    df_te = base_preprocess(df_test)
    
    fare_median = train_split['fare'].median()
    df_tr, df_va, df_te = advanced_features(df_tr, df_va, df_te, fare_median)
    
    X_tr, y_tr = df_tr[FEATURES].values, df_tr['survived'].values
    X_va, y_va = df_va[FEATURES].values, df_va['survived'].values
    X_te = df_te[FEATURES].values
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)
    
    base_models = {
        'rf': RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42, n_jobs=-1),
        'xgb': XGBClassifier(n_estimators=500, learning_rate=0.02, max_depth=4, random_state=42, eval_metric='logloss'),
        'lgbm': LGBMClassifier(n_estimators=500, learning_rate=0.02, max_depth=4, random_state=42, verbose=-1),
        'lr': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    oof_preds = np.zeros((len(X_va), len(base_models)))
    test_preds = np.zeros((len(X_te), len(base_models)))
    
    for i, (name, model) in enumerate(base_models.items()):
        X_fit = X_tr_s if name == 'lr' else X_tr
        X_val = X_va_s if name == 'lr' else X_va
        X_tst = X_te_s if name == 'lr' else X_te
        
        model.fit(X_fit, y_tr)
        oof_preds[:, i] = model.predict_proba(X_val)[:, 1]
        test_preds[:, i] = model.predict_proba(X_tst)[:, 1]
    
    meta_model = LogisticRegression(C=1.0, random_state=42)
    meta_model.fit(oof_preds, y_va)
    
    meta_val_probs = meta_model.predict_proba(oof_preds)[:, 1]
    best_thr, best_acc = 0.5, 0
    for thr in np.arange(0.4, 0.61, 0.01):
        acc = accuracy_score(y_va, (meta_val_probs > thr).astype(int))
        if acc > best_acc: best_acc, best_thr = acc, thr
            
    print(f"🏆 Meta Model Validation Acc: {best_acc:.4f} (Threshold: {best_thr:.2f})")
    test_probs = meta_model.predict_proba(test_preds)[:, 1]
    return test_probs, df_te, best_thr

# ─────────────────────────────────────────
# [4] Stage 1: 초기 학습 및 확률 예측
# ─────────────────────────────────────────
stage1_probs, _, stage1_thr = train_stacking(train_raw, test_raw, prefix="Stage 1 (Initial)")

# ─────────────────────────────────────────
# [5] Stage 2: Pseudo Labeling (가짜 정답 추출)
# ─────────────────────────────────────────
# 매우 확실한 예측(>98% 또는 <2%)만 추출하여 정답으로 간주합니다.
CONFIDENT_UPPER = 0.98
CONFIDENT_LOWER = 0.02

confident_mask = (stage1_probs >= CONFIDENT_UPPER) | (stage1_probs <= CONFIDENT_LOWER)
pseudo_data = test_raw[confident_mask].copy()

# 확률을 기반으로 1과 0 정답 부여
pseudo_data['survived'] = (stage1_probs[confident_mask] >= CONFIDENT_UPPER).astype(int)

# 기존 Train 데이터에 덧붙이기
augmented_train = pd.concat([train_raw, pseudo_data], ignore_index=True)

print(f"\n{'='*50}")
print(" 💉 [Pseudo Labeling 적용]")
print(f" - 원래 학습 데이터 수: {len(train_raw)}명")
print(f" - 테스트 데이터에서 추출한 확신 데이터 수: {len(pseudo_data)}명")
print(f" - 합쳐진 최종 학습 데이터 수: {len(augmented_train)}명")
print(f"{'='*50}")

# ─────────────────────────────────────────
# [6] Stage 3: 증강된 데이터로 최종 재학습
# ─────────────────────────────────────────
final_test_probs, df_te_final, final_thr = train_stacking(augmented_train, test_raw, prefix="Stage 3 (Pseudo Augmented)")
final_predictions = (final_test_probs > final_thr).astype(int)

# ─────────────────────────────────────────
# [7] Stage 4: 최종 후처리 (Rule-base Post-processing)
# ─────────────────────────────────────────
print("\n🚨 [최종 Rule-base 후처리 적용 중...]")
original_predictions = final_predictions.copy()

# 룰 A: 대가족 페널티
penalty_mask = (df_te_final['pclass'] == 3) & (df_te_final['family_size'] >= 5)
final_predictions[penalty_mask] = 0

# 룰 B: Master 룰
master_mask = (df_te_final['title_code'] == 0)
master_die_mask = master_mask & (df_te_final['wcg_survival_index'] == 0)
final_predictions[master_die_mask] = 0

master_survive_mask = master_mask & (df_te_final['wcg_survival_index'] > 0) & ~penalty_mask
final_predictions[master_survive_mask] = 1

# 변화량 요약
changed_family = (original_predictions == 1) & (final_predictions == 0) & penalty_mask
changed_master_to_1 = (original_predictions == 0) & (final_predictions == 1) & master_survive_mask
changed_master_to_0 = (original_predictions == 1) & (final_predictions == 0) & master_die_mask

print(f" ✔️ 대가족 페널티 (1->0 강제사망): {changed_family.sum()}명")
print(f" ✔️ Master 구조 룰 (0->1 강제생존): {changed_master_to_1.sum()}명")
print(f" ✔️ Master 동반 사망 (1->0 강제사망): {changed_master_to_0.sum()}명")

# ─────────────────────────────────────────
# [8] 제출 파일 생성
# ─────────────────────────────────────────
submission['survived'] = final_predictions
submission.to_csv("submission_pseudo_labeled_limit_break.csv", index=False)
print("\n✅ 궁극의 제출 파일 생성 완료: submission_pseudo_labeled_limit_break.csv")