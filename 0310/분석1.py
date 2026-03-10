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

# 글로벌 시드 고정 함수
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# 시드 적용!
seed_everything(42)

print("🌱 Global Seed Fixed to 42")

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# [1] 데이터 로드
# ─────────────────────────────────────────
train_raw = pd.read_csv("/kaggle/input/competitions/skn-27-ml/train.csv")
test_raw  = pd.read_csv("/kaggle/input/competitions/skn-27-ml/test.csv")
submission = pd.read_csv("/kaggle/input/competitions/skn-27-ml/submission.csv")

# ─────────────────────────────────────────
# [2] 기초 전처리
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

    # Family & Name Length
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone']    = (df['family_size'] == 1).astype(int)
    df['name_len']    = df['name'].apply(len)

    # Cabin & Side
    df['cabin_known'] = df['cabin'].notna().astype(int)
    df['cabin_deck']  = df['cabin'].str[0].fillna('U')
    deck_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8,'U':0}
    df['cabin_deck_code'] = df['cabin_deck'].map(deck_map).fillna(0).astype(int)
    
    df['cabin_num'] = df['cabin'].str.extract('(\d+)').astype(float)
    df['cabin_side'] = df['cabin_num'] % 2
    df['cabin_side'] = df['cabin_side'].fillna(-1)

    return df

# ─────────────────────────────────────────
# [3] 추가 파생 피처 (WCG Index 포함)
# ─────────────────────────────────────────
def advanced_features(df_tr, df_va, df_te, fare_median):
    # 1. WCG (Women & Children Group) 생존율 지표 계산
    # 누수 방지를 위해 오직 학습 데이터(df_tr)에서만 WCG 생존율을 계산합니다.
    # WCG 조건: 여성(gender_code==1) 이거나 아이(Master 타이틀, title_code==0)
    df_tr['is_wcg'] = ((df_tr['gender_code'] == 1) | (df_tr['title_code'] == 0)).astype(int)
    wcg_train = df_tr[df_tr['is_wcg'] == 1]
    
    # 티켓 그룹별 WCG 생존율 딕셔너리 생성
    ticket_wcg_survival = wcg_train.groupby('ticket')['survived'].mean().to_dict()
    global_wcg_mean = wcg_train['survived'].mean() if not wcg_train.empty else 0.5

    # 2. 기타 통계 및 매핑
    tgs = df_tr['ticket'].value_counts().to_dict()
    age_lut = df_tr.groupby(['title_code','pclass','gender_code'])['age'].median().to_dict()
    
    def fill_age(r):
        if not pd.isna(r['age']): return r['age']
        return age_lut.get((r['title_code'], r['pclass'], r['gender_code']), 28)

    # 각 데이터셋(Train, Val, Test)에 피처 일괄 적용
    for df in [df_tr, df_va, df_te]:
        # WCG Index 적용 (해당 티켓 그룹 내 여성/아이의 생존율)
        df['wcg_survival_index'] = df['ticket'].map(ticket_wcg_survival).fillna(global_wcg_mean)

        # Ticket & Fare
        df['ticket_group_size'] = df['ticket'].map(tgs).fillna(1)
        df['fare'] = df['fare'].replace(0, fare_median).fillna(fare_median)
        df['fare_per_person'] = df['fare'] / df['ticket_group_size']
        fare_cap = df_tr['fare'].quantile(0.99)
        df['fare_log'] = np.log1p(df['fare'].clip(upper=fare_cap))
        df['fare_rank'] = df.groupby('pclass')['fare'].rank(pct=True)

        # Age
        df['age'] = df.apply(fill_age, axis=1)
        df['age_log'] = np.log1p(df['age'])
        
        # 교차 피처
        df['age_x_pclass'] = df['age'] * df['pclass']
        df['fare_x_pclass'] = df['fare_log'] * df['pclass']
        df['gender_x_pclass'] = df['gender_code'] * df['pclass']

    return df_tr, df_va, df_te

# ─────────────────────────────────────────
# [4] 데이터 분할 (train_test_split 적용)
# ─────────────────────────────────────────
print("=" * 60)
print("  Stacking Ensemble (Holdout Blending + WCG + Family Penalty)")
print("=" * 60)

# 요청하신 train_test_split 코드 적용 (stratify=target)
train_split, val_split = train_test_split(
    train_raw,
    test_size=0.2,
    random_state=42,
    stratify=train_raw['survived']
)
# 기초 전처리
df_tr = base_preprocess(train_split)
df_va = base_preprocess(val_split)
df_te = base_preprocess(test_raw)

# 고급 전처리 (WCG 생존 지표 등)
fare_median = train_split['fare'].median()
df_tr, df_va, df_te = advanced_features(df_tr, df_va, df_te, fare_median)

# 모델 학습에 사용할 피처 리스트 (wcg_survival_index 추가됨)
FEATURES = [
    'pclass', 'age_log', 'title_code', 'embarked_code', 'fare_log', 'fare_per_person', 'fare_rank',
    'gender_code', 'family_size', 'is_alone', 'name_len', 'cabin_known', 'cabin_deck_code', 'cabin_side',
    'age_x_pclass', 'fare_x_pclass', 'gender_x_pclass', 'ticket_group_size', 'wcg_survival_index'
]

X_tr, y_tr = df_tr[FEATURES].values, df_tr['survived'].values
X_va, y_va = df_va[FEATURES].values, df_va['survived'].values
X_te = df_te[FEATURES].values

# 스케일링 (LogisticRegression을 위함)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)

# ─────────────────────────────────────────
# [5] Base 모델 개별 학습 및 예측 (요청한 4개 모델)
# ─────────────────────────────────────────
BASE_MODELS = {
    'rf': RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42, n_jobs=-1),
    'xgb': XGBClassifier(n_estimators=500, learning_rate=0.02, max_depth=4, random_state=42, eval_metric='logloss'),
    'lgbm': LGBMClassifier(n_estimators=500, learning_rate=0.02, max_depth=4, random_state=42, verbose=-1),
    'lr': LogisticRegression(random_state=42, max_iter=1000)
}

# 메타 모델 학습용 검증/테스트 예측값 저장소
oof_preds = np.zeros((len(X_va), len(BASE_MODELS)))
test_preds = np.zeros((len(X_te), len(BASE_MODELS)))

print("🔹 Training Base Models...")
for i, (name, model) in enumerate(BASE_MODELS.items()):
    # LR 모델은 스케일링된 데이터를, 트리 모델은 원본 데이터를 사용
    X_fit = X_tr_s if name == 'lr' else X_tr
    X_val = X_va_s if name == 'lr' else X_va
    X_tst = X_te_s if name == 'lr' else X_te
    
    model.fit(X_fit, y_tr)
    oof_preds[:, i] = model.predict_proba(X_val)[:, 1]
    test_preds[:, i] = model.predict_proba(X_tst)[:, 1]
    print(f"  - {name.upper()} 완료")

# ─────────────────────────────────────────
# [6] Meta-Learner 학습 (Stacking)
# ─────────────────────────────────────────
print("\n🔹 Training Meta-Learner...")
meta_model = LogisticRegression(C=1.0, random_state=42)
meta_model.fit(oof_preds, y_va)

# 검증 세트 기준 최적의 임계값(Threshold) 탐색
meta_val_probs = meta_model.predict_proba(oof_preds)[:, 1]
best_thr, best_acc = 0.5, 0
for thr in np.arange(0.4, 0.61, 0.01):
    acc = accuracy_score(y_va, (meta_val_probs > thr).astype(int))
    if acc > best_acc:
        best_acc, best_thr = acc, thr

print(f"🏆 Meta Model Validation Acc: {best_acc:.4f} (Threshold: {best_thr:.2f})")

# Test 데이터 최종 예측
final_test_probs = meta_model.predict_proba(test_preds)[:, 1]
final_predictions = (final_test_probs > best_thr).astype(int)

# ─────────────────────────────────────────
# [7] 대가족 페널티 & Master 타이틀 생존율 조정 (Post-processing)
# ─────────────────────────────────────────
# 1. 변환 전 원본 예측값 보존
original_predictions = final_predictions.copy()

# -----------------------------------------
# 룰 A: 대가족 페널티 발동 (기존)
# -----------------------------------------
penalty_mask = (df_te['pclass'] == 3) & (df_te['family_size'] >= 5)
final_predictions[penalty_mask] = 0

# -----------------------------------------
# 룰 B: Master (남자 아이) 생존율 강제 조정 (신규)
# -----------------------------------------
# title_code == 0 은 'Master'
master_mask = (df_te['title_code'] == 0)

# B-1: 동반 사망 룰 (같은 그룹의 여성/아이가 전원 사망한 경우 -> 사망)
master_die_mask = master_mask & (df_te['wcg_survival_index'] == 0)
final_predictions[master_die_mask] = 0

# B-2: 구조 룰 (그룹 내 누군가 살았거나, 혼자라 정보가 없는 경우 -> 생존)
# 단, 대가족 페널티에 이미 걸려 사망 확정인 경우는 제외 (~penalty_mask)
master_survive_mask = master_mask & (df_te['wcg_survival_index'] > 0) & ~penalty_mask
final_predictions[master_survive_mask] = 1

# ==========================================
# 🔍 변환 결과 검증 (Sanity Check)
# ==========================================
# 각각 어떤 룰에 의해 바뀌었는지 마스크 추출
changed_family = (original_predictions == 1) & (final_predictions == 0) & penalty_mask
changed_master_to_1 = (original_predictions == 0) & (final_predictions == 1) & master_survive_mask
changed_master_to_0 = (original_predictions == 1) & (final_predictions == 0) & master_die_mask

print("=" * 60)
print("🚨 [Post-Processing 결과 요약]")
print(f"✔️ 대가족 페널티 (1->0 강제사망): {changed_family.sum()}명")
print(f"✔️ Master 구조 룰 (0->1 강제생존): {changed_master_to_1.sum()}명")
print(f"✔️ Master 동반 사망 (1->0 강제사망): {changed_master_to_0.sum()}명")
print("=" * 60)

changed_mask = changed_family | changed_master_to_1 | changed_master_to_0

if changed_mask.sum() > 0:
    print("\n🔍 [강제 변환된 승객 명단 전체 확인]")
    check_df = test_raw[changed_mask].copy()
    check_df['Family_Size'] = df_te.loc[changed_mask, 'family_size']
    check_df['WCG_Index'] = df_te.loc[changed_mask, 'wcg_survival_index'].round(2)
    check_df['Original_Pred'] = original_predictions[changed_mask]
    check_df['New_Pred'] = final_predictions[changed_mask]
    
    # 원인(Reason) 텍스트 매핑
    check_df['Reason'] = ''
    check_df.loc[changed_family[changed_mask], 'Reason'] = 'Large Family'
    check_df.loc[changed_master_to_1[changed_mask], 'Reason'] = 'Master Survive'
    check_df.loc[changed_master_to_0[changed_mask], 'Reason'] = 'Master Die (WCG=0)'
    
    # 확인하기 쉽게 주요 컬럼만 정렬해서 출력
    display_cols = ['passengerid', 'name', 'pclass', 'gender', 'age', 'Family_Size', 'WCG_Index', 'Reason', 'Original_Pred', 'New_Pred']
    
    # Jupyter 환경 고려
    try:
        display(check_df[display_cols].sort_values(by='Reason'))
    except NameError:
        print(check_df[display_cols].sort_values(by='Reason').to_string(index=False))
else:
    print("✅ 예측이 강제로 변경된 승객이 없습니다. (모델이 이미 완벽하게 예측함)")

# ─────────────────────────────────────────
# [8] 최종 제출 파일 저장
# ─────────────────────────────────────────
submission['survived'] = final_predictions
submission.to_csv("submission_stacked_final_master.csv", index=False)
print("\n✅ 제출 파일 생성 완료: submission_stacked_final_master.csv")

# ─────────────────────────────────────────
# [9] XAI (설명 가능한 AI): SHAP 분석
# ─────────────────────────────────────────
import shap
import matplotlib.pyplot as plt

print("\n🔍 SHAP 분석을 시작합니다. (LightGBM 모델 기준)")

# 1. SHAP 자바스크립트 시각화 초기화 (Jupyter 환경용)
shap.initjs()

# 2. X_va(검증 데이터)를 데이터프레임으로 변환 (그래프에 피처 이름을 띄우기 위함)
X_va_df = pd.DataFrame(X_va, columns=FEATURES)

# 3. Base 모델 중 대표로 LightGBM 모델 선택
lgbm_model = BASE_MODELS['lgbm']

# 4. TreeExplainer 생성 및 SHAP 값 계산
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_va_df)

# SHAP 버전에 따라 이진 분류 시 반환 형태가 다를 수 있음 (생존(1) 클래스 기준 추출)
if isinstance(shap_values, list):
    shap_values_survived = shap_values[1]
else:
    shap_values_survived = shap_values

# 5. SHAP Summary Plot 시각화
plt.figure(figsize=(10, 8))
plt.title("SHAP Feature Importance (Survival Impact)", fontsize=16, pad=20)
shap.summary_plot(shap_values_survived, X_va_df, plot_type="dot")
plt.show()

# 6. SHAP Bar Plot (단순 중요도 순위 시각화)
plt.figure(figsize=(10, 8))
plt.title("Overall Feature Importance Ranking", fontsize=16, pad=20)
shap.summary_plot(shap_values_survived, X_va_df, plot_type="bar")
plt.show()