# 전략
1. Deck를 살리자. 
 - 객실 번호의 홀짝(좌현 우현) 배치가 생존에 크게 의미를 둔다.
2. 티켓번호의 연속성에 '물리적 인접성'을 찾자
3. 여성과 아이 그룹 생존 로직의 정교화
 - 후처리 로직을 더 발전 시키기 위해 티켓번호가 인접한 승객들이 동일한 창구에서 예약하여 물리적으로 가까운 위치에 머물렀을 가능성을 추가


 --- [Titanic Master Pipeline] CV & Ensemble Start ---
Fold 1 Accuracy: 0.8533
Fold 2 Accuracy: 0.9016
Fold 3 Accuracy: 0.8361
Fold 4 Accuracy: 0.8525
Fold 5 Accuracy: 0.9071

✅ Total Cross-Validation Accuracy: 0.8701

--- [Titanic Master Pipeline] CV & Ensemble Start ---
Fold 1 Accuracy: 0.8533
Fold 2 Accuracy: 0.9016
Fold 3 Accuracy: 0.8361
Fold 4 Accuracy: 0.8579
Fold 5 Accuracy: 0.9071

✅ Total Cross-Validation Accuracy: 0.8712
🚀 [Master Strategy] Submission file is ready. LB 0.90+ Challenge!

--- [Titanic Master Pipeline] CV with is_alone Start ---
Fold 1 Accuracy: 0.8533
Fold 2 Accuracy: 0.8962
Fold 3 Accuracy: 0.8415
Fold 4 Accuracy: 0.8579
Fold 5 Accuracy: 0.9071

--- [Titanic Master Pipeline] CV & Ensemble Start ---
Fold 1 Accuracy: 0.8533
Fold 2 Accuracy: 0.9016
Fold 3 Accuracy: 0.8361
Fold 4 Accuracy: 0.8579
Fold 5 Accuracy: 0.9071

✅ Total Cross-Validation Accuracy: 0.8712

--- [Blind Spot Fix] CV & Ensemble Start ---
Fold 1 Accuracy: 0.8587
Fold 2 Accuracy: 0.8962
Fold 3 Accuracy: 0.8361
Fold 4 Accuracy: 0.8579
Fold 5 Accuracy: 0.9016

✅ Total Cross-Validation Accuracy: 0.8701
🚀 [Master Strategy] Submission file generated.