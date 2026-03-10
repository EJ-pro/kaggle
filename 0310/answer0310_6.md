기존 베이스 코드로 복귀
대가족/남성아이 강제변환 넣었음

x_tr, X_te, y_tr, y_te = train_test_split(
	digits.data,
	target,
	test_size=0.2,
	random_state=42,
	stratify=target
)
stratify=target추가함

