import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

# '분류 모델' 페이지에서 처음 골라져 있던 네 가지 입력 속성 (체질량지수는 뺌)
SELECTED = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
CHECK_COLUMNS = SELECTED  # 숫자로 바꿔서 쓸 열

# 화면에 쓸 모델 이름 (교과서의 이름을 앞에)
LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

CANNOT = "계산할 수 없음"
MAX_ROWS = 20

st.set_page_config(page_title=f"채점표 · {APP_TITLE}", page_icon=APP_ICON, layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def confusion(actual, pred):
    """뇌졸중(1)을 양성으로 두고 TP, FP, FN, TN 사람 수를 센다."""
    pairs = list(zip(actual, pred))
    tp = sum(1 for a, p in pairs if a == 1 and p == 1)
    fn = sum(1 for a, p in pairs if a == 1 and p == 0)
    fp = sum(1 for a, p in pairs if a == 0 and p == 1)
    tn = sum(1 for a, p in pairs if a == 0 and p == 0)
    return tp, fp, fn, tn


def scores(tp, fp, fn, tn):
    """정확도·재현율·정밀도·F1. 계산할 수 없으면 None."""
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None  # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) > 0 else None  # 분모: 뇌졸중이라 예측한 사람 수
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0  # 정밀도와 재현율이 모두 0이면 F1은 0으로 둠
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1


def pct(v):
    return CANNOT if v is None else f"{v * 100:.2f}%"


def confusion_table(tp, fp, fn, tn):
    """행은 실제, 열은 예측. 칸마다 약자와 사람 수, 행·열 합계를 함께 적는다."""
    total = tp + fp + fn + tn
    return pd.DataFrame(
        [
            [f"TP · {tp:,}명", f"FN · {fn:,}명", f"{tp + fn:,}명"],
            [f"FP · {fp:,}명", f"TN · {tn:,}명", f"{fp + tn:,}명"],
            [f"{tp + fp:,}명", f"{fn + tn:,}명", f"{total:,}명"],
        ],
        index=["실제: 뇌졸중", "실제: 아님", "합계"],
        columns=["예측: 뇌졸중", "예측: 아님", "합계"],
    )


def person_list(frame):
    """번호, 나이, 평균 혈당, 고혈압 여부, 심장병 여부를 나이가 많은 순서로. (전체 표, 보여 줄 표)"""
    frame = frame.sort_values(["age", "id"], ascending=[False, True])
    full = pd.DataFrame(
        {
            "번호": frame["id"],
            "나이": frame["age"],
            "평균 혈당": frame["avg_glucose_level"],
            "고혈압": frame["hypertension"].map({1: "있음", 0: "없음"}),
            "심장병": frame["heart_disease"].map({1: "있음", 0: "없음"}),
        }
    )
    return full, full.head(MAX_ROWS)


LIST_CONFIG = {
    "번호": st.column_config.NumberColumn("번호", format="%d"),
    "나이": st.column_config.NumberColumn("나이", format="%g"),
    "평균 혈당": st.column_config.NumberColumn("평균 혈당", format="%.2f"),
}

# ---------------------------------------------------------------
# 데이터 읽기, 나누기, 모델 학습 ('분류 모델' 페이지와 같은 방법)
# ---------------------------------------------------------------
st.title(f"{APP_ICON} 채점표")
st.write(
    "뇌졸중(stroke = 1)을 **양성**으로 두고, 테스트 데이터로 두 모델을 채점해 봐요. "
    "입력 속성은 '분류 모델' 페이지에서 처음 골라져 있던 **나이, 평균 혈당, 고혈압, 심장병** 넷을 썼어요."
)

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# 번호(id) 순으로 정렬
data = raw.sort_values("id").reset_index(drop=True)
for c in CHECK_COLUMNS:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data["stroke"] = data["stroke"].astype(int)

# 번호 순으로 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정
is_test = pd.Series(data.index % 10 < 3, index=data.index)
train = data[~is_test]
test = data[is_test]

X_train = train[SELECTED].copy()
X_test = test[SELECTED].copy()
y_train = train["stroke"]
y_test = test["stroke"]

# 크기 맞추기는 훈련용으로만 배운다 (로지스틱 회귀에 사용)
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)

lr = LogisticRegression().fit(Xs_train, y_train)

# 의사결정트리는 크기를 맞추지 않은 원래 값으로 배운다
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)

base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

actual = y_test.tolist()
preds = {
    LR_NAME: lr.predict(Xs_test).tolist(),
    TREE_NAME: tree.predict(X_test).tolist(),
    BASE_NAME: base.predict(X_test).tolist(),
}
counts = {name: confusion(actual, p) for name, p in preds.items()}

# ---------------------------------------------------------------
# 혼동행렬
# ---------------------------------------------------------------
st.header("1. 혼동행렬")
n_test = len(test)
n_pos = int((y_test == 1).sum())
st.write(f"테스트용 **{n_test:,}명** 가운데 실제 뇌졸중인 사람은 **{n_pos:,}명**이에요.")

col_l, col_r = st.columns(2)
with col_l:
    st.subheader(LR_NAME)
    st.table(confusion_table(*counts[LR_NAME]))
with col_r:
    st.subheader(TREE_NAME)
    st.table(confusion_table(*counts[TREE_NAME]))
st.caption(
    "행은 실제, 열은 예측이에요. "
    "TP: 뇌졸중인데 뇌졸중이라 예측(맞힘) · FN: 뇌졸중인데 아니라고 예측(놓침) · "
    "FP: 아닌데 뇌졸중이라 예측(헛짚음) · TN: 아닌데 아니라고 예측(맞힘)"
)

# ---------------------------------------------------------------
# 네 가지 지표
# ---------------------------------------------------------------
st.header("2. 네 가지 지표")
rows = []
for name in (LR_NAME, TREE_NAME, BASE_NAME):
    acc_v, rec_v, pre_v, f1_v = scores(*counts[name])
    rows.append([pct(acc_v), pct(rec_v), pct(pre_v), pct(f1_v)])
metric_table = pd.DataFrame(
    rows,
    index=[LR_NAME, TREE_NAME, BASE_NAME],
    columns=["정확도", "재현율", "정밀도", "F1"],
)
st.table(metric_table)
st.caption(
    "정확도 = (TP + TN) ÷ 테스트용 전체 사람 수 · "
    "재현율 = TP ÷ 실제 뇌졸중인 사람 수 · "
    "정밀도 = TP ÷ 뇌졸중이라 예측한 사람 수 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율)"
)

# ---------------------------------------------------------------
# 놓친 환자 / 헛짚은 사람 목록
# ---------------------------------------------------------------
st.header("3. 놓친 환자와 헛짚은 사람")
chosen_name = st.radio(
    "목록을 볼 모델",
    [LR_NAME, TREE_NAME],
    horizontal=True,
)

view = test.copy()
view["pred"] = pd.Series(preds[chosen_name], index=test.index)
missed_full, missed_show = person_list(view[(view["stroke"] == 1) & (view["pred"] == 0)])
wrong_full, wrong_show = person_list(view[(view["stroke"] == 0) & (view["pred"] == 1)])

left, right = st.columns(2)
with left:
    title = f"놓친 환자 — 전체 {len(missed_full):,}명"
    if len(missed_full) > MAX_ROWS:
        title += f" (앞 {MAX_ROWS}명만 보여 줘요)"
    st.subheader(title)
    st.caption("실제 뇌졸중인데 아니라고 예측한 사람 · 나이가 많은 순서")
    if len(missed_full) == 0:
        st.write("해당하는 사람이 없어요.")
    else:
        st.dataframe(missed_show, hide_index=True, column_config=LIST_CONFIG)
with right:
    title = f"헛짚은 사람 — 전체 {len(wrong_full):,}명"
    if len(wrong_full) > MAX_ROWS:
        title += f" (앞 {MAX_ROWS}명만 보여 줘요)"
    st.subheader(title)
    st.caption("아닌데 뇌졸중이라고 예측한 사람 · 나이가 많은 순서")
    if len(wrong_full) == 0:
        st.write("해당하는 사람이 없어요.")
    else:
        st.dataframe(wrong_show, hide_index=True, column_config=LIST_CONFIG)

_, fp_n, fn_n, _ = counts[chosen_name]
st.write(
    f"{chosen_name}의 목록 사람 수: 놓친 환자 {len(missed_full):,}명 = 혼동행렬의 FN {fn_n:,}명, "
    f"헛짚은 사람 {len(wrong_full):,}명 = 혼동행렬의 FP {fp_n:,}명으로 같아요."
)
