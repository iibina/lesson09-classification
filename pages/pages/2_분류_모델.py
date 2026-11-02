import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

# 입력으로 고를 수 있는 속성과 우리말 이름
FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = ["hypertension", "heart_disease"]  # 0 또는 1 값만 가지는 속성
ANSWER_NAME = {0: "아님", 1: "뇌졸중"}  # stroke 0 = 아님, 1 = 뇌졸중(양성)

# 화면에 쓸 모델 이름 (교과서의 이름을 앞에)
LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

# 색
COLOR_NO = "#636EFA"  # 실제로 뇌졸중 아님 (점)
COLOR_YES = "#EF553B"  # 실제로 뇌졸중 (점)
FILL_NO = "rgba(99, 110, 250, 0.13)"  # 아님이라고 답하는 칸 (옅은 파랑)
FILL_YES = "rgba(239, 85, 59, 0.20)"  # 뇌졸중이라고 답하는 칸 (옅은 빨강)
NODE_NO = "#DCE3FF"
NODE_YES = "#FAD3CB"
LINE_COLOR = "#00CC96"

st.set_page_config(page_title=f"분류 모델 · {APP_TITLE}", page_icon=APP_ICON, layout="wide")

st.markdown(
    """
    <style>
    .acc-card { border: 1px solid rgba(128,128,128,0.35); border-radius: 12px;
                padding: 1rem 1.1rem; height: 100%; }
    .acc-name { font-size: 0.95rem; font-weight: 600; min-height: 3.4rem; }
    .acc-label { font-size: 0.85rem; opacity: 0.7; }
    .acc-big { font-size: 2.8rem; font-weight: 700; line-height: 1.2; }
    .acc-small { display: flex; justify-content: space-between; gap: 0.5rem;
                 font-size: 0.8rem; opacity: 0.75; margin-top: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def accuracy_card(name, train_acc, test_acc):
    """테스트 정확도를 크게, 훈련·테스트 정확도를 나란히 작게 적은 카드."""
    return (
        '<div class="acc-card">'
        f'<div class="acc-name">{name}</div>'
        '<div class="acc-label">테스트 데이터 정확도</div>'
        f'<div class="acc-big">{test_acc * 100:.2f}%</div>'
        '<div class="acc-small">'
        f"<span>훈련 데이터 {train_acc * 100:.2f}%</span>"
        f"<span>테스트 데이터 {test_acc * 100:.2f}%</span>"
        "</div></div>"
    )


def num(v):
    """숫자를 보기 좋게: 소수 둘째 자리까지, 끝의 0은 지움."""
    return f"{v:.2f}".rstrip("0").rstrip(".")


def line_in_box(a, b, c, x0, x1, y0, y1):
    """a*x + b*y + c = 0 인 직선이 상자(x0~x1, y0~y1) 안을 지나는 두 끝점. 안 지나면 None."""
    pts = []
    eps = 1e-9
    if abs(b) > 1e-12:
        for x in (x0, x1):
            y = -(a * x + c) / b
            if y0 - eps <= y <= y1 + eps:
                pts.append((x, min(max(y, y0), y1)))
    if abs(a) > 1e-12:
        for y in (y0, y1):
            x = -(b * y + c) / a
            if x0 - eps <= x <= x1 + eps:
                pts.append((min(max(x, x0), x1), y))
    uniq = []
    for p in pts:
        if all(abs(p[0] - q[0]) > 1e-6 or abs(p[1] - q[1]) > 1e-6 for q in uniq):
            uniq.append(p)
    return (uniq[0], uniq[1]) if len(uniq) >= 2 else None


def leaf_boxes(tree_, ix, iy, fixed, box, node=0):
    """두 축(ix, iy)만 움직이고 나머지는 fixed 값에 세웠을 때,
    나무가 그림을 나눈 칸(끝 마디)들의 사각형과 마디 번호를 돌려준다."""
    left = tree_.children_left[node]
    right = tree_.children_right[node]
    if left == -1:  # 끝 마디
        return [(box, node)]
    f = int(tree_.feature[node])
    thr = float(tree_.threshold[node])
    x0, x1, y0, y1 = box
    if f == ix:
        parts = ((left, (x0, min(x1, thr), y0, y1)), (right, (max(x0, thr), x1, y0, y1)))
    elif f == iy:
        parts = ((left, (x0, x1, y0, min(y1, thr))), (right, (x0, x1, max(y0, thr), y1)))
    else:  # 고정한 속성: 그 값이 가는 쪽으로만 내려간다
        return leaf_boxes(tree_, ix, iy, fixed, box, left if fixed[f] <= thr else right)
    out = []
    for child, b in parts:
        if b[0] < b[1] and b[2] < b[3]:
            out += leaf_boxes(tree_, ix, iy, fixed, b, child)
    return out


# ---------------------------------------------------------------
# 제목과 데이터
# ---------------------------------------------------------------
st.title(f"{APP_ICON} 분류 모델")
st.write(
    "stroke가 **1이면 뇌졸중, 0이면 아님**이에요. 뇌졸중(stroke = 1)을 **양성**으로 두고 "
    "모델이 뇌졸중인지 아닌지 맞히게 해 봐요."
)

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# 번호(id) 순으로 정렬
data = raw.sort_values("id").reset_index(drop=True)
for c in FEATURES:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data["stroke"] = data["stroke"].astype(int)

# ---------------------------------------------------------------
# 입력 속성 고르기
# ---------------------------------------------------------------
st.header("1. 입력으로 사용할 속성 고르기")
chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: NAMES[c],
)
selected = [c for c in FEATURES if c in chosen]  # 항상 같은 순서로

if len(selected) < 2:
    st.info("입력으로 사용할 속성을 **두 개 이상** 골라 주세요. 두 개 이상 고르면 아래 내용이 나타나요.")
    st.stop()

# ---------------------------------------------------------------
# 훈련용 / 테스트용 나누기
# ---------------------------------------------------------------
# 번호 순으로 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정
is_test = pd.Series(data.index % 10 < 3, index=data.index)
train = data[~is_test]
test = data[is_test]

X_train = train[selected].copy()
X_test = test[selected].copy()
y_train = train["stroke"]
y_test = test["stroke"]

st.write(
    f"전체 {len(data):,}명을 번호 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 정했어요. "
    f"**테스트용 {len(test):,}명**, **훈련용 {len(train):,}명**이에요."
)

# 체질량지수는 고른 경우에만, 훈련용의 중앙값으로 빈 값을 채운다
if "bmi" in selected:
    bmi_median = float(X_train["bmi"].median())
    n_filled = int(X_train["bmi"].isna().sum() + X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    st.write(
        f"체질량지수가 빈 {n_filled:,}명은 훈련용의 중앙값({bmi_median:.2f})으로 채웠어요."
    )

# ---------------------------------------------------------------
# 모델 만들기
# ---------------------------------------------------------------
# 크기 맞추기는 훈련용으로만 배운다 (로지스틱 회귀에 사용)
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)

lr = LogisticRegression().fit(Xs_train, y_train)

# 의사결정트리는 크기를 맞추지 않은 원래 값으로 배워서, 질문의 숫자를 그대로 읽을 수 있다
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)

base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

acc = {
    "lr": (accuracy_score(y_train, lr.predict(Xs_train)), accuracy_score(y_test, lr.predict(Xs_test))),
    "tree": (accuracy_score(y_train, tree.predict(X_train)), accuracy_score(y_test, tree.predict(X_test))),
    "base": (accuracy_score(y_train, base.predict(X_train)), accuracy_score(y_test, base.predict(X_test))),
}

# ---------------------------------------------------------------
# 정확도 카드
# ---------------------------------------------------------------
st.header("2. 모델의 정확도")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(accuracy_card(LR_NAME, *acc["lr"]), unsafe_allow_html=True)
with c2:
    st.markdown(accuracy_card(TREE_NAME, *acc["tree"]), unsafe_allow_html=True)
with c3:
    st.markdown(accuracy_card(BASE_NAME, *acc["base"]), unsafe_allow_html=True)
st.caption("기준 모델은 입력을 보지 않아요. 두 모델의 정확도가 이 숫자보다 높은지 비교해 보세요.")

# ---------------------------------------------------------------
# 산점도: 테스트 데이터 + 로지스틱 회귀 0.5 선 + 의사결정트리 칸
# ---------------------------------------------------------------
st.header("3. 테스트 데이터 그림")

sx, sy = st.columns(2)
x_col = sx.selectbox("가로축", selected, index=0, format_func=lambda c: NAMES[c])
y_options = [c for c in selected if c != x_col]
y_col = sy.selectbox("세로축", y_options, index=0, format_func=lambda c: NAMES[c])
ix, iy = selected.index(x_col), selected.index(y_col)

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세운다
fixed = [float(X_test[c].median()) for c in selected]
others = [c for c in selected if c not in (x_col, y_col)]


def axis_range(col):
    if col in BINARY:
        return -0.5, 1.5
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


xr0, xr1 = axis_range(x_col)
yr0, yr1 = axis_range(y_col)

# --- 로지스틱 회귀의 0.5 선 (a*x + b*y + c = 0) ---
w = lr.coef_[0]
b0 = float(lr.intercept_[0])
mu = scaler.mean_
sd = scaler.scale_
a_coef = float(w[ix] / sd[ix])
b_coef = float(w[iy] / sd[iy])
c_coef = b0 - float(w[ix] * mu[ix] / sd[ix]) - float(w[iy] * mu[iy] / sd[iy])
for k in range(len(selected)):
    if k not in (ix, iy):
        c_coef += float(w[k]) * (fixed[k] - float(mu[k])) / float(sd[k])
line = line_in_box(a_coef, b_coef, c_coef, xr0, xr1, yr0, yr1)

# --- 의사결정트리가 그림을 나눈 칸 ---
tree_ = tree.tree_
boxes = leaf_boxes(tree_, ix, iy, fixed, (xr0, xr1, yr0, yr1))


def leaf_answer(node):
    return int(tree.classes_[int(tree_.value[node][0].argmax())])


fig = go.Figure()

for (bx0, bx1, by0, by1), node in boxes:
    fig.add_shape(
        type="rect",
        x0=bx0, x1=bx1, y0=by0, y1=by1,
        fillcolor=FILL_YES if leaf_answer(node) == 1 else FILL_NO,
        line=dict(color="rgba(128,128,128,0.55)", width=1),
        layer="below",
    )

shown_answers = {leaf_answer(node) for _, node in boxes}
for ans, fill in ((0, FILL_NO), (1, FILL_YES)):
    if ans in shown_answers:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(symbol="square", size=14, color=fill,
                            line=dict(color="rgba(128,128,128,0.6)", width=1)),
                name=f"칠한 칸: 의사결정트리가 '{ANSWER_NAME[ans]}'이라고 답하는 칸",
            )
        )

# 테스트 데이터 점 (0/1 값 속성은 겹치지 않게 그림에서만 살짝 흩뜨림)
xs = X_test[x_col].tolist()
ys = X_test[y_col].tolist()
orig = [[num(a), num(b)] for a, b in zip(xs, ys)]
if x_col in BINARY:
    xs = [v + 0.3 * (((j * 7919) % 101) / 100 - 0.5) for j, v in enumerate(xs)]
if y_col in BINARY:
    ys = [v + 0.3 * (((j * 104729) % 103) / 102 - 0.5) for j, v in enumerate(ys)]
actual = y_test.tolist()

for label, cls, color, opacity in (
    ("실제로 뇌졸중 아님(0)", 0, COLOR_NO, 0.55),
    ("실제로 뇌졸중(1)", 1, COLOR_YES, 0.95),
):
    idx = [j for j, v in enumerate(actual) if v == cls]
    fig.add_trace(
        go.Scatter(
            x=[xs[j] for j in idx],
            y=[ys[j] for j in idx],
            mode="markers",
            name=label,
            marker=dict(color=color, size=6, opacity=opacity),
            customdata=[orig[j] for j in idx],
            hovertemplate=(
                f"{NAMES[x_col]}: %{{customdata[0]}}<br>"
                f"{NAMES[y_col]}: %{{customdata[1]}}<extra>{label}</extra>"
            ),
        )
    )

if line is not None:
    (lx0, ly0), (lx1, ly1) = line
    fig.add_trace(
        go.Scatter(
            x=[lx0, lx1], y=[ly0, ly1], mode="lines",
            line=dict(color=LINE_COLOR, width=3, dash="dash"),
            name="로지스틱 회귀의 0.5 선",
        )
    )


def axis_layout(col, lo, hi):
    d = dict(title=NAMES[col], range=[lo, hi])
    if col in BINARY:
        d.update(tickmode="array", tickvals=[0, 1], ticktext=["없음(0)", "있음(1)"])
    return d


fig.update_layout(
    xaxis=axis_layout(x_col, xr0, xr1),
    yaxis=axis_layout(y_col, yr0, yr1),
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    height=600,
    margin=dict(t=30),
)
st.plotly_chart(fig)

# 그림 아래 설명 글
if others:
    parts = [
        f"{NAMES[c]} = {num(fixed[selected.index(c)])}" for c in others
    ]
    st.write(
        "두 축이 아닌 속성은 **테스트 데이터의 중앙값**에 세워 두고 계산했어요: " + ", ".join(parts)
    )
else:
    st.write("고른 속성이 두 개뿐이라 따로 세워 둔 값은 없어요.")

if line is not None:
    st.write(
        f"초록 점선은 {LR_NAME}이 뇌졸중일 확률을 0.5로 보는 자리예요. "
        "선의 한쪽은 '뇌졸중', 반대쪽은 '아님'이라고 답해요."
    )
else:
    center_x, center_y = (xr0 + xr1) / 2, (yr0 + yr1) / 2
    side = a_coef * center_x + b_coef * center_y + c_coef
    word = "아님" if side < 0 else "뇌졸중"
    st.write(
        f"{LR_NAME}의 0.5 선은 **이 그림 밖에 있어서 보이지 않아요.** "
        f"그림 안은 모두 같은 쪽이라 '{word}'이라고 답해요."
    )

if x_col in BINARY or y_col in BINARY:
    st.caption("고혈압·심장병은 0 또는 1이라 점이 겹쳐요. 그림에서만 점을 살짝 흩뜨려 찍었고, 계산에는 영향이 없어요.")

# ---------------------------------------------------------------
# 의사결정트리의 질문 가지 그림
# ---------------------------------------------------------------
st.header(f"4. {TREE_NAME}이 던진 질문")

# 마디마다 그 자리에 온 훈련용 사람 수와 그중 뇌졸중인 사람 수
path = tree.decision_path(X_train)
node_n = [int(v) for v in path.sum(axis=0).tolist()[0]]
node_k = [int(v) for v in path.T.dot(y_train.astype(int).to_numpy()).tolist()]

dot = [
    "digraph Tree {",
    'graph [rankdir=TB, nodesep=0.3, ranksep=0.5, bgcolor="transparent"];',
    'node [shape=box, style="rounded,filled", fillcolor="white", fontcolor="black", '
    'color="#888888", fontsize=11, margin="0.15,0.08"];',
    'edge [fontsize=11, color="#888888", fontcolor="#888888"];',
]
leaf_answers = []
asked_idx = set()
for node in range(tree_.node_count):
    n, k = node_n[node], node_k[node]
    pct = k / n * 100 if n else 0.0
    stats = f"훈련용 {n:,}명\\n뇌졸중 {k:,}명 ({pct:.1f}%)"
    left = int(tree_.children_left[node])
    right = int(tree_.children_right[node])
    if left == -1:  # 더 묻지 않고 답을 내는 마디
        ans = leaf_answer(node)
        leaf_answers.append(ans)
        fill = NODE_YES if ans == 1 else NODE_NO
        dot.append(f'n{node} [label="답: {ANSWER_NAME[ans]}\\n{stats}", fillcolor="{fill}"];')
    else:
        f = int(tree_.feature[node])
        asked_idx.add(f)
        col = selected[f]
        thr = float(tree_.threshold[node])
        if col in BINARY:
            question = f"{NAMES[col]}이 있나요?"
            yes_child, no_child = right, left
        else:
            question = f"{NAMES[col]} ≤ {num(thr)}?"
            yes_child, no_child = left, right
        dot.append(f'n{node} [label="{question}\\n{stats}"];')
        dot.append(f'n{node} -> n{yes_child} [label="예"];')
        dot.append(f'n{node} -> n{no_child} [label="아니요"];')
dot.append("}")

st.graphviz_chart("\n".join(dot))
st.caption("각 칸에는 그 자리에 온 훈련용 사람 수, 그중 실제 뇌졸중인 사람 수와 비율이 적혀 있어요. "
           "파랑 칸은 '아님', 빨강 칸은 '뇌졸중'이라고 답하는 마디예요.")

n_leaf = len(leaf_answers)
n_no = sum(1 for a in leaf_answers if a == 0)
st.write(f"답을 내는 마디는 모두 **{n_leaf}칸**이고, 그중 **{n_no}칸**이 '아님'이라고 답해요.")

asked_names = [NAMES[selected[i]] for i in sorted(asked_idx)]
if asked_names:
    st.write("고른 속성 가운데 이 나무가 실제로 물은 것: **" + ", ".join(asked_names) + "**")
else:
    st.write("고른 속성 가운데 이 나무가 실제로 물은 것: **하나도 없어요** (질문 없이 바로 답했어요)")
