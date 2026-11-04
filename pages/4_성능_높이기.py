import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
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

# 화면에 쓸 모델 이름 (교과서의 이름을 앞에)
LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"

CANNOT = "계산할 수 없음"
BEFORE = "가중치 맞추기 전"
AFTER = "가중치 맞춘 뒤"

# 색
COLOR_NO = "#636EFA"  # 실제로 뇌졸중 아님 (점)
COLOR_YES = "#EF553B"  # 실제로 뇌졸중 (점)
BEFORE_COLOR = "#E6A800"  # 맞추기 전 0.5 경계선·평면
AFTER_COLOR = "#00B386"  # 맞춘 뒤 0.5 경계선·평면

st.set_page_config(page_title=f"성능 높이기 · {APP_TITLE}", page_icon=APP_ICON, layout="wide")


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


def num(v):
    """숫자를 보기 좋게: 소수 둘째 자리까지, 끝의 0은 지움."""
    return f"{v:.2f}".rstrip("0").rstrip(".")


def compare_table(before_counts, after_counts):
    """네 칸과 네 지표를 전·후로 나란히 놓은 표."""
    rows = []
    labels = [
        "TP (찾아낸 환자)",
        "FN (놓친 환자)",
        "FP (헛짚은 사람)",
        "TN (아닌 사람을 아니라고 맞힘)",
    ]
    tp0, fp0, fn0, tn0 = before_counts
    tp1, fp1, fn1, tn1 = after_counts
    cells0 = [tp0, fn0, fp0, tn0]
    cells1 = [tp1, fn1, fp1, tn1]
    for label, v0, v1 in zip(labels, cells0, cells1):
        rows.append([label, f"{v0:,}명", f"{v1:,}명"])
    s0 = scores(*before_counts)
    s1 = scores(*after_counts)
    for label, a, b in zip(["정확도", "재현율", "정밀도", "F1"], s0, s1):
        rows.append([label, pct(a), pct(b)])
    table = pd.DataFrame(rows, columns=["", BEFORE, AFTER]).set_index("")
    return table


def change_line(before_counts, after_counts):
    tp0, fp0, fn0, _ = before_counts
    tp1, fp1, fn1, _ = after_counts
    return (
        f"찾아낸 환자 {tp0:,}명에서 {tp1:,}명으로, "
        f"놓친 환자 {fn0:,}명에서 {fn1:,}명으로, "
        f"헛짚은 사람 {fp0:,}명에서 {fp1:,}명으로 바뀌었어요."
    )


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


def plane_mesh(coefs, const, ranges):
    """a*x + b*y + c*z + const = 0 인 평면이 상자 안을 지나는 부분을 삼각형 조각으로 돌려준다.
    (x, y, z, i, j, k) 또는 상자 밖이면 None."""

    def f(p):
        return coefs[0] * p[0] + coefs[1] * p[1] + coefs[2] * p[2] + const

    corners = [(x, y, z) for x in ranges[0] for y in ranges[1] for z in ranges[2]]
    pts = [p for p in corners if abs(f(p)) < 1e-9]
    for i, p in enumerate(corners):
        for q in corners[i + 1:]:
            if sum(1 for k in range(3) if p[k] != q[k]) != 1:
                continue  # 모서리(한 좌표만 다른 두 꼭짓점)만 본다
            fp, fq = f(p), f(q)
            if fp * fq < 0:
                t = fp / (fp - fq)
                pts.append(tuple(p[k] + t * (q[k] - p[k]) for k in range(3)))
    uniq = []
    for p in pts:
        if all(max(abs(p[k] - q[k]) for k in range(3)) > 1e-6 for q in uniq):
            uniq.append(p)
    if len(uniq) < 3:
        return None

    # 평면 위의 점들을 한 바퀴 돌며 차례대로 늘어놓는다
    cen = [sum(p[k] for p in uniq) / len(uniq) for k in range(3)]
    n = coefs

    def cross(u, v):
        return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])

    def dot(u, v):
        return sum(u[k] * v[k] for k in range(3))

    norm = math.sqrt(dot(n, n))
    helper = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 * norm else (0.0, 1.0, 0.0)
    u = cross(n, helper)
    v = cross(n, u)
    uniq.sort(key=lambda p: math.atan2(dot([p[k] - cen[k] for k in range(3)], v),
                                       dot([p[k] - cen[k] for k in range(3)], u)))
    m = len(uniq)
    return (
        [p[0] for p in uniq],
        [p[1] for p in uniq],
        [p[2] for p in uniq],
        [0] * (m - 2),
        list(range(1, m - 1)),
        list(range(2, m)),
    )


def jitter(values, col, mult, mod):
    """0/1 값 속성은 점이 겹치므로 그림에서만 살짝 흩뜨린다."""
    if col not in BINARY:
        return list(values)
    return [v + 0.3 * (((j * mult) % mod) / (mod - 1) - 0.5) for j, v in enumerate(values)]


# ---------------------------------------------------------------
# 제목과 입력 속성 고르기
# ---------------------------------------------------------------
st.title(f"{APP_ICON} 성능 높이기")

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
# 데이터 읽기, 나누기 ('채점표' 페이지와 같은 방법)
# ---------------------------------------------------------------
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

# 번호 순으로 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정
is_test = pd.Series(data.index % 10 < 3, index=data.index)
train = data[~is_test]
test = data[is_test]

X_train = train[selected].copy()
X_test = test[selected].copy()
y_train = train["stroke"]
y_test = test["stroke"]

# 체질량지수는 고른 경우에만, 훈련용의 중앙값으로 빈 값을 채운다
if "bmi" in selected:
    bmi_median = float(X_train["bmi"].median())
    n_filled = int(X_train["bmi"].isna().sum() + X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    st.write(f"체질량지수가 빈 {n_filled:,}명은 훈련용의 중앙값({bmi_median:.2f})으로 채웠어요.")

n_test = len(test)
n_pos = int((y_test == 1).sum())
train_rate = float((y_train == 1).mean()) * 100
st.write(
    f"테스트용 **{n_test:,}명** 가운데 실제 뇌졸중인 사람은 **{n_pos:,}명**이에요. "
    f"훈련용에서도 뇌졸중인 사람은 {train_rate:.1f}%뿐이에요."
)

# ---------------------------------------------------------------
# 모델 학습: 가중치를 맞추기 전(기본)과 맞춘 뒤(balanced)
# ---------------------------------------------------------------
# 크기 맞추기는 훈련용으로만 배운다 (로지스틱 회귀에 사용)
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)
mu = scaler.mean_
sd = scaler.scale_


def fit_lr(class_weight):
    return LogisticRegression(class_weight=class_weight).fit(Xs_train, y_train)


def fit_tree(class_weight):
    # 질문은 세 번까지, 마지막 마디에 사람이 다섯 명 미만이 되게는 나누지 않고, 난수 고정
    # (크기를 맞추지 않은 원래 값으로 배워서 질문의 숫자를 그대로 읽을 수 있다)
    model = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, random_state=42, class_weight=class_weight
    )
    return model.fit(X_train, y_train)


lr_before, lr_after = fit_lr(None), fit_lr("balanced")
tree_before, tree_after = fit_tree(None), fit_tree("balanced")

actual = y_test.tolist()
counts = {
    LR_NAME: (
        confusion(actual, lr_before.predict(Xs_test).tolist()),
        confusion(actual, lr_after.predict(Xs_test).tolist()),
    ),
    TREE_NAME: (
        confusion(actual, tree_before.predict(X_test).tolist()),
        confusion(actual, tree_after.predict(X_test).tolist()),
    ),
}

# ---------------------------------------------------------------
# 전·후 비교 표
# ---------------------------------------------------------------
st.header("1. 가중치를 맞추기 전과 맞춘 뒤")
st.write(
    "뇌졸중인 사람이 드물어서, 모델이 거의 '아님'이라고만 답해도 정확도는 높게 나와요. "
    "**양쪽의 가중치를 맞추면** 드문 쪽(뇌졸중)의 틀림을 더 무겁게 다루며 학습해요."
)

col_l, col_r = st.columns(2)
for col, name in ((col_l, LR_NAME), (col_r, TREE_NAME)):
    before_c, after_c = counts[name]
    with col:
        st.subheader(name)
        st.table(compare_table(before_c, after_c))
        st.write(change_line(before_c, after_c))

# ---------------------------------------------------------------
# 0.5 경계선 계산 준비
# ---------------------------------------------------------------
# 두 축(또는 세 축)이 아닌 속성은 테스트 데이터의 중앙값에 세운다
fixed = [float(X_test[c].median()) for c in selected]


def axis_range(col):
    if col in BINARY:
        return -0.5, 1.5
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


def boundary(model, axes):
    """로지스틱 회귀의 0.5 경계: coefs·(축 값) + const = 0 (원래 값 기준)."""
    w = model.coef_[0]
    b0 = float(model.intercept_[0])
    coefs = [float(w[i] / sd[i]) for i in axes]
    const = b0 - sum(float(w[i] * mu[i] / sd[i]) for i in axes)
    for k in range(len(selected)):
        if k not in axes:
            const += float(w[k]) * (fixed[k] - float(mu[k])) / float(sd[k])
    return coefs, const


def axis_layout(col, lo, hi):
    d = dict(title=NAMES[col], range=[lo, hi])
    if col in BINARY:
        d.update(tickmode="array", tickvals=[0, 1], ticktext=["없음(0)", "있음(1)"])
    return d


def fixed_text(axis_cols):
    rest = [c for c in selected if c not in axis_cols]
    if not rest:
        return "고른 속성이 모두 축으로 쓰여서 따로 세워 둔 값은 없어요."
    parts = [f"{NAMES[c]} = {num(fixed[selected.index(c)])}" for c in rest]
    return "축이 아닌 속성은 **테스트 데이터의 중앙값**에 세워 두고 계산했어요: " + ", ".join(parts)


# ---------------------------------------------------------------
# 2차원 산점도
# ---------------------------------------------------------------
st.header("2. 산점도와 0.5 경계선")
st.write(
    f"{LR_NAME}이 뇌졸중일 확률을 0.5로 보는 자리를, "
    "가중치를 맞추기 전(금색 선)과 맞춘 뒤(초록 선)로 겹쳐 그려요. "
    "선의 한쪽은 '뇌졸중', 반대쪽은 '아님'이라고 답해요."
)

sx, sy = st.columns(2)
x_col = sx.selectbox("가로축", selected, index=0, format_func=lambda c: NAMES[c], key="x2d")
y_options = [c for c in selected if c != x_col]
y_col = sy.selectbox("세로축", y_options, index=0, format_func=lambda c: NAMES[c], key="y2d")
ix, iy = selected.index(x_col), selected.index(y_col)

xr0, xr1 = axis_range(x_col)
yr0, yr1 = axis_range(y_col)

fig2 = go.Figure()
xs = X_test[x_col].tolist()
ys = X_test[y_col].tolist()
orig2 = [[num(a), num(b)] for a, b in zip(xs, ys)]
xs = jitter(xs, x_col, 7919, 101)
ys = jitter(ys, y_col, 104729, 103)

for label, cls, color, opacity in (
    ("실제로 뇌졸중 아님(0)", 0, COLOR_NO, 0.55),
    ("실제로 뇌졸중(1)", 1, COLOR_YES, 0.95),
):
    idx = [j for j, v in enumerate(actual) if v == cls]
    fig2.add_trace(
        go.Scatter(
            x=[xs[j] for j in idx],
            y=[ys[j] for j in idx],
            mode="markers",
            name=label,
            marker=dict(color=color, size=6, opacity=opacity),
            customdata=[orig2[j] for j in idx],
            hovertemplate=(
                f"{NAMES[x_col]}: %{{customdata[0]}}<br>"
                f"{NAMES[y_col]}: %{{customdata[1]}}<extra>{label}</extra>"
            ),
        )
    )

notes2 = []
for model, label, color in (
    (lr_before, "가중치를 맞추기 전의 0.5 경계선", BEFORE_COLOR),
    (lr_after, "가중치를 맞춘 뒤의 0.5 경계선", AFTER_COLOR),
):
    coefs, const = boundary(model, [ix, iy])
    line = line_in_box(coefs[0], coefs[1], const, xr0, xr1, yr0, yr1)
    if line is not None:
        (lx0, ly0), (lx1, ly1) = line
        fig2.add_trace(
            go.Scatter(
                x=[lx0, lx1], y=[ly0, ly1], mode="lines",
                line=dict(color=color, width=3, dash="dash"),
                name=label,
            )
        )
    else:
        side = coefs[0] * (xr0 + xr1) / 2 + coefs[1] * (yr0 + yr1) / 2 + const
        word = "아님" if side < 0 else "뇌졸중"
        notes2.append(
            f"{label}은 **이 그림 밖에 있어서 보이지 않아요.** 그림 안은 모두 같은 쪽이라 '{word}'이라고 답해요."
        )

fig2.update_layout(
    xaxis=axis_layout(x_col, xr0, xr1),
    yaxis=axis_layout(y_col, yr0, yr1),
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    height=600,
    margin=dict(t=30),
)
st.plotly_chart(fig2)

st.write(fixed_text([x_col, y_col]))
for t in notes2:
    st.write(t)
if x_col in BINARY or y_col in BINARY:
    st.caption("고혈압·심장병은 0 또는 1이라 점이 겹쳐요. 그림에서만 점을 살짝 흩뜨려 찍었고, 계산에는 영향이 없어요.")

# ---------------------------------------------------------------
# 3차원 산점도
# ---------------------------------------------------------------
st.header("3. 3차원 산점도와 0.5 평면")

if len(selected) < 3:
    st.info("입력 속성을 **세 개 이상** 고르면 3차원 그림도 볼 수 있어요.")
else:
    st.write("마우스로 끌면 그림을 돌려 볼 수 있고, 휠로 크기를 바꿀 수 있어요.")
    s1, s2, s3 = st.columns(3)
    a_col = s1.selectbox("가로축(x)", selected, index=0, format_func=lambda c: NAMES[c], key="x3d")
    b_options = [c for c in selected if c != a_col]
    b_col = s2.selectbox("세로축(y)", b_options, index=0, format_func=lambda c: NAMES[c], key="y3d")
    c_options = [c for c in selected if c not in (a_col, b_col)]
    c_col = s3.selectbox("깊이축(z)", c_options, index=0, format_func=lambda c: NAMES[c], key="z3d")
    axes_cols = [a_col, b_col, c_col]
    axes_idx = [selected.index(c) for c in axes_cols]
    ranges3 = [axis_range(c) for c in axes_cols]

    fig3 = go.Figure()
    vals = [X_test[c].tolist() for c in axes_cols]
    orig3 = [[num(vals[0][j]), num(vals[1][j]), num(vals[2][j])] for j in range(n_test)]
    pos = [
        jitter(vals[0], a_col, 7919, 101),
        jitter(vals[1], b_col, 104729, 103),
        jitter(vals[2], c_col, 1299709, 107),
    ]

    for label, cls, color, opacity in (
        ("실제로 뇌졸중 아님(0)", 0, COLOR_NO, 0.45),
        ("실제로 뇌졸중(1)", 1, COLOR_YES, 0.95),
    ):
        idx = [j for j, v in enumerate(actual) if v == cls]
        fig3.add_trace(
            go.Scatter3d(
                x=[pos[0][j] for j in idx],
                y=[pos[1][j] for j in idx],
                z=[pos[2][j] for j in idx],
                mode="markers",
                name=label,
                marker=dict(color=color, size=3, opacity=opacity),
                customdata=[orig3[j] for j in idx],
                hovertemplate=(
                    f"{NAMES[a_col]}: %{{customdata[0]}}<br>"
                    f"{NAMES[b_col]}: %{{customdata[1]}}<br>"
                    f"{NAMES[c_col]}: %{{customdata[2]}}<extra>{label}</extra>"
                ),
            )
        )

    notes3 = []
    for model, label, color in (
        (lr_before, "가중치를 맞추기 전의 0.5 평면", BEFORE_COLOR),
        (lr_after, "가중치를 맞춘 뒤의 0.5 평면", AFTER_COLOR),
    ):
        coefs, const = boundary(model, axes_idx)
        mesh = plane_mesh(coefs, const, ranges3)
        if mesh is not None:
            mx, my, mz, mi, mj, mk = mesh
            fig3.add_trace(
                go.Mesh3d(
                    x=mx, y=my, z=mz, i=mi, j=mj, k=mk,
                    color=color, opacity=0.40, name=label,
                    showlegend=True, hoverinfo="skip", flatshading=True,
                )
            )
        else:
            center = [(r[0] + r[1]) / 2 for r in ranges3]
            side = sum(coefs[k] * center[k] for k in range(3)) + const
            word = "아님" if side < 0 else "뇌졸중"
            notes3.append(
                f"{label}은 **이 그림 상자 밖에 있어서 보이지 않아요.** 상자 안은 모두 같은 쪽이라 '{word}'이라고 답해요."
            )

    fig3.update_layout(
        scene=dict(
            xaxis=axis_layout(a_col, *ranges3[0]),
            yaxis=axis_layout(b_col, *ranges3[1]),
            zaxis=axis_layout(c_col, *ranges3[2]),
        ),
        legend=dict(orientation="h", yanchor="top", y=0, xanchor="left", x=0),
        height=700,
        margin=dict(t=10, l=0, r=0, b=0),
    )
    st.plotly_chart(fig3)

    st.write(fixed_text(axes_cols))
    for t in notes3:
        st.write(t)
    if any(c in BINARY for c in axes_cols):
        st.caption("고혈압·심장병은 0 또는 1이라 점이 겹쳐요. 그림에서만 점을 살짝 흩뜨려 찍었고, 계산에는 영향이 없어요.")
