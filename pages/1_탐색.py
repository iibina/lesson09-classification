import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/iibina/data/refs/heads/main/stroke.csv"

st.set_page_config(page_title=f"탐색 · {APP_TITLE}", page_icon=APP_ICON, layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title(f"{APP_ICON} 탐색")
st.caption("데이터를 그림과 표로 살펴봐요.")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# 그림과 표에서 쓸 한글 이름
df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map({0: "겪지 않음", 1: "겪음"})
GROUP_ORDER = ["겪지 않음", "겪음"]

# ---------------------------------------------------------------
# 1) 나이와 평균 혈당의 분포
# ---------------------------------------------------------------
st.header("1. 나이와 평균 혈당의 분포")
col_a, col_b = st.columns(2)

with col_a:
    fig = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=40,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# 2) 뇌졸중 여부에 따른 나이와 평균 혈당 비교
# ---------------------------------------------------------------
st.header("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
col_c, col_d = st.columns(2)

with col_c:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="age",
        category_orders={"뇌졸중 여부": GROUP_ORDER},
        title="나이 비교",
        labels={"age": "나이"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col_d:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        category_orders={"뇌졸중 여부": GROUP_ORDER},
        title="평균 혈당 비교",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    st.plotly_chart(fig, use_container_width=True)

mean_table = (
    df.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .reindex(GROUP_ORDER)
    .round(2)
    .reset_index()
)
mean_table.columns = ["그룹", "나이 평균", "평균 혈당 평균"]
st.subheader("두 그룹의 평균값")
st.dataframe(mean_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# 3) 고혈압 · 심장병 여부별 뇌졸중 비율
# ---------------------------------------------------------------
st.header("3. 고혈압·심장병 여부별 뇌졸중 비율")


def stroke_rate_by(column, title):
    """column 값(0/1)별로 뇌졸중 비율(%)을 구해 막대그래프를 만든다."""
    rate = df.groupby(column)["stroke"].mean().mul(100).round(2).reset_index()
    rate.columns = ["값", "뇌졸중 비율(%)"]
    rate["구분"] = rate["값"].map({0: "없음", 1: "있음"})
    fig = px.bar(
        rate,
        x="구분",
        y="뇌졸중 비율(%)",
        category_orders={"구분": ["없음", "있음"]},
        text="뇌졸중 비율(%)",
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(xaxis_title=None)
    fig.update_yaxes(range=[0, max(rate["뇌졸중 비율(%)"].max() * 1.2, 1)])
    return fig


col_e, col_f = st.columns(2)
with col_e:
    st.plotly_chart(
        stroke_rate_by("hypertension", "고혈압 여부별 뇌졸중 비율"),
        use_container_width=True,
    )
with col_f:
    st.plotly_chart(
        stroke_rate_by("heart_disease", "심장병 여부별 뇌졸중 비율"),
        use_container_width=True,
    )

# ---------------------------------------------------------------
# 4) bmi가 빈 사람들의 뇌졸중 비율
# ---------------------------------------------------------------
st.header("4. bmi가 비어 있는 사람들")

bmi_empty = df[df["bmi"].isna()]
bmi_table = pd.DataFrame(
    {
        "구분": ["bmi가 빈 사람", "전체"],
        "사람 수": [len(bmi_empty), len(df)],
        "뇌졸중 사람 수": [int((bmi_empty["stroke"] == 1).sum()), int((df["stroke"] == 1).sum())],
    }
)
bmi_table["뇌졸중 비율(%)"] = (
    bmi_table["뇌졸중 사람 수"] / bmi_table["사람 수"] * 100
).round(2)
st.dataframe(bmi_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# 5) 흡연 상태별 사람 수
# ---------------------------------------------------------------
st.header("5. 흡연 상태별 사람 수")

smoking = df["smoking_status"].value_counts(dropna=False).reset_index()
smoking.columns = ["흡연 상태(smoking_status)", "사람 수"]
st.dataframe(smoking, use_container_width=True, hide_index=True)
