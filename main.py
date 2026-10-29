import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/iibina/data/refs/heads/main/stroke.csv"

# 내 기록을 저장할 위치: main.py 옆의 "내_기록" 폴더
RECORD_DIR = Path(__file__).parent / "내_기록"
RECORD_FILE = RECORD_DIR / "기록.json"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# 큰 숫자 카드 모양 꾸미기
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 2.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"] p { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- 저장 / 불러오기 ----------
def load_record():
    """내_기록/기록.json을 읽어 온다. 없거나 깨졌으면 빈 기록을 돌려준다."""
    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        meanings = data.get("우리말_뜻", {})
        source = data.get("출처", "")
        if not isinstance(meanings, dict):
            meanings = {}
        return {str(k): str(v) for k, v in meanings.items()}, str(source)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}, ""


def save_record(meanings, source):
    RECORD_DIR.mkdir(exist_ok=True)
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"우리말_뜻": meanings, "출처": source},
            f,
            ensure_ascii=False,
            indent=2,
        )


# ---------- 데이터 ----------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def describe_values(series):
    """열의 '값의 종류'를 짧은 글로 설명한다."""
    s = series.dropna()
    if s.empty:
        return "값 없음"
    if series.dtype == object:
        kinds = sorted(s.unique().tolist())
        return f"{len(kinds)}가지: " + ", ".join(map(str, kinds))
    if s.nunique() <= 2:
        return "0 또는 1 (두 가지)"
    return f"숫자: {s.min():g} ~ {s.max():g}"


# ---------- 화면 ----------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("첫 화면 · 이 데이터는 무엇일까요?")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# 처음 한 번만 저장된 기록을 읽어 둔다 (편집 중 표가 초기화되지 않도록)
if "base_table" not in st.session_state:
    saved_meanings, saved_source = load_record()
    table = pd.DataFrame(
        {
            "열 이름": df.columns,
            "우리말 뜻": [saved_meanings.get(c, "") for c in df.columns],
            "값의 종류": [describe_values(df[c]) for c in df.columns],
            "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
        }
    )
    st.session_state["base_table"] = table
    st.session_state["saved_meanings"] = saved_meanings
    st.session_state["saved_source"] = saved_source

# 1) 큰 숫자 카드 네 개
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100

c1, c2, c3, c4 = st.columns(4)
with c1.container(border=True):
    st.metric("전체 사람 수", f"{total:,}명")
with c2.container(border=True):
    st.metric("열 개수", f"{n_cols}개")
with c3.container(border=True):
    st.metric("stroke가 1인 사람 수", f"{n_stroke:,}명")
with c4.container(border=True):
    st.metric("stroke가 1인 비율", f"{ratio:.2f}%")

# 2) 열 설명 표 (우리말 뜻만 직접 입력)
st.subheader("열 설명 표")
st.write("교재를 보고 **우리말 뜻** 칸을 직접 채워 보세요. 적은 내용은 자동으로 저장돼요.")

edited = st.data_editor(
    st.session_state["base_table"],
    key="column_table",
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻", help="교재를 보고 직접 적어 주세요."
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수"),
    },
)

current_meanings = {
    str(row["열 이름"]): ("" if pd.isna(row["우리말 뜻"]) else str(row["우리말 뜻"]))
    for _, row in edited.iterrows()
}
if current_meanings != st.session_state["saved_meanings"]:
    save_record(current_meanings, st.session_state["saved_source"])
    st.session_state["saved_meanings"] = current_meanings

# 3) 처음 다섯 줄
st.subheader("데이터 첫 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# 4) 데이터 출처
st.subheader("데이터 출처")
source_text = st.text_area(
    "출처",
    value=st.session_state["saved_source"],
    placeholder="교재에 나온 데이터 출처를 여기에 적어 주세요.",
    height=120,
    label_visibility="collapsed",
)
st.caption("글을 쓴 뒤 칸 밖을 클릭하면 저장돼요.")

if source_text != st.session_state["saved_source"]:
    save_record(st.session_state["saved_meanings"], source_text)
    st.session_state["saved_source"] = source_text
