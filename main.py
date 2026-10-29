import json
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/iibina/data/refs/heads/main/stroke.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "뇌졸중_데이터_기록.json"

st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고, 각 열의 의미와 출처를 교재를 참고해 직접 기록하는 실습 공간입니다.")

# -----------------------------
# 기록 파일 읽기 / 저장하기
# -----------------------------
def load_record():
    """저장된 뜻과 출처를 불러옵니다. 파일이 없거나 내용이 잘못되었으면 빈 기록을 사용합니다."""
    try:
        if RECORD_FILE.exists():
            with RECORD_FILE.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError):
        st.warning("기존 기록 파일을 읽지 못해 새 기록으로 시작합니다.")
    return {"meanings": {}, "source": ""}


def save_record(meanings, source):
    """입력한 내용을 UTF-8 JSON 파일로 저장합니다."""
    try:
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        data = {"meanings": meanings, "source": source}
        with RECORD_FILE.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        return True
    except OSError as error:
        st.error(
            "기록 파일을 저장하지 못했습니다. 배포 환경에서는 파일 시스템이 "
            "초기화될 수 있으므로 저장소의 쓰기 권한도 확인해 주세요. "
            f"(오류: {error})"
        )
        return False


saved_record = load_record()

# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")

try:
    df = load_data(DATA_URL)
except Exception as error:
    st.error("데이터를 인터넷에서 불러오지 못했습니다.")
    st.write("데이터 주소:", DATA_URL)
    st.write("오류 내용:", str(error))
    st.stop()

if "stroke" not in df.columns:
    st.error("데이터에 'stroke' 열이 없습니다. 데이터 주소와 열 이름을 확인해 주세요.")
    st.stop()

stroke_numeric = pd.to_numeric(df["stroke"], errors="coerce")
stroke_count = int((stroke_numeric == 1).sum())
stroke_ratio = (stroke_count / len(df) * 100) if len(df) else 0.0

# -----------------------------
# 요약 카드
# -----------------------------
st.subheader("📊 데이터 한눈에 보기")
col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 사람 수", f"{len(df):,}명")
col2.metric("열 개수", f"{len(df.columns):,}개")
col3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
col4.metric("stroke가 1인 비율", f"{stroke_ratio:.2f}%")

st.info(
    "이 데이터는 사람별 건강 관련 정보와 뇌졸중(stroke) 여부를 담고 있습니다. "
    "이 화면은 데이터 탐색 실습용이며, 개인의 뇌졸중 발생 여부를 진단하거나 "
    "의료적 판단을 내리는 도구가 아닙니다."
)
