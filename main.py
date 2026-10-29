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

# -----------------------------
# 열 설명표: 뜻을 직접 입력
# -----------------------------
st.subheader("📝 열 이름과 값 살펴보기")
st.write("‘우리말 뜻’ 칸은 교재를 참고해 직접 입력해 보세요. 입력한 내용은 JSON 파일로 저장됩니다.")

previous_meanings = saved_record.get("meanings", {})
if not isinstance(previous_meanings, dict):
    previous_meanings = {}

rows = []
for column in df.columns:
    values = df[column]
    rows.append({
        "열 이름": column,
        "우리말 뜻": str(previous_meanings.get(column, "")),
        "값의 종류": int(values.nunique(dropna=True)),
        "빈 값 개수": int(values.isna().sum()),
    })

meaning_df = pd.DataFrame(rows, columns=["열 이름", "우리말 뜻", "값의 종류", "빈 값 개수"])
edited_meaning_df = st.data_editor(
    meaning_df,
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고해 직접 입력하세요.",
            width="large",
        ),
        "값의 종류": st.column_config.NumberColumn("값의 종류"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수"),
    },
    key="meaning_editor",
)

st.subheader("🔎 데이터 처음 다섯 줄")
st.dataframe(df.head(5), hide_index=True, use_container_width=True)

# -----------------------------
# 데이터 출처 기록
# -----------------------------
st.subheader("📚 데이터 출처 기록")
st.write("교재에 안내된 데이터 출처를 아래에 입력하세요.")
source_text = st.text_area(
    "데이터 출처",
    value=str(saved_record.get("source", "")),
    height=100,
    placeholder="교재에 적힌 데이터 출처를 입력하세요.",
)

if st.button("💾 우리말 뜻과 출처 저장", type="primary"):
    meanings = {
        str(row["열 이름"]): str(row["우리말 뜻"])
        for _, row in edited_meaning_df.iterrows()
    }
    if save_record(meanings, source_text):
        st.success(f"기록을 저장했습니다: {RECORD_FILE.as_posix()}")
        st.caption("새로고침 후에도 저장된 기록을 다시 불러옵니다.")

st.caption("기록 파일 위치: 내_기록/뇌졸중_데이터_기록.json")
