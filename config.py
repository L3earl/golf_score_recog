"""
골프 스코어카드 인식 프로젝트 설정 파일
모든 모듈에서 공통으로 사용하는 설정값들을 중앙 관리
"""

import os

# ==================== 기본 경로 설정 ====================
# 프로젝트 루트 디렉토리
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 데이터 폴더 경로 (data/ 폴더 내부로 변경)
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")

# 입력/출력 폴더 경로들
RAW_IMG_FOLDER = os.path.join(DATA_FOLDER, "raw_img")
RAW_CROP_NUM_FOLDER = os.path.join(DATA_FOLDER, "raw_crop")
RAW_CLEAN_NUM_FOLDER = os.path.join(DATA_FOLDER, "raw_clean")
RESULT_CONVERT_NUM_FOLDER = os.path.join(DATA_FOLDER, "result_convert")
RESULT_CLAUDE_FOLDER = os.path.join(DATA_FOLDER, "result_claude")
RAW_TEMPLATE_CROP_FOLDER = os.path.join(DATA_FOLDER, "raw_template_crop")

# ==================== 케이스 설정 ====================
# 케이스 목록
CASES = ["case1", "case2", "case3"]

# 케이스별 폴더 경로 함수
def get_case_folder(base_folder, case):
    """케이스별 폴더 경로를 반환합니다."""
    return os.path.join(base_folder, case)

# ==================== Case2 설정 ====================
# case2 숫자 이미지 범위
CASE2_NUM_RANGE = (0, 21)  # 0~20번

# case2 기호 이미지 범위  
CASE2_SIGN_RANGE = (21, 39)  # 21~38번

# case2 기호 매핑
CASE2_SYMBOL_MAP = {3: "-", 4: "", 5: "."}

# ==================== 이미지 정리 설정 ====================
# 색상 필터링 파라미터
BLACK_THRESHOLD = 50  # 검은색 임계값 (0~255, 낮을수록 더 어두운 색만 검은색으로 인식)

# 처리 제외할 이미지 번호들
EXCLUDE_INDICES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8,  # 0~8번
    45, 46, 47, 48, 49, 50, 51, 52, 53,  # 45~53번
    90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104  # 90~104번
]

# ==================== OCR 모델 설정 ====================
# TrOCR 모델 설정
TROCR_MODEL_NAME = "microsoft/trocr-large-printed"

# ==================== Claude API 설정 ====================
# 기본 Claude 모델
DEFAULT_CLAUDE_MODEL = "claude-opus-4-1-20250805"

# Claude API 프롬프트
# CLAUDE_PROMPT = """
# 첨부된 골프 스코어카드 이미지에서 숫자를 테이블 형태로 뽑아줘
# 결과는 "scores"라는 단일 키를 가진 JSON 객체 형식으로만 출력해줘.
# 모든 값은 숫자로 정확하게 인식하고, 특히 음수 값에 유의해줘.
# JSON 객체 앞뒤로 어떤 설명 텍스트도 절대 포함하지 마.
# """

# CLAUDE_PROMPT = """
# 첨부된 골프 스코어카드 이미지에서 표의 내용을 추출해 줘.

# [출력 형식]
# 1. CSV(Comma-Separated Values) 형식으로만 출력해.
# 2. 첫 번째 줄은 이미지에 보이는 그대로 컬럼명(헤더)을 포함해야 해.
# 3. 각 행은 줄바꿈으로 구분하고, 각 셀의 값은 쉼표(,)로 구분해야 해.

# [데이터 처리 규칙]
# 1. 모든 값은 숫자로 정확하게 인식하고, 특히 '-1'과 같은 음수 값에 유의해 줘.

# [매우 중요한 규칙]
# - CSV 데이터 외에 다른 어떤 설명이나 코드 블록 마크(```)도 절대 추가하지 마.
# """

# CLAUDE_PROMPT = """
# 첨부된 골프 스코어카드 이미지에서 **'PAR' 기준 타수 데이터**와 **플레이어 스코어 데이터**를 추출해 줘.

# [출력 형식]
# 1. CSV(Comma-Separated Values) 형식으로만 출력해.
# 2. 각 행은 줄바꿈으로 구분하고, 각 셀의 값은 쉼표(,)로 구분해야 해.

# [데이터 처리 규칙]
# 1. **'HOLE' 행** (또는 'Hoyo', '홀', '1, 2, 3...')처럼 컬럼 번호를 나열하는 행은 **결과에 절대 포함하지 마.**

# 2. **'PAR' 행** (기준 타수)과 **플레이어 행** (이름/ID + 스코어)은 **이미지에 보이는 순서 그대로** 모두 추출해 줘.
#    - (중요: 테이블이 전/후반으로 나뉘어 있어도 절대 병합하지 말고, 보이는 그대로 여러 줄로 출력해 줘.)

# 3. 모든 스코어 값은 숫자로 정확하게 인식하고, 아이콘(🦋, 🌸)이 붙은 값도 숫자(예: -1, +2)로 변환해 줘. (만약 -1, -2가 아니라 3, 6이라면 그 숫자를 그대로 가져와 줘)

# [매우 중요한 규칙]
# - CSV 데이터 외에 다른 어떤 설명이나 코드 블록 마크(```)도 절대 추가하지 마.
# """


# CLAUDE_PROMPT = """
# 너는 프로 골퍼야.
# = 첨부된 이미지는 골프 스코어카드야
# - 나는 이미지에서 par개수, 플레이어들의 스코어만 필요해. 데이터로 추출해줘, 그 외 데이터는 절대 추가하지 마
# - 추출된 데이터를 1홀~18홀, 전반전의 합, 후반전의 합, 총압 순서로 만들어서 줘

# [출력 형식]
# 1. CSV(Comma-Separated Values) 형식으로만 출력해.
# 2. 각 행은 줄바꿈으로 구분하고, 각 셀의 값은 쉼표(,)로 구분해야 해.

# [데이터 처리 규칙]
# 1. 요청한 값은 숫자로 정확하게 인식하고, 특히 '-1'과 같은 음수 값에 유의해 줘.

# [매우 중요한 규칙]
# - CSV 데이터 외에 다른 어떤 설명이나 코드 블록 마크(```)도 절대 추가하지 마.
# """

CLAUDE_PROMPT = """
You are a professional golfer.
= The attached image is a golf scorecard.
- I only need the PAR values and the players' scores from the image. Extract them as data. Do not add any other data whatsoever.
- Create and provide the extracted data in the following order: Holes 1-18, Front 9 Total, Back 9 Total, Grand Total.

[Output Format]
1. Output only in CSV (Comma-Separated Values) format.
2. Separate each row with a newline, and separate each cell's value with a comma (,).

[Data Processing Rules]
1. Accurately recognize the requested values as numbers, and pay special attention to negative values like '-1'.

[Very Important Rule]
- Absolutely do not add any other explanations or code block marks (```) besides the CSV data.
"""

# ==================== 골프 스코어카드 데이터 구조 설정 ====================
# 컬럼명 설정
SCORECARD_COLUMNS = [f"{i}홀" for i in range(1, 19)] + ["total1", "total2", "sum"]

# 플레이어 수
NUM_PLAYERS = 5

# 이미지 개수
NUM_IMAGES = 123  # 0~122

# ==================== 파일 확장자 설정 ====================
# 지원하는 이미지 확장자
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

# ==================== 출력 설정 ====================
# CSV 인코딩
CSV_ENCODING = 'utf-8-sig'
