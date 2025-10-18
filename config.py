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
RAW_TABLE_CROP_FOLDER = os.path.join(DATA_FOLDER, "raw_table_crop")

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

# ==================== 후처리 설정 ====================
# 후처리 임계값
MIN_TOTAL_THRESHOLD = 30  # total1, total2 최소값 임계값

# ==================== OCR 모델 설정 ====================
# TrOCR 모델 설정
TROCR_MODEL_NAME = "microsoft/trocr-large-printed"

# ==================== 로깅 설정 ====================
import logging

# 로깅 레벨 설정
LOGGING_LEVEL = logging.INFO  # 운영: INFO, 개발: DEBUG

def setup_logging():
    """로깅 시스템 초기화
    
    의도: 프로젝트 전체의 로깅을 일관되게 설정
    운영(INFO): 주요 작업 흐름, 성공/실패 결과
    개발(DEBUG): 상세한 디버깅 정보
    """
    logging.basicConfig(
        level=LOGGING_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# 프로젝트 시작 시 로깅 설정
setup_logging()
# 기본 Claude 모델
DEFAULT_CLAUDE_MODEL = "claude-opus-4-1-20250805"

# Claude API 프롬프트
CLAUDE_GOLF_PROMPT = """
골프 스코어카드 이미지에서 다음 데이터를 정확히 추출해주세요:

1. **PAR 수**: 각 홀의 기준 타수 (필수, 최소 18홀 이상)
2. **플레이어 스코어**: 각 플레이어의 각 홀별 실제 타수
   - 플레이어1 스코어는 필수입니다
   - 플레이어2~4 스코어는 이미지에 있는 경우에만 추출해주세요

**중요 사항:**
- 모든 값은 정수로 정확히 인식해주세요
- 음수 값(-1, -2 등)도 정확히 인식해주세요
- 각 배열은 최소 18개 이상의 값을 포함해야 합니다
- 모든 배열(par, player1_score 등)은 같은 길이여야 합니다
- 이미지에 보이는 데이터만 추출하고 추측하지 마세요

제공된 tool을 사용하여 구조화된 JSON 형식으로 응답해주세요.
"""

# Claude API Tools 정의
CLAUDE_EXTRACT_SCORECARD_TOOL = {
    "name": "extract_golf_scorecard",
    "description": "골프 스코어카드에서 par 수와 플레이어들의 스코어를 추출합니다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "par": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 18,
                "maxItems": 18,
                "description": "각 홀의 par 수 (18개)"
            },
            "player1_score": {
                "type": "array", 
                "items": {"type": "integer"},
                "minItems": 18,
                "maxItems": 18,
                "description": "플레이어1의 각 홀 스코어 (18개)"
            },
            "player2_score": {
                "type": "array",
                "items": {"type": "integer"}, 
                "minItems": 18,
                "maxItems": 18,
                "description": "플레이어2의 각 홀 스코어 (18개)"
            },
            "player3_score": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 18, 
                "maxItems": 18,
                "description": "플레이어3의 각 홀 스코어 (18개)"
            },
            "player4_score": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 18,
                "maxItems": 18, 
                "description": "플레이어4의 각 홀 스코어 (18개)"
            }
        },
        "required": ["par", "player1_score"]
    }
}

# Claude API 파라미터
CLAUDE_MAX_TOKENS = 4000

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
