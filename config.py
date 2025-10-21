"""
골프 스코어카드 인식 프로젝트 설정 파일

의도:
    - 모든 모듈에서 공통으로 사용하는 설정값들을 중앙에서 일관되게 관리합니다.
    - 기능(동작)을 변경하지 않기 위해 기존 기본값을 그대로 유지하고,
      필요 시 환경변수(.env 포함)로 덮어쓸 수 있도록 합니다.

환경변수 규칙(기본값 유지):
    - 값이 제공되지 않으면 기존 하드코딩 값이 그대로 사용됩니다.
    - 제공되면 안전한 파싱을 거쳐 해당 설정에만 반영됩니다.

Google-style Docstring 규약을 따라 모듈의 의도를 명확히 기술합니다.
"""

import os
from typing import Optional

# ==================== 기본 경로 설정 ====================
# 프로젝트 루트 디렉토리
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 데이터 폴더 경로 (data/ 폴더 내부로 변경)
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")

# 입력/출력 폴더 경로들
RAW_IMG_FOLDER = os.path.join(DATA_FOLDER, "raw_img")
RAW_IMG_UPSCALE_FOLDER = os.path.join(DATA_FOLDER, "raw_img_upscale")  # 업스케일링된 이미지 저장 폴더
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

# ==================== 환경변수 로딩 유틸 ====================
def _get_env_str(key: str, default: str) -> str:
    """문자열 환경변수 로더.

    의도:
        - 기능 불변을 위해 기본값을 유지하되, 환경변수가 있으면 덮어씁니다.

    Args:
        key: 환경변수 키
        default: 기본 문자열 값

    Returns:
        최종 문자열 값
    """
    return os.getenv(key, default)


def _get_env_int(key: str, default: int) -> int:
    """정수 환경변수 로더.

    Args:
        key: 환경변수 키
        default: 기본 정수 값

    Returns:
        최종 정수 값 (파싱 실패 시 기본값)
    """
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    """실수 환경변수 로더.

    Args:
        key: 환경변수 키
        default: 기본 실수 값

    Returns:
        최종 실수 값 (파싱 실패 시 기본값)
    """
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    """불리언 환경변수 로더.

    Args:
        key: 환경변수 키
        default: 기본 불리언 값

    Returns:
        최종 불리언 값 (파싱 실패 시 기본값)
    """
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in {"1", "true", "t", "yes", "y"}


# ==================== Case2 설정 ====================
# case2 숫자 이미지 범위
CASE2_NUM_RANGE = (0, 21)  # 0~20번

# case2 기호 이미지 범위  
CASE2_SIGN_RANGE = (21, 39)  # 21~38번

# case2 기호 매핑
CASE2_SYMBOL_MAP = {3: "-", 4: "", 5: "."}

# ==================== 이미지 정리 설정 ====================
# 색상 필터링 파라미터
BLACK_THRESHOLD = _get_env_int("BLACK_THRESHOLD", 50)  # 0~255

# 처리 제외할 이미지 번호들
EXCLUDE_INDICES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8,  # 0~8번
    45, 46, 47, 48, 49, 50, 51, 52, 53,  # 45~53번
    90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104  # 90~104번
]

# ==================== 후처리 설정 ====================
# 후처리 임계값
MIN_TOTAL_THRESHOLD = _get_env_int("MIN_TOTAL_THRESHOLD", 30)  # total1, total2 최소값 임계값

# ==================== OCR 모델 설정 ====================
# TrOCR 모델 설정
TROCR_MODEL_NAME = _get_env_str("TROCR_MODEL_NAME", "microsoft/trocr-large-printed")

# ==================== 로깅 설정 ====================
import logging

# 로깅 레벨 설정
_LOG_LEVEL_NAME = _get_env_str("LOGGING_LEVEL", "INFO").upper()
LOGGING_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)  # 운영: INFO, 개발: DEBUG

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
# 기본 Claude 모델 (환경변수로 덮어쓰기 가능)
DEFAULT_CLAUDE_MODEL = _get_env_str("DEFAULT_CLAUDE_MODEL", "claude-opus-4-1-20250805")

# Claude API 개선된 프롬프트
CLAUDE_GOLF_PROMPT = """당신은 골프 스코어카드 분석 전문가입니다. 이미지에서 각 홀(1-18)의 'PAR' 정보와 플레이어의 'SCORE' 정보만을 추출해 주세요.

[필수 준수 규칙]

'PAR' 행과 'SCORE' (또는 '점수', 'Rnd' 등) 행만 추출합니다.

무시할 행: 'YARDS'(거리), 'PUTT'(퍼트수), 'Status', 'Points', '센서영상' 등 스코어와 직접 관련 없는 숫자 행은 반드시 무시하세요.

무시할 열: 'OUT', 'IN', 'TOTAL', 'T', '합계' 등 요약/합계 열은 반드시 무시하세요. 오직 1번부터 18번 홀까지의 데이터만 순서대로 추출합니다.

레이아웃 처리: 스코어카드가 전반(1-9)과 후반(10-18)으로 나뉘어 있어도, 항상 1번부터 18번 홀까지 순서대로 결합하여 제공해야 합니다.

스코어 유형: 스코어는 4, 5, 3처럼 절대 타수일 수도 있고, -1, 0, 1처럼 PAR 기준 상대 타수일 수도 있습니다. 테이블에 보이는 그대로 추출해 주세요.

다중 플레이어: 여러 명의 플레이어가 있다면 모두 감지하여 각각의 스코어를 추출합니다.

플레이어 이름: 이름이 명확히 보이면 그대로 사용하고, 이름이 없으면 'Player 1', 'Player 2' 등으로 지정해 주세요.

제공된 tool을 사용하여 구조화된 JSON 형식으로 응답해 주세요."""

# Claude API 유연한 Tool 스키마
CLAUDE_EXTRACT_SCORECARD_TOOL = {
    "name": "extract_golf_scorecard",
    "description": "골프 스코어카드에서 PAR 및 모든 플레이어의 스코어를 추출합니다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "par": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "홀 1~18의 PAR 값 배열 (총 18개)"
            },
            "players": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "플레이어 이름 (e.g., 'Tiger Woods', '신재운'). 이름이 없으면 'Player 1' 등으로 지정."
                        },
                        "scores": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "해당 플레이어의 홀 1~18 스코어 배열 (총 18개). 테이블에 보이는 값 그대로(절대 또는 상대 스코어)."
                        }
                    },
                    "required": ["name", "scores"]
                },
                "description": "스코어카드에 있는 모든 플레이어의 목록"
            }
        },
        "required": ["par", "players"]
    }
}

# Claude API 파라미터
CLAUDE_MAX_TOKENS = _get_env_int("CLAUDE_MAX_TOKENS", 4000)

# ==================== 골프 스코어카드 데이터 구조 설정 ====================
# 컬럼명 설정
SCORECARD_COLUMNS = [f"{i}홀" for i in range(1, 19)] + ["total1", "total2", "sum"]

# 플레이어 수
NUM_PLAYERS = _get_env_int("NUM_PLAYERS", 5)

# 이미지 개수
NUM_IMAGES = _get_env_int("NUM_IMAGES", 123)  # 0~122

# ==================== 파일 확장자 설정 ====================
# 지원하는 이미지 확장자
IMAGE_EXTENSIONS = [
    ext.strip() for ext in _get_env_str(
        "IMAGE_EXTENSIONS",
        ".png,.jpg,.jpeg,.bmp,.tiff",
    ).split(",") if ext.strip()
]

# ==================== 출력 설정 ====================
# CSV 인코딩
CSV_ENCODING = _get_env_str('CSV_ENCODING', 'utf-8-sig')

# ==================== Super Resolution 설정 ====================
# EDSR 모델 폴더
MODELS_FOLDER = os.path.join(PROJECT_ROOT, "models")

# EDSR 모델 다운로드 URL
EDSR_MODEL_URL = _get_env_str(
    "EDSR_MODEL_URL",
    "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
)

# 업스케일링 후 이미지 최대 크기
UPSCALE_MAX_SIZE = _get_env_int("UPSCALE_MAX_SIZE", 2000)

# ==================== Claude API 개선 설정 ====================
# API 재시도 횟수
CLAUDE_MAX_RETRIES = _get_env_int("CLAUDE_MAX_RETRIES", 3)

# 이미지 압축 최대 크기 (MB)
IMAGE_MAX_SIZE_MB = _get_env_float("IMAGE_MAX_SIZE_MB", 4.5)

# JPEG 압축 초기 품질
JPEG_INITIAL_QUALITY = _get_env_int("JPEG_INITIAL_QUALITY", 95)

# JPEG 압축 최소 품질
JPEG_MIN_QUALITY = _get_env_int("JPEG_MIN_QUALITY", 30)