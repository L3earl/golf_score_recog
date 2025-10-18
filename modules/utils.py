"""
공통 유틸리티 모듈

의도: 프로젝트 전체에서 재사용되는 기본 기능 제공
- 파일/디렉토리 처리
- 데이터 변환 유틸리티
"""

import os
import re
import logging
from typing import List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ==================== 파일/디렉토리 유틸리티 ====================
def get_image_files(folder: str, extensions: List[str]) -> List[str]:
    """폴더에서 이미지 파일 목록을 가져옵니다.
    
    의도: 지정된 확장자의 이미지 파일만 필터링하여 정렬된 리스트 반환
    
    Args:
        folder: 검색할 폴더 경로
        extensions: 허용할 확장자 리스트 (예: ['.png', '.jpg'])
    
    Returns:
        이미지 파일 경로 리스트 (정렬됨)
    
    Example:
        >>> get_image_files('data/images', ['.png', '.jpg'])
        ['data/images/img1.png', 'data/images/img2.jpg']
    """
    if not os.path.exists(folder):
        logger.debug(f"폴더가 존재하지 않음: {folder}")
        return []
    
    try:
        image_files = []
        for file in os.listdir(folder):
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(folder, file))
        
        logger.debug(f"{folder}에서 {len(image_files)}개 이미지 파일 발견")
        return sorted(image_files)
    except Exception as e:
        logger.error(f"이미지 파일 검색 실패 ({folder}): {e}")
        return []


def ensure_directory(path: str) -> None:
    """디렉토리가 없으면 생성합니다.
    
    의도: 안전하게 디렉토리를 생성 (이미 존재해도 에러 없음)
    
    Args:
        path: 생성할 디렉토리 경로
    
    Raises:
        Exception: 디렉토리 생성 실패 시
    """
    try:
        os.makedirs(path, exist_ok=True)
        logger.debug(f"디렉토리 확인/생성: {path}")
    except Exception as e:
        logger.error(f"디렉토리 생성 실패 ({path}): {e}")
        raise


# ==================== 데이터 변환 유틸리티 ====================
def is_numeric(value: Any) -> bool:
    """값이 숫자인지 확인합니다.
    
    의도: 문자열, 숫자 타입 모두 지원하는 범용 숫자 검증
    
    Args:
        value: 검증할 값
    
    Returns:
        숫자 여부 (정수, 실수, 음수 모두 포함)
    
    Example:
        >>> is_numeric("123")
        True
        >>> is_numeric("-45.6")
        True
        >>> is_numeric("abc")
        False
    """
    if pd.isna(value) or value is None:
        return False
    
    if isinstance(value, str):
        pattern = r'^-?\d+(\.\d+)?$'
        return bool(re.match(pattern, value.strip()))
    
    return isinstance(value, (int, float, np.integer, np.floating))


def convert_to_numeric(value: Any) -> Optional[float]:
    """값을 숫자로 변환합니다.
    
    의도: 다양한 타입의 값을 안전하게 숫자로 변환
    
    Args:
        value: 변환할 값
    
    Returns:
        변환된 숫자 또는 None (변환 실패 시)
    
    Example:
        >>> convert_to_numeric("123.45")
        123.45
        >>> convert_to_numeric("abc")
        None
    """
    if pd.isna(value) or value is None:
        return None
    
    try:
        if isinstance(value, str):
            return float(value.strip())
        return float(value)
    except (ValueError, TypeError) as e:
        logger.debug(f"숫자 변환 실패 ({value}): {e}")
        return None