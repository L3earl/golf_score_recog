"""
이미지 전처리 모듈

의도: 이미지 크롭과 정리를 순차적으로 수행하는 통합 처리 파이프라인
- 이미지 크롭 (crop): 지정된 좌표로 개별 숫자/기호 추출
- 얼룩 제거 (clean): OCR 정확도 향상을 위한 이미지 정리
"""

import os
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_IMG_FOLDER, 
    RAW_CROP_NUM_FOLDER, 
    RAW_CLEAN_NUM_FOLDER, 
    get_case_folder
)
from crop_coordinates import CROP_COORDINATES
from modules.image_cropper import ImageCropper
from modules.image_cleaner import ImageCleaner

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """이미지 전처리 클래스
    
    의도: 이미지 크롭과 정리를 순차적으로 수행하는 통합 처리 파이프라인 관리
    """
    
    def __init__(self, case):
        """전처리기 초기화
        
        의도: 케이스별 설정을 로드하고 크롭/정리 모듈을 초기화
        
        Args:
            case: 처리 케이스 ('case1', 'case2', 'case3')
        """
        self.case = case
        
        # case3는 crop_coordinates.py를 사용하지 않음 (case1과 동일한 처리)
        if case == "case3":
            self.coordinates = CROP_COORDINATES.get("case1", [])
        else:
            self.coordinates = CROP_COORDINATES.get(case, [])
        
        # 싱글톤 인스턴스 사용
        self.cropper = ImageCropper.get_instance()
        self.cleaner = ImageCleaner.get_instance()
        logger.debug(f"ImagePreprocessor 초기화 완료 (케이스: {case})")
        
        # 케이스별 폴더 경로 설정
        self.raw_crop_folder = get_case_folder(RAW_CROP_NUM_FOLDER, case)
        self.raw_clean_folder = get_case_folder(RAW_CLEAN_NUM_FOLDER, case)
    
    def process_all_images(self, target_files=None):
        """전체 이미지 전처리 파이프라인 실행
        
        의도: 크롭과 정리를 순차적으로 수행하여 OCR 준비된 이미지 생성
        
        Args:
            target_files: 처리할 파일명 리스트 (None이면 전체)
        
        Returns:
            전처리 성공 여부
        """
        try:
            logger.info(f"{self.case} 전처리 시작")
            
            # case3는 OCR 크롭된 이미지에서 크롭, 나머지는 원본 이미지에서 크롭
            if self.case == "case1":
                input_folder = RAW_IMG_FOLDER
            
                # 1단계: 이미지 크롭 (case3는 이미 크롭된 이미지들을 다시 크롭)
                if not self.cropper.crop_all_images(input_folder, self.coordinates, self.raw_crop_folder, target_files):
                    logger.error(f"{self.case} 크롭 실패")
                    return False
            
            # 2단계: 이미지 정리
            if not self.cleaner.clean_folder(self.raw_crop_folder, self.raw_clean_folder, self.case):
                logger.error(f"{self.case} 이미지 정리 실패")
                return False
            
            logger.info(f"{self.case} 전처리 완료")
            return True
        except Exception as e:
            logger.error(f"{self.case} 전처리 중 오류 발생: {e}")
            return False
