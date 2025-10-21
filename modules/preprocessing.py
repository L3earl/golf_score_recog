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
from modules.simple_ocr_crop import process_case3_ocr_crop

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """이미지 전처리 클래스
    
    의도: 이미지 크롭과 정리를 순차적으로 수행하는 통합 처리 파이프라인 관리
    """
    
    def __init__(self, case, use_ocr_crop=False):
        """전처리기 초기화
        
        의도: 케이스별 설정을 로드하고 크롭/정리 모듈을 초기화
        
        Args:
            case: 처리 케이스 ('case1', 'case2', 'case3')
            use_ocr_crop: case3에서 OCR 크롭 사용 여부 (기본값: False)
        """
        self.case = case
        self.use_ocr_crop = use_ocr_crop
        
        # case3는 crop_coordinates.py를 사용하지 않음 (case1과 동일한 처리)
        if case == "case3":
            self.coordinates = CROP_COORDINATES.get("case1", [])
        else:
            self.coordinates = CROP_COORDINATES.get(case, [])
        
        # 싱글톤 인스턴스 사용
        self.cropper = ImageCropper.get_instance()
        self.cleaner = ImageCleaner.get_instance()
        logger.debug(f"ImagePreprocessor 초기화 완료 (케이스: {case}, OCR 크롭: {use_ocr_crop})")
        
        # 케이스별 폴더 경로 설정
        self.raw_crop_folder = get_case_folder(RAW_CROP_NUM_FOLDER, case)
        self.raw_clean_folder = get_case_folder(RAW_CLEAN_NUM_FOLDER, case)
    
    def process_all_images(self, target_files=None):
        """전체 이미지 전처리 파이프라인 실행
        
        의도: 크롭과 정리를 순차적으로 수행하여 OCR 준비된 이미지 생성
        
        Args:
            target_files: 처리할 파일명 리스트 (None이면 전체)
        
        Returns:
            전처리 성공 여부 또는 실패한 파일 리스트 (case3 OCR 크롭 실패 시)
        """
        try:
            logger.info(f"{self.case} 전처리 시작")
            
            # case3에서 OCR 크롭 사용
            if self.case == "case3" and self.use_ocr_crop:
                logger.info("OCR 크롭 실행 중...")
                success_files = process_case3_ocr_crop(target_files)
                
                if not success_files:
                    logger.error("OCR 크롭 실패")
                    return target_files  # 실패한 파일 리스트 반환
                
                # OCR 크롭 실패한 파일들 추적
                ocr_crop_failed_files = [f for f in target_files if f not in success_files]
                
                logger.info(f"OCR 크롭 성공: {len(success_files)}개 파일")
                if ocr_crop_failed_files:
                    logger.warning(f"OCR 크롭 실패: {len(ocr_crop_failed_files)}개 파일")
                    return ocr_crop_failed_files  # 실패한 파일들을 예외로 반환
                
                # 성공한 파일들만 계속 처리
                target_files = success_files
            
            # case1, case2는 원본 이미지에서 크롭
            elif self.case in ["case1", "case2"]:
                input_folder = RAW_IMG_FOLDER
                
                # 1단계: 이미지 크롭
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
