"""
이미지 전처리 모듈
- 이미지 크롭 (crop)
- 얼룩 제거 (clean)
두 단계를 순차적으로 수행하는 통합 모듈
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_IMG_FOLDER, 
    RAW_CROP_NUM_FOLDER, 
    RAW_CLEAN_NUM_FOLDER, 
    RAW_TEMPLATE_CROP_FOLDER,
    get_case_folder
)
from crop_coordinates import CROP_COORDINATES
from modules.image_cropper import ImageCropper
from modules.image_cleaner import ImageCleaner

class ImagePreprocessor:
    """이미지 전처리 클래스"""
    
    def __init__(self, case):
        """전처리기 초기화"""
        self.case = case
        self.coordinates = CROP_COORDINATES.get(case, [])
        
        # 싱글톤 인스턴스 사용
        self.cropper = ImageCropper.get_instance()
        self.cleaner = ImageCleaner.get_instance()
        
        # 케이스별 폴더 경로 설정
        self.raw_crop_folder = get_case_folder(RAW_CROP_NUM_FOLDER, case)
        self.raw_clean_folder = get_case_folder(RAW_CLEAN_NUM_FOLDER, case)
    
    def process_all_images(self, target_files=None):
        """
        전체 이미지 전처리 파이프라인 실행
        target_files: None이면 전체, 리스트면 해당 파일명만 처리
        """
        # case3는 템플릿 매칭된 이미지에서 크롭, 나머지는 원본 이미지에서 크롭
        if self.case == "case3":
            input_folder = RAW_TEMPLATE_CROP_FOLDER
        else:
            input_folder = RAW_IMG_FOLDER
        
        # 1단계: 이미지 크롭
        if not self.cropper.crop_all_images(input_folder, self.coordinates, self.raw_crop_folder, target_files):
            print(f"❌ {self.case} 크롭 실패")
            return False
        
        # 2단계: 이미지 정리
        if not self.cleaner.clean_folder(self.raw_crop_folder, self.raw_clean_folder, self.case):
            print(f"❌ {self.case} 이미지 정리 실패")
            return False
        
        print(f"  ✓ {self.case} 정리 완료")
        return True
