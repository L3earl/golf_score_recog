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
    
    def process_all_images(self):
        """전체 이미지 전처리 파이프라인 실행"""
        # 1단계: 이미지 크롭
        if not self.cropper.crop_all_images(RAW_IMG_FOLDER, self.coordinates, self.raw_crop_folder):
            print(f"❌ {self.case} 크롭 실패")
            return False
        
        # 2단계: 이미지 정리
        if not self.cleaner.clean_folder(self.raw_crop_folder, self.raw_clean_folder, self.case):
            print(f"❌ {self.case} 이미지 정리 실패")
            return False
        
    def process_specific_folder(self, folder_name):
        """특정 폴더만 전처리"""
        try:
            # 원본 이미지에서 특정 폴더만 크롭
            raw_img_path = os.path.join(RAW_IMG_FOLDER, f"{folder_name}.png")
            if not os.path.exists(raw_img_path):
                print(f"  ❌ 원본 이미지 없음: {raw_img_path}")
                return False
            
            # 크롭된 이미지 저장할 폴더
            crop_folder = os.path.join(self.raw_crop_folder, folder_name)
            os.makedirs(crop_folder, exist_ok=True)
            
            # 특정 이미지만 크롭
            if not self.cropper.crop_image(raw_img_path, self.coordinates, self.raw_crop_folder):
                print(f"  ❌ {folder_name} 크롭 실패")
                return False
            
            # 특정 폴더만 정리
            if not self.cleaner.clean_folder(crop_folder, os.path.join(self.raw_clean_folder, folder_name), self.case):
                print(f"  ❌ {folder_name} 이미지 정리 실패")
                return False
            
            print(f"  ✓ {folder_name} 전처리 완료")
            return True
            
        except Exception as e:
            print(f"  ❌ {folder_name} 전처리 중 오류: {e}")
            return False
