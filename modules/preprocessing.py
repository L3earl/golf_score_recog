"""
이미지 전처리 모듈
- 이미지 크롭 (crop)
- 얼룩 제거 (clean)
두 단계를 순차적으로 수행하는 통합 모듈
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CROP_COORDINATES_FILE, 
    RAW_IMG_FOLDER, 
    RAW_CROP_NUM_FOLDER, 
    RAW_CLEAN_NUM_FOLDER, 
    BLACK_THRESHOLD, 
    EXCLUDE_INDICES,
    IMAGE_EXTENSIONS
)

class ImagePreprocessor:
    """이미지 전처리 클래스"""
    
    def __init__(self):
        """전처리기 초기화"""
        self.coordinates = self._load_crop_coordinates()
        self.black_threshold = BLACK_THRESHOLD
        self.exclude_indices = EXCLUDE_INDICES
        
    def _load_crop_coordinates(self):
        """좌표 정보를 JSON 파일에서 불러오기"""
        try:
            with open(CROP_COORDINATES_FILE, 'r', encoding='utf-8') as f:
                coordinates = json.load(f)
            return coordinates
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ 좌표 파일 로드 실패: {e}")
            return []
    
    def _should_process_image(self, image_name):
        """이미지 번호가 처리 대상인지 확인"""
        try:
            # 파일명에서 번호 추출 (예: "0.png" -> 0)
            image_number = int(image_name.split('.')[0])
            return image_number not in self.exclude_indices
        except:
            return False
    
    def _remove_non_black_colors(self, image_path):
        """검은색 외의 색상 제거 (얼룩 제거)"""
        # 이미지 읽기
        img = cv2.imread(image_path)
        
        # 그레이스케일 변환
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 검은색 마스크 생성
        black_mask = gray < self.black_threshold
        
        # 검은색이 아닌 픽셀을 흰색으로 변경
        cleaned = gray.copy()
        cleaned[~black_mask] = 255  # 검은색이 아닌 픽셀을 흰색으로
        
        return cleaned
    
    def crop_image(self, image_path, output_dir):
        """단일 이미지 크롭"""
        if not self.coordinates:
            return False
            
        try:
            image = Image.open(image_path)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            img_output_dir = os.path.join(output_dir, image_name)
            os.makedirs(img_output_dir, exist_ok=True)
            
            for i, (left, top, right, bottom) in enumerate(self.coordinates):
                try:
                    cropped = image.crop((left, top, right, bottom))
                    output_path = os.path.join(img_output_dir, f"{i}.png")
                    cropped.save(output_path)
                except:
                    continue
            
            return True
        except:
            return False
    
    def clean_image(self, input_folder, output_folder):
        """이미지 정리 (얼룩 제거)"""
        if not os.path.exists(input_folder):
            return False
            
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for folder_name in os.listdir(input_folder):
            folder_path = os.path.join(input_folder, folder_name)
            
            if os.path.isdir(folder_path):
                output_folder_path = os.path.join(output_folder, folder_name)
                os.makedirs(output_folder_path, exist_ok=True)
                
                image_files = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(tuple(IMAGE_EXTENSIONS))]
                
                for image_file in image_files:
                    try:
                        input_image_path = os.path.join(folder_path, image_file)
                        output_image_path = os.path.join(output_folder_path, image_file)
                        
                        if self._should_process_image(image_file):
                            cleaned_image = self._remove_non_black_colors(input_image_path)
                            cv2.imwrite(output_image_path, cleaned_image)
                        else:
                            original_img = cv2.imread(input_image_path)
                            cv2.imwrite(output_image_path, original_img)
                    except:
                        continue
        
        return True
    
    def process_all_images(self):
        """전체 이미지 전처리 파이프라인 실행"""
        # 1단계: 이미지 크롭
        if not os.path.exists(RAW_IMG_FOLDER):
            print(f"❌ 입력 폴더 없음: {RAW_IMG_FOLDER}")
            return False
            
        if not os.path.exists(RAW_CROP_NUM_FOLDER):
            os.makedirs(RAW_CROP_NUM_FOLDER)
        
        image_files = [f for f in os.listdir(RAW_IMG_FOLDER) 
                      if f.lower().endswith(tuple(IMAGE_EXTENSIONS))]
        
        if not image_files:
            print(f"❌ 이미지 파일 없음: {RAW_IMG_FOLDER}")
            return False
        
        for filename in image_files:
            img_path = os.path.join(RAW_IMG_FOLDER, filename)
            self.crop_image(img_path, RAW_CROP_NUM_FOLDER)
        
        print(f"  ✓ 크롭: {len(image_files)}개")
        
        # 2단계: 이미지 정리
        if not self.clean_image(RAW_CROP_NUM_FOLDER, RAW_CLEAN_NUM_FOLDER):
            print("❌ 이미지 정리 실패")
            return False
        
        print(f"  ✓ 정리 완료")
        return True
