"""
이미지 정리 모듈 (싱글톤 패턴)
- 케이스별 이미지 정리 로직
- case1: 검은색 외 색상 제거
- case2: 번호별 다른 처리 (normal/symbol mode)
"""

import os
import cv2
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BLACK_THRESHOLD, EXCLUDE_INDICES, IMAGE_EXTENSIONS

class ImageCleaner:
    """이미지 정리 싱글톤 클래스"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageCleaner, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.black_threshold = BLACK_THRESHOLD
            self.exclude_indices = EXCLUDE_INDICES
            self._initialized = True
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = ImageCleaner()
        return cls._instance
    
    def _should_process_image(self, image_name):
        """이미지 번호가 처리 대상인지 확인"""
        try:
            image_number = int(image_name.split('.')[0])
            return image_number not in self.exclude_indices
        except:
            return False
    
    def clean_case1(self, image_path):
        """case1 이미지 정리: 검은색 외 색상 제거"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # 그레이스케일 변환
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 검은색 마스크 생성
        black_mask = gray < self.black_threshold
        
        # 검은색이 아닌 픽셀을 흰색으로 변경
        cleaned = gray.copy()
        cleaned[~black_mask] = 255
        
        return cleaned
    
    def clean_case2(self, image_path, image_index):
        """case2 이미지 정리: 번호별 다른 처리"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_index >= 21:
            # 21~38번: 기호 추출 모드
            return self._extract_symbols(gray)
        elif image_index in [9, 19]:
            # 9, 19번: 배경만 검은색
            return self._process_background_mode(gray)
        else:
            # 0~8, 10~18, 20번: 전경 검은색, 배경 흰색
            return self._process_normal_mode(gray)
    
    def _extract_symbols(self, gray):
        """기호 추출 모드"""
        # 흰색 픽셀 추출 (밝은 픽셀)
        white_mask = gray > 200
        
        # 이진화
        binary = np.zeros_like(gray)
        binary[white_mask] = 255
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 결과 이미지 (검은색 배경)
        result = np.zeros_like(gray)
        
        for contour in contours:
            # 면적이 너무 작으면 무시
            area = cv2.contourArea(contour)
            if area < 10:
                continue
                
            # 경계 사각형
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # "-" 모양 판별 (가로가 세로보다 훨씬 긴 경우)
            if aspect_ratio > 2.0 and w > 5:
                cv2.fillPoly(result, [contour], 255)
            
            # "." 모양 판별 (원형에 가까운 경우)
            elif 0.7 <= aspect_ratio <= 1.3 and area < 100:
                cv2.fillPoly(result, [contour], 255)
        
        return result
    
    def _process_normal_mode(self, gray):
        """전경 검은색, 배경 흰색 모드"""
        # Otsu 임계값으로 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 전경을 검은색으로, 배경을 흰색으로
        result = np.ones_like(gray) * 255
        result[binary == 255] = 0
        
        return result
    
    def _process_background_mode(self, gray):
        """배경만 검은색 모드"""
        # Otsu 임계값으로 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 배경만 검은색으로
        result = gray.copy()
        result[binary == 0] = 0
        
        return result
    
    def clean_folder(self, input_folder, output_folder, case):
        """폴더 내 모든 이미지 정리"""
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
                            # 이미지 번호 추출
                            image_index = int(image_file.split('.')[0])
                            
                            if case == "case1":
                                cleaned_image = self.clean_case1(input_image_path)
                            elif case == "case2":
                                cleaned_image = self.clean_case2(input_image_path, image_index)
                            else:
                                raise ValueError(f"Invalid case: {case}")
                            
                            if cleaned_image is not None:
                                cv2.imwrite(output_image_path, cleaned_image)
                        else:
                            original_img = cv2.imread(input_image_path)
                            cv2.imwrite(output_image_path, original_img)
                    except:
                        continue
        
        return True
