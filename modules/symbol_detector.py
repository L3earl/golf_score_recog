"""
기호 검출 모듈 (싱글톤 패턴)
- case2 기호 검출 전용
- "-", "." 패턴 인식
"""

import cv2
import numpy as np

class SymbolDetector:
    """기호 검출 싱글톤 클래스"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SymbolDetector, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = SymbolDetector()
        return cls._instance
    
    def detect(self, image_path):
        """이미지에서 기호 검출"""
        img = cv2.imread(image_path)
        if img is None:
            return 4  # 기호 없음
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 흰색 픽셀 추출
        white_mask = gray > 200
        binary = np.zeros_like(gray)
        binary[white_mask] = 255
        
        # 모폴로지 연산
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # "-" 모양 판별
            if aspect_ratio > 2.0 and w > 5:
                return 5
            
            # "." 모양 판별
            elif 0.7 <= aspect_ratio <= 1.3 and area < 100:
                return 3
        
        return 4  # 기호 없음
    
    def detect_batch(self, image_paths):
        """여러 이미지에서 기호 검출"""
        results = []
        for image_path in image_paths:
            symbol_value = self.detect(image_path)
            results.append(symbol_value)
        return results
