"""
기호 검출 모듈

의도: 이미지에서 특정 기호나 패턴을 검출하여 위치 정보 제공
- OpenCV를 활용한 이미지 분석
- 템플릿 매칭 및 윤곽선 검출
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SymbolDetector:
    """기호 검출 싱글톤 클래스
    
    의도: 프로젝트 전체에서 단일 인스턴스로 기호 검출 작업 수행
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SymbolDetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """기호 검출기 초기화
        
        의도: 싱글톤 인스턴스 초기화
        """
        if not self._initialized:
            # 초기화 코드 (현재는 없음)
            self._initialized = True
            logger.debug("SymbolDetector 초기화 완료")
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 반환
        
        의도: 어디서든 동일한 인스턴스 접근 가능
        """
        if cls._instance is None:
            cls._instance = SymbolDetector()
        return cls._instance
    
    def detect(self, image_path):
        """이미지에서 기호 검출
        
        의도: 지정된 이미지에서 특정 기호나 패턴을 검출하여 숫자로 반환
        
        Args:
            image_path: 검출할 이미지 경로
        
        Returns:
            검출된 기호 번호 (3-5) 또는 None (실패 시)
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"이미지 로드 실패: {image_path}")
            return 4  # 기호 없음
        
        try:
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
                    logger.debug(f"기호 검출 완료: {image_path} -> '-' (5)")
                    return 5
                
                # "." 모양 판별
                elif 0.7 <= aspect_ratio <= 1.3 and area < 100:
                    logger.debug(f"기호 검출 완료: {image_path} -> '.' (3)")
                    return 3
            
            logger.debug(f"기호 검출 완료: {image_path} -> 기호 없음 (4)")
            return 4  # 기호 없음
        except Exception as e:
            logger.error(f"기호 검출 실패 ({image_path}): {e}")
            return 4
    
    def detect_batch(self, image_paths):
        """여러 이미지에서 기호 검출
        
        의도: 여러 이미지를 한 번에 처리하여 기호 검출 결과 리스트 반환
        
        Args:
            image_paths: 검출할 이미지 경로 리스트
        
        Returns:
            검출 결과 리스트
        """
        results = []
        try:
            logger.info(f"배치 기호 검출 시작: {len(image_paths)}개 이미지")
            
            for image_path in image_paths:
                symbol_value = self.detect(image_path)
                results.append(symbol_value)
            
            logger.info(f"배치 기호 검출 완료: {len(results)}개 처리")
            return results
        except Exception as e:
            logger.error(f"배치 기호 검출 실패: {e}")
            return []
