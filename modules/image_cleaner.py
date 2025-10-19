"""
이미지 정리 모듈

의도: 케이스별로 이미지를 정리하여 OCR 정확도 향상
- case1: 검은색 외 색상 제거
- case2: 번호별 다른 처리 (숫자/기호 모드)
- case3: case1과 동일
"""

import os
import cv2
import numpy as np
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils import ensure_directory
from config import BLACK_THRESHOLD, EXCLUDE_INDICES, IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

class ImageCleaner:
    """이미지 정리 싱글톤 클래스
    
    의도: 프로젝트 전체에서 단일 인스턴스로 이미지 정리 작업 수행
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageCleaner, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """이미지 정리기 초기화
        
        의도: 설정값을 config에서 로드하여 인스턴스 생성
        """
        if not self._initialized:
            self.black_threshold = BLACK_THRESHOLD
            self.exclude_indices = EXCLUDE_INDICES
            self._initialized = True
            logger.debug("ImageCleaner 초기화 완료")
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 반환
        
        의도: 어디서든 동일한 인스턴스 접근 가능
        """
        if cls._instance is None:
            cls._instance = ImageCleaner()
        return cls._instance
    
    def _should_process_image(self, image_name, case):
        """이미지 번호가 처리 대상인지 확인
        
        의도: case1과 case3에서만 특정 번호의 이미지를 제외하여 처리
        
        Args:
            image_name: 이미지 파일명
            case: 처리 케이스 ('case1', 'case2', 'case3')
        
        Returns:
            처리 여부
        """
        try:
            # case3는 그룹명 형식 (between_group_1.png, after_group_1.png)
            if case == "case3":
                # case3는 모든 그룹 이미지를 처리
                return True
            
            # case1, case2는 숫자 형식 (0.png, 1.png, ...)
            image_number = int(image_name.split('.')[0])
            
            if case == "case1":
                should_process = image_number not in self.exclude_indices
                if not should_process:
                    logger.debug(f"{case}에서 제외된 이미지: {image_name}")
                return should_process
            else:
                # case2 모든 이미지 처리
                return True
        except Exception as e:
            logger.error(f"이미지 번호 파싱 실패 ({image_name}): {e}")
            return False
    
    def clean_case1(self, image_path):
        """case1 이미지 정리
        
        의도: 검은색 외 모든 색상을 제거하여 OCR 정확도 향상
        
        Args:
            image_path: 처리할 이미지 경로
        
        Returns:
            정리된 이미지 또는 None (실패 시)
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"이미지 로드 실패: {image_path}")
            return None
        
        try:
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
            
            logger.debug(f"case1 이미지 정리 완료: {image_path}")
            return cleaned
        except Exception as e:
            logger.error(f"case1 이미지 정리 실패 ({image_path}): {e}")
            return None
    
    def clean_case2(self, image_path, image_index):
        """case2 이미지 정리
        
        의도: 번호별로 다른 처리 방식 적용 (21번 이상은 기호 모드)
        
        Args:
            image_path: 처리할 이미지 경로
            image_index: 이미지 번호 (0부터 시작)
        
        Returns:
            정리된 이미지 또는 None (실패 시)
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"이미지 로드 실패: {image_path}")
            return None
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if image_index >= 21:
                # 21~38번: 기호 추출 모드
                result = self._extract_symbols(gray)
                logger.debug(f"case2 기호 모드 처리: {image_path}")
            else:
                # 0~8, 10~18, 20번: 전경 검은색, 배경 흰색
                result = self._process_normal_mode(gray)
                logger.debug(f"case2 숫자 모드 처리: {image_path}")
            
            return result
        except Exception as e:
            logger.error(f"case2 이미지 정리 실패 ({image_path}): {e}")
            return None
    
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
        """폴더 내 모든 이미지 정리
        
        의도: 지정된 폴더의 모든 이미지를 케이스별로 정리하여 출력 폴더에 저장
        
        Args:
            input_folder: 입력 폴더 경로
            output_folder: 출력 폴더 경로
            case: 처리 케이스 ('case1', 'case2', 'case3')
        
        Returns:
            처리 성공 여부
        """
        if not os.path.exists(input_folder):
            logger.error(f"입력 폴더가 존재하지 않음: {input_folder}")
            return False
        
        try:
            logger.info(f"{case} 이미지 정리 시작: {input_folder}")
            processed_count = 0
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            for folder_name in os.listdir(input_folder):
                folder_path = os.path.join(input_folder, folder_name)
                
                if os.path.isdir(folder_path):
                    output_folder_path = os.path.join(output_folder, folder_name)
                    ensure_directory(output_folder_path)
                    
                    image_files = [f for f in os.listdir(folder_path) 
                                 if f.lower().endswith(tuple(IMAGE_EXTENSIONS))]
                    
                    for image_file in image_files:
                        try:
                            input_image_path = os.path.join(folder_path, image_file)
                            output_image_path = os.path.join(output_folder_path, image_file)
                            
                            if self._should_process_image(image_file, case):
                                if case == "case1":
                                    # case1: 숫자 형식 (0.png, 1.png, ...)
                                    image_index = int(image_file.split('.')[0])
                                    cleaned_image = self.clean_case1(input_image_path)
                                elif case == "case2":
                                    # case2: 숫자 형식 (0.png, 1.png, ...)
                                    image_index = int(image_file.split('.')[0])
                                    cleaned_image = self.clean_case2(input_image_path, image_index)
                                elif case == "case3":
                                    # case3: 그룹명 형식 (between_group_1.png, after_group_1.png)
                                    cleaned_image = self.clean_case1(input_image_path)  # case1과 동일
                                else:
                                    raise ValueError(f"Invalid case: {case}")
                                
                                if cleaned_image is not None:
                                    cv2.imwrite(output_image_path, cleaned_image)
                                    processed_count += 1
                                else:
                                    logger.warning(f"이미지 정리 실패: {image_file}")
                            else:
                                # 처리하지 않는 이미지는 원본 복사
                                original_img = cv2.imread(input_image_path)
                                cv2.imwrite(output_image_path, original_img)
                        except Exception as e:
                            logger.error(f"이미지 처리 실패 ({image_file}): {e}")
                            continue
            
            logger.info(f"{case} 이미지 정리 완료: {processed_count}개 처리")
            return True
        except Exception as e:
            logger.error(f"{case} 이미지 정리 중 오류 발생: {e}")
            return False
