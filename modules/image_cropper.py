"""
이미지 크롭 모듈

의도: 이미지를 지정된 좌표로 크롭하여 개별 숫자/기호 추출
- crop_coordinates.py의 좌표 정보 활용
- 각 이미지를 개별 파일로 저장
"""

import os
from PIL import Image
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils import ensure_directory
from config import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

class ImageCropper:
    """이미지 크롭 싱글톤 클래스
    
    의도: 프로젝트 전체에서 단일 인스턴스로 이미지 크롭 작업 수행
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageCropper, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """이미지 크롭기 초기화
        
        의도: 싱글톤 인스턴스 초기화
        """
        if not self._initialized:
            # 초기화 코드 (현재는 없음)
            self._initialized = True
            logger.debug("ImageCropper 초기화 완료")
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 반환
        
        의도: 어디서든 동일한 인스턴스 접근 가능
        """
        if cls._instance is None:
            cls._instance = ImageCropper()
        return cls._instance
    
    def crop_image(self, image_path, coordinates, output_dir):
        """단일 이미지 크롭
        
        의도: 지정된 좌표로 이미지를 크롭하여 개별 파일로 저장
        
        Args:
            image_path: 크롭할 이미지 경로
            coordinates: 크롭 좌표 리스트 [(left, top, right, bottom), ...]
            output_dir: 출력 디렉토리
        
        Returns:
            크롭 성공 여부
        """
        if not coordinates:
            logger.warning(f"크롭 좌표가 없음: {image_path}")
            return False
        
        try:
            image = Image.open(image_path)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            img_output_dir = os.path.join(output_dir, image_name)
            ensure_directory(img_output_dir)
            
            for i, (left, top, right, bottom) in enumerate(coordinates):
                try:
                    cropped = image.crop((left, top, right, bottom))
                    output_path = os.path.join(img_output_dir, f"{i}.png")
                    cropped.save(output_path)
                except Exception as e:
                    logger.error(f"크롭 실패 ({image_path}, 좌표 {i}): {e}")
                    continue
            
            logger.debug(f"이미지 크롭 완료: {image_path} -> {len(coordinates)}개")
            return True
        except Exception as e:
            logger.error(f"이미지 크롭 실패 ({image_path}): {e}")
            return False
    
    def crop_all_images(self, input_folder, coordinates, output_folder, target_files=None):
        """폴더 내 모든 이미지 크롭
        
        의도: 지정된 폴더의 모든 이미지를 동일한 좌표로 크롭
        ImageCleaner와 동일한 하위 폴더 처리 방식 사용
        
        Args:
            input_folder: 입력 폴더 경로
            coordinates: 크롭 좌표 리스트
            output_folder: 출력 폴더 경로
            target_files: 처리할 파일명 리스트 (None이면 전체)
        
        Returns:
            크롭 성공 여부
        """
        if not os.path.exists(input_folder):
            logger.error(f"입력 폴더가 존재하지 않음: {input_folder}")
            return False
        
        try:
            logger.info(f"이미지 크롭 시작: {input_folder}")
            processed_count = 0
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # ImageCleaner와 동일한 방식으로 하위 폴더 처리
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
                            
                            # target_files가 지정되면 해당 파일만 처리
                            if target_files:
                                image_name = os.path.splitext(image_file)[0]
                                if image_name not in target_files:
                                    continue
                            
                            if self.crop_image(input_image_path, coordinates, output_folder_path):
                                processed_count += 1
                            else:
                                logger.warning(f"이미지 크롭 실패: {image_file}")
                        except Exception as e:
                            logger.error(f"이미지 처리 실패 ({image_file}): {e}")
                            continue
                else:
                    # 폴더가 아닌 경우 (직접 이미지 파일들)
                    if folder_name.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                        try:
                            # target_files가 지정되면 해당 파일만 처리
                            if target_files:
                                image_name = os.path.splitext(folder_name)[0]
                                if image_name not in target_files:
                                    continue
                            
                            if self.crop_image(folder_path, coordinates, output_folder):
                                processed_count += 1
                            else:
                                logger.warning(f"이미지 크롭 실패: {folder_name}")
                        except Exception as e:
                            logger.error(f"이미지 처리 실패 ({folder_name}): {e}")
                            continue
            
            logger.info(f"이미지 크롭 완료: {processed_count}개 처리")
            return processed_count > 0
        except Exception as e:
            logger.error(f"이미지 크롭 중 오류 발생: {e}")
            return False
