"""
이미지 크롭 모듈 (싱글톤 패턴)
- 케이스별 좌표로 이미지 크롭
- 모든 케이스에서 재사용 가능
"""

import os
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_EXTENSIONS

class ImageCropper:
    """이미지 크롭 싱글톤 클래스"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageCropper, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = ImageCropper()
        return cls._instance
    
    def crop_image(self, image_path, coordinates, output_dir):
        """단일 이미지 크롭"""
        if not coordinates:
            return False
            
        try:
            image = Image.open(image_path)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            img_output_dir = os.path.join(output_dir, image_name)
            os.makedirs(img_output_dir, exist_ok=True)
            
            for i, (left, top, right, bottom) in enumerate(coordinates):
                try:
                    cropped = image.crop((left, top, right, bottom))
                    output_path = os.path.join(img_output_dir, f"{i}.png")
                    cropped.save(output_path)
                except:
                    continue
            
            return True
        except:
            return False
    
    def crop_all_images(self, input_folder, coordinates, output_folder, target_files=None):
        """
        폴더 내 모든 이미지 크롭
        target_files: None이면 전체, 리스트면 해당 파일명만 처리
        """
        if not os.path.exists(input_folder):
            print(f"❌ 입력 폴더 없음: {input_folder}")
            return False
            
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(tuple(IMAGE_EXTENSIONS))]
        
        if not image_files:
            print(f"❌ 이미지 파일 없음: {input_folder}")
            return False
        
        processed_count = 0
        for filename in image_files:
            # target_files가 지정되면 해당 파일만 처리
            if target_files:
                image_name = os.path.splitext(filename)[0]
                if image_name not in target_files:
                    continue
            
            img_path = os.path.join(input_folder, filename)
            if self.crop_image(img_path, coordinates, output_folder):
                processed_count += 1
        
        print(f"  ✓ 크롭: {processed_count}개")
        return processed_count > 0
