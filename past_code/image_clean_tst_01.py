"""
test 폴더의 이미지들을 case1 방식으로 클리닝하여 test_clean 폴더에 저장하는 테스트 스크립트
"""

import os
import logging
import sys
import cv2

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.image_cleaner import ImageCleaner

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """test 폴더의 이미지들을 case1 방식으로 클리닝"""
    
    # 폴더 경로 설정
    test_folder = "test"
    test_clean_folder = "test_clean"
    
    # test_clean 폴더가 없으면 생성
    os.makedirs(test_clean_folder, exist_ok=True)
    logger.info(f"출력 폴더 생성: {test_clean_folder}")
    
    # test 폴더 존재 확인
    if not os.path.exists(test_folder):
        logger.error(f"입력 폴더가 존재하지 않습니다: {test_folder}")
        return False
    
    # test 폴더의 이미지 파일 개수 확인
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    logger.info(f"처리할 이미지 파일 개수: {len(image_files)}")
    
    if not image_files:
        logger.warning("처리할 이미지 파일이 없습니다.")
        return False
    
    try:
        # ImageCleaner 인스턴스 가져오기
        cleaner = ImageCleaner.get_instance()
        logger.info("ImageCleaner 인스턴스 초기화 완료")
        
        # case1 방식으로 개별 이미지 클리닝 수행
        logger.info("case1 방식으로 이미지 클리닝 시작...")
        processed_count = 0
        
        for image_file in image_files:
            try:
                input_path = os.path.join(test_folder, image_file)
                output_path = os.path.join(test_clean_folder, image_file)
                
                # case1 방식으로 클리닝
                cleaned_image = cleaner.clean_case1(input_path)
                
                if cleaned_image is not None:
                    cv2.imwrite(output_path, cleaned_image)
                    processed_count += 1
                    logger.debug(f"처리 완료: {image_file}")
                else:
                    logger.warning(f"클리닝 실패: {image_file}")
                    
            except Exception as e:
                logger.error(f"이미지 처리 실패 ({image_file}): {e}")
                continue
        
        if processed_count > 0:
            logger.info(f"✅ 이미지 클리닝 완료: {processed_count}개 처리")
            return True
        else:
            logger.error("❌ 처리된 이미지가 없습니다")
            return False
            
    except Exception as e:
        logger.error(f"클리닝 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 테스트 완료: test 폴더의 이미지들이 test_clean 폴더에 클리닝되어 저장되었습니다.")
    else:
        print("\n❌ 테스트 실패: 이미지 클리닝 중 오류가 발생했습니다.")
