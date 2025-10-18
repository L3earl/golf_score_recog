#!/usr/bin/env python3
"""
테이블 감지 및 크롭 테스트 스크립트 (템플릿 매칭 방식)

이 스크립트는 data/raw_img 폴더의 이미지들을 가져와서
data/template/case3.png 템플릿을 사용하여 템플릿 매칭으로 테이블을 감지하고,
해당 영역을 크롭하여 test 폴더에 저장합니다.

템플릿 매칭 방식의 특징:
- 정확한 템플릿이 있을 때 매우 정확한 감지
- 2개의 템플릿을 찾아서 사이 간격을 계산
- 동적 크롭 영역 계산
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories(input_dir: str, output_dir: str, template_dir: str) -> None:
    """입력, 출력, 템플릿 디렉토리 설정"""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
    
    if not os.path.exists(template_dir):
        raise FileNotFoundError(f"템플릿 디렉토리가 존재하지 않습니다: {template_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"입력 디렉토리: {input_dir}")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"템플릿 디렉토리: {template_dir}")

def find_template_matches(image_path: str, template_path: str, threshold: float = 0.8) -> List[Tuple[int, int, int, int]]:
    """템플릿 매칭으로 템플릿 위치 찾기"""
    try:
        # 이미지와 템플릿 로드
        image = cv2.imread(image_path)
        template = cv2.imread(template_path)
        
        if image is None:
            logger.error(f"이미지를 로드할 수 없습니다: {image_path}")
            return []
        
        if template is None:
            logger.error(f"템플릿을 로드할 수 없습니다: {template_path}")
            return []
        
        # 이미지와 템플릿 크기 확인
        img_height, img_width = image.shape[:2]
        templ_height, templ_width = template.shape[:2]
        
        logger.info(f"이미지 크기: {img_width}x{img_height}")
        logger.info(f"템플릿 크기: {templ_width}x{templ_height}")
        
        # 템플릿이 이미지보다 큰 경우 에러 처리
        if templ_width > img_width or templ_height > img_height:
            logger.error(f"템플릿이 이미지보다 큽니다. 템플릿: {templ_width}x{templ_height}, 이미지: {img_width}x{img_height}")
            return []
        
        # 템플릿 매칭
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        
        # 최대 매칭값 확인
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        logger.info(f"최대 매칭값: {max_val:.3f} (임계값: {threshold})")
        
        # 임계값 이상의 매칭 위치 찾기
        locations = np.where(result >= threshold)
        matches = []
        
        # 템플릿 크기
        h, w = template.shape[:2]
        
        for pt in zip(*locations[::-1]):
            x, y = pt
            matches.append((x, y, w, h))
        
        # 중복 제거 (너무 가까운 매칭들)
        filtered_matches = []
        for match in matches:
            x, y, w, h = match
            is_duplicate = False
            
            for existing in filtered_matches:
                ex, ey, ew, eh = existing
                if abs(x - ex) < w//2 and abs(y - ey) < h//2:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_matches.append(match)
        
        logger.info(f"템플릿 매칭 완료: {len(filtered_matches)}개 매칭 발견")
        return filtered_matches
        
    except Exception as e:
        logger.error(f"템플릿 매칭 중 오류 발생: {e}")
        return []

def calculate_crop_area(matches: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    """매칭된 템플릿들을 기반으로 크롭 영역 계산"""
    if len(matches) < 2:
        logger.warning("최소 2개의 템플릿 매칭이 필요합니다.")
        return None
    
    # 첫 번째와 두 번째 템플릿 선택
    template1 = matches[0]  # (x, y, w, h)
    template2 = matches[1]  # (x, y, w, h)
    
    x1, y1, w1, h1 = template1
    x2, y2, w2, h2 = template2
    
    # 1번 템플릿의 좌우 길이를 넓이로 설정
    width = w1
    
    # 1번과 2번 템플릿 사이의 간격 계산
    gap = abs(x2 - x1) - w1
    
    # 크롭 영역 계산: (넓이, 템플릿 높이 + 간격*2)
    crop_width = width
    crop_height = h1 + gap * 2
    
    # 크롭 시작점 (1번 템플릿 시작점)
    crop_x = x1
    crop_y = y1
    
    logger.info(f"템플릿1: ({x1}, {y1}, {w1}, {h1})")
    logger.info(f"템플릿2: ({x2}, {y2}, {w2}, {h2})")
    logger.info(f"간격: {gap}")
    logger.info(f"크롭 영역: ({crop_x}, {crop_y}, {crop_width}, {crop_height})")
    
    return (crop_x, crop_y, crop_width, crop_height)

def crop_and_save_table(image_path: str, crop_area: Tuple[int, int, int, int], 
                       filename: str, output_dir: str) -> None:
    """계산된 영역을 크롭하여 저장"""
    try:
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"이미지를 다시 로드할 수 없습니다: {image_path}")
            return
        
        x, y, w, h = crop_area
        
        # 이미지 범위 확인
        img_height, img_width = image.shape[:2]
        x = max(0, min(x, img_width))
        y = max(0, min(y, img_height))
        w = max(0, min(w, img_width - x))
        h = max(0, min(h, img_height - y))
        
        # 유효한 크롭 영역인지 확인
        if w <= 0 or h <= 0:
            logger.warning(f"유효하지 않은 크롭 영역: ({x}, {y}, {w}, {h})")
            return
        
        # 크롭
        cropped_image = image[y:y+h, x:x+w]
        
        # 파일명 생성
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_template_crop.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 저장
        cv2.imwrite(output_path, cropped_image)
        logger.info(f"✅ 템플릿 크롭 저장: {output_path}")
        logger.info(f"   크롭된 이미지 크기: {cropped_image.shape}")
        
    except Exception as e:
        logger.error(f"템플릿 크롭 저장 중 오류: {e}")

def process_images(input_dir: str, output_dir: str, template_path: str, threshold: float = 0.8) -> None:
    """모든 이미지 파일 처리 (템플릿 매칭 방식)"""
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        logger.warning(f"{input_dir}에 이미지 파일이 없습니다.")
        return
    
    logger.info(f"처리할 이미지 파일 수: {len(image_files)}")
    logger.info(f"템플릿 파일: {template_path}")
    logger.info(f"매칭 임계값: {threshold}")
    
    success_count = 0
    fail_count = 0
    
    # 각 이미지 처리
    for filename in image_files:
        try:
            logger.info(f"처리 중: {filename}")
            
            # 이미지 경로
            image_path = os.path.join(input_dir, filename)
            
            # 템플릿 매칭
            matches = find_template_matches(image_path, template_path, threshold)
            
            if len(matches) >= 2:
                # 크롭 영역 계산
                crop_area = calculate_crop_area(matches)
                
                if crop_area:
                    # 크롭 및 저장
                    crop_and_save_table(image_path, crop_area, filename, output_dir)
                    success_count += 1
                else:
                    logger.warning(f"{filename}: 크롭 영역 계산 실패")
                    fail_count += 1
            else:
                logger.warning(f"{filename}: 충분한 템플릿 매칭을 찾지 못했습니다. (발견: {len(matches)}개)")
                fail_count += 1
            
        except Exception as e:
            logger.error(f"{filename} 처리 중 오류: {e}")
            fail_count += 1
            continue
    
    # 처리 결과 요약
    logger.info("=" * 60)
    logger.info("처리 결과 요약")
    logger.info("=" * 60)
    logger.info(f"총 처리 파일: {len(image_files)}")
    logger.info(f"성공: {success_count}")
    logger.info(f"실패: {fail_count}")
    logger.info(f"성공률: {(success_count / len(image_files) * 100):.1f}%")

def main():
    """메인 함수"""
    try:
        # 디렉토리 설정
        input_dir = 'data/raw_img'
        output_dir = 'test'
        template_dir = 'data/template_img'
        template_path = os.path.join(template_dir, 'case3.png')
        
        setup_directories(input_dir, output_dir, template_dir)
        
        # 설정 옵션들
        threshold = 0.3  # 템플릿 매칭 임계값 (0.1~1.0, 높을수록 더 정확한 매칭) - 낮춤
        
        logger.info("=" * 60)
        logger.info("템플릿 매칭을 사용한 테이블 감지 및 크롭 시작")
        logger.info("=" * 60)
        
        # 이미지 처리
        process_images(input_dir, output_dir, template_path, threshold)
        
        logger.info("=" * 60)
        logger.info("모든 이미지 처리 완료!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {e}")
        raise

if __name__ == "__main__":
    main()
