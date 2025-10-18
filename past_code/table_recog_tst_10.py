#!/usr/bin/env python3
"""
테이블 감지 및 크롭 테스트 스크립트 (템플릿 매칭 방식)

이 스크립트는 data/raw_img 폴더의 이미지들을 가져와서
data/template_img/case3_02.png 템플릿을 사용하여 템플릿 매칭으로 테이블을 감지하고,
2개의 매칭을 찾아서 그 사이의 영역을 크롭하여 test 폴더에 저장합니다.

템플릿 매칭 방식의 특징:
- cv2.matchTemplate()을 사용한 빠른 매칭
- 여러 매칭 위치 찾기
- 위쪽과 아래쪽 매칭 사이의 거리 계산
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

def find_template_matches(image_path: str, template_path: str, threshold: float = 0.8) -> List[Tuple[int, int]]:
    """템플릿 매칭으로 여러 위치 찾기 (그레이스케일 변환으로 색상 영향 제거)"""
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
        
        # 그레이스케일로 변환 (색상 영향 제거)
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
            
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template
        
        # 이미지와 템플릿 크기 확인
        img_height, img_width = image_gray.shape[:2]
        templ_height, templ_width = template_gray.shape[:2]
        
        logger.info(f"이미지 크기: {img_width}x{img_height} (그레이스케일)")
        logger.info(f"템플릿 크기: {templ_width}x{templ_height} (그레이스케일)")
        
        # 템플릿 매칭 수행 (그레이스케일 이미지 사용)
        result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # 임계값 이상인 위치들 찾기
        locations = np.where(result >= threshold)
        matches = list(zip(locations[1], locations[0]))  # (x, y) 형태로 변환
        
        logger.info(f"임계값 {threshold} 이상인 매칭: {len(matches)}개")
        
        # 너무 가까운 매칭들 제거 (중복 제거)
        filtered_matches = []
        for match in matches:
            is_duplicate = False
            for existing in filtered_matches:
                distance = np.sqrt((match[0] - existing[0])**2 + (match[1] - existing[1])**2)
                if distance < min(templ_width, templ_height) * 0.5:  # 템플릿 크기의 절반 이내면 중복으로 간주
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_matches.append(match)
        
        logger.info(f"중복 제거 후 매칭: {len(filtered_matches)}개")
        return filtered_matches
        
    except Exception as e:
        logger.error(f"템플릿 매칭 중 오류 발생: {e}")
        return []

def calculate_crop_dimensions(matches: List[Tuple[int, int]], template_path: str) -> Tuple[int, int, int, int]:
    """매칭된 위치들을 기반으로 크롭 영역 계산"""
    try:
        if len(matches) < 2:
            logger.warning("2개 이상의 매칭이 필요합니다.")
            return (0, 0, 0, 0)
        
        # 템플릿 크기 가져오기
        template = cv2.imread(template_path)
        templ_height, templ_width = template.shape[:2]
        
        # y 좌표로 정렬 (위쪽부터)
        sorted_matches = sorted(matches, key=lambda x: x[1])
        
        # 위쪽과 아래쪽 매칭 선택
        top_match = sorted_matches[0]  # 가장 위쪽
        bottom_match = sorted_matches[-1]  # 가장 아래쪽
        
        logger.info(f"위쪽 매칭: {top_match}")
        logger.info(f"아래쪽 매칭: {bottom_match}")
        
        # 높이 계산 (아래쪽 매칭의 시작점 - 위쪽 매칭의 시작점)
        height = bottom_match[1] - top_match[1]
        
        # 너비 계산 (템플릿의 너비 사용)
        width = templ_width
        
        # 시작점 (위쪽 매칭의 시작점)
        start_x = top_match[0]
        start_y = top_match[1]
        
        # 크롭 영역 크기 계산 (너비 * (높이 * 2))
        crop_width = width
        crop_height = height * 2
        
        logger.info(f"계산된 높이: {height}")
        logger.info(f"계산된 너비: {width}")
        logger.info(f"크롭 영역: ({start_x}, {start_y}, {crop_width}, {crop_height})")
        
        return (start_x, start_y, crop_width, crop_height)
        
    except Exception as e:
        logger.error(f"크롭 영역 계산 중 오류 발생: {e}")
        return (0, 0, 0, 0)

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
                crop_area = calculate_crop_dimensions(matches, template_path)
                
                if crop_area[2] > 0 and crop_area[3] > 0:  # 유효한 크기인지 확인
                    # 크롭 및 저장
                    crop_and_save_table(image_path, crop_area, filename, output_dir)
                    success_count += 1
                else:
                    logger.warning(f"{filename}: 유효하지 않은 크롭 영역")
                    fail_count += 1
            else:
                logger.warning(f"{filename}: 2개 이상의 매칭을 찾을 수 없음 (발견: {len(matches)}개)")
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
        template_path = os.path.join(template_dir, 'case3_02.png')
        
        setup_directories(input_dir, output_dir, template_dir)
        
        # 설정 옵션들
        threshold = 0.8  # 템플릿 매칭 임계값 (0.0~1.0, 높을수록 더 정확한 매칭)
        
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
