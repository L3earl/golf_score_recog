#!/usr/bin/env python3
"""
테이블 감지 및 크롭 테스트 스크립트 (OpenCV 방식)

이 스크립트는 data/raw_img 폴더의 이미지들을 가져와서
OpenCV를 사용한 컴퓨터 비전 방식으로 테이블의 좌표를 감지하고,
해당 영역을 크롭하여 test 폴더에 저장합니다.

OpenCV 방식의 특징:
- 가장 큰 윤곽선을 테이블로 간주
- 이진화 및 윤곽선 검출 사용
- 다크 모드 이미지도 처리 가능
"""

import os
import cv2
import numpy as np
import logging
from typing import Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories(input_dir: str, output_dir: str) -> None:
    """입력 및 출력 디렉토리 설정"""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"입력 디렉토리: {input_dir}")
    logger.info(f"출력 디렉토리: {output_dir}")

def detect_table_cv(image_path: str, invert_colors: bool = False) -> Optional[Tuple[int, int, int, int]]:
    """OpenCV를 사용한 테이블 감지 (엄격한 필터링 적용)"""
    try:
        # 1. 이미지 불러오기
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"이미지를 로드할 수 없습니다: {image_path}")
            return None
        
        height, width = image.shape[:2]
        total_area = height * width
        logger.info(f"이미지 로드 완료: {os.path.basename(image_path)} (크기: {image.shape})")
        
        # 2. 이미지 전처리
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 색상 반전 처리 (다크 모드 이미지용)
        if invert_colors:
            logger.info("색상 반전 적용")
            gray = cv2.bitwise_not(gray)
        
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 이진화 (배경과 객체를 분리)
        # THRESH_BINARY_INV: 배경(어두운 부분)을 흰색으로, 객체(밝은 부분)을 검은색으로 반전
        # Otsu의 알고리즘을 사용해 임계값을 자동으로 결정
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. 윤곽선 찾기
        # cv2.RETR_EXTERNAL: 가장 바깥쪽 윤곽선만 찾음
        # cv2.CHAIN_APPROX_SIMPLE: 윤곽선의 꼭짓점 좌표만 저장하여 메모리 절약
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("윤곽선을 찾지 못했습니다.")
            return None
        
        logger.info(f"총 {len(contours)}개의 윤곽선을 찾았습니다.")
        
        # 4. 엄격한 필터링으로 윤곽선 탐색
        valid_contours = []
        
        for i, contour in enumerate(contours):
            # 윤곽선 면적 계산
            area = cv2.contourArea(contour)
            area_ratio = area / total_area * 100
            
            # 경계 박스 계산
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # 가장자리 거리 계산 (이미지 가장자리에서 얼마나 떨어져 있는지)
            margin_x = min(x, width - x - w)
            margin_y = min(y, height - y - h)
            min_margin = min(margin_x, margin_y)
            
            logger.info(f"윤곽선 {i+1}: 면적={area:.0f} ({area_ratio:.1f}%), "
                       f"크기=({w}x{h}), 종횡비={aspect_ratio:.2f}, 가장자리거리={min_margin}")
            
            # 엄격한 필터링 조건들
            conditions = {
                'min_area': area >= total_area * 0.01,  # 최소 면적: 전체의 1% 이상
                'max_area': area <= total_area * 0.95,  # 최대 면적: 전체의 95% 이하 (이미지 프레임 제외)
                'min_size': w >= 50 and h >= 50,        # 최소 크기: 50x50 픽셀 이상
                'aspect_ratio': 0.2 <= aspect_ratio <= 5.0,  # 종횡비: 0.2 ~ 5.0 (너무 얇거나 긴 것 제외)
                'reasonable_area': 0.5 <= area_ratio <= 80.0  # 합리적인 면적 비율: 0.5% ~ 80%
            }
            
            # 모든 조건을 만족하는지 확인
            if all(conditions.values()):
                valid_contours.append((contour, area, area_ratio, x, y, w, h))
                logger.info(f"✅ 윤곽선 {i+1} 통과: 모든 조건 만족")
            else:
                failed_conditions = [k for k, v in conditions.items() if not v]
                logger.info(f"❌ 윤곽선 {i+1} 제외: 실패 조건 = {failed_conditions}")
        
        if not valid_contours:
            logger.warning("조건을 만족하는 윤곽선이 없습니다.")
            return None
        
        # 5. 조건을 만족하는 윤곽선 중 가장 큰 것 선택
        best_contour = max(valid_contours, key=lambda x: x[1])  # 면적으로 정렬
        contour, area, area_ratio, x, y, w, h = best_contour
        
        logger.info(f"✅ 최적 테이블 감지!")
        logger.info(f"   좌표: x={x}, y={y}, 너비={w}, 높이={h}")
        logger.info(f"   면적: {area:.0f} 픽셀 ({area_ratio:.1f}%)")
        
        return (x, y, w, h)
        
    except Exception as e:
        logger.error(f"테이블 감지 중 오류 발생: {e}")
        return None

def crop_and_save_table(image_path: str, table_coords: Tuple[int, int, int, int], 
                       output_dir: str) -> None:
    """감지된 테이블 영역을 크롭하여 저장"""
    try:
        # 이미지 다시 로드
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"이미지를 다시 로드할 수 없습니다: {image_path}")
            return
        
        x, y, w, h = table_coords
        
        # 이미지 범위 확인
        height, width = image.shape[:2]
        x = max(0, min(x, width))
        y = max(0, min(y, height))
        w = max(0, min(w, width - x))
        h = max(0, min(h, height - y))
        
        # 유효한 크롭 영역인지 확인
        if w <= 0 or h <= 0:
            logger.warning(f"유효하지 않은 크롭 영역: ({x}, {y}, {w}, {h})")
            return
        
        # 원본 이미지에서 해당 좌표로 잘라내기
        cropped_image = image[y:y+h, x:x+w]
        
        # 파일명 생성
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{base_name}_table_cv.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 잘라낸 이미지 저장
        cv2.imwrite(output_path, cropped_image)
        logger.info(f"✅ 테이블 크롭 저장: {output_path}")
        logger.info(f"   크롭된 이미지 크기: {cropped_image.shape}")
        
    except Exception as e:
        logger.error(f"테이블 크롭 저장 중 오류: {e}")

def process_images(input_dir: str, output_dir: str, invert_colors: bool = False) -> None:
    """모든 이미지 파일 처리 (OpenCV 방식)"""
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        logger.warning(f"{input_dir}에 이미지 파일이 없습니다.")
        return
    
    logger.info(f"처리할 이미지 파일 수: {len(image_files)}")
    logger.info(f"색상 반전 옵션: {'ON' if invert_colors else 'OFF'}")
    
    success_count = 0
    fail_count = 0
    
    # 각 이미지 처리
    for filename in image_files:
        try:
            logger.info(f"처리 중: {filename}")
            
            # 이미지 경로
            image_path = os.path.join(input_dir, filename)
            
            # 테이블 감지
            table_coords = detect_table_cv(image_path, invert_colors)
            
            if table_coords:
                # 테이블 크롭 및 저장
                crop_and_save_table(image_path, table_coords, output_dir)
                success_count += 1
            else:
                logger.warning(f"{filename}: 테이블을 감지하지 못했습니다.")
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
        
        setup_directories(input_dir, output_dir)
        
        # 설정 옵션들
        invert_colors = True  # 다크 모드 이미지용 색상 반전 (True/False)
        
        logger.info("=" * 60)
        logger.info("OpenCV를 사용한 테이블 감지 및 크롭 시작")
        logger.info("=" * 60)
        
        # 이미지 처리
        process_images(input_dir, output_dir, invert_colors)
        
        logger.info("=" * 60)
        logger.info("모든 이미지 처리 완료!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {e}")
        raise

if __name__ == "__main__":
    main()
