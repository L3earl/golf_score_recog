#!/usr/bin/env python3
"""
테이블 감지 및 크롭 테스트 스크립트 (특징점 매칭 방식)

이 스크립트는 data/raw_img 폴더의 이미지들을 가져와서
data/template/case3_01.png 템플릿을 사용하여 특징점 매칭으로 테이블을 감지하고,
해당 영역을 크롭하여 test 폴더에 저장합니다.

특징점 매칭 방식의 특징:
- SIFT 특징점을 사용한 정확한 매칭
- 호모그래피 변환으로 정확한 위치 계산
- 회전, 크기 변화에 강함
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

def find_feature_matches(image_path: str, template_path: str, min_matches: int = 10) -> Optional[np.ndarray]:
    """특징점 매칭으로 템플릿 위치 찾기"""
    try:
        # 이미지와 템플릿 로드
        image = cv2.imread(image_path)
        template = cv2.imread(template_path)
        
        if image is None:
            logger.error(f"이미지를 로드할 수 없습니다: {image_path}")
            return None
        
        if template is None:
            logger.error(f"템플릿을 로드할 수 없습니다: {template_path}")
            return None
        
        # 이미지와 템플릿 크기 확인
        img_height, img_width = image.shape[:2]
        templ_height, templ_width = template.shape[:2]
        
        logger.info(f"이미지 크기: {img_width}x{img_height}")
        logger.info(f"템플릿 크기: {templ_width}x{templ_height}")
        
        # SIFT 특징점 검출기 생성
        sift = cv2.SIFT_create()
        
        # 특징점과 디스크립터 검출
        kp1, des1 = sift.detectAndCompute(image, None)
        kp2, des2 = sift.detectAndCompute(template, None)
        
        logger.info(f"이미지 특징점: {len(kp1)}개")
        logger.info(f"템플릿 특징점: {len(kp2)}개")
        
        if des1 is None or des2 is None:
            logger.warning("특징점 디스크립터를 생성할 수 없습니다.")
            return None
        
        # BFMatcher로 매칭
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des2, des1, k=2)
        
        # 좋은 매칭점 필터링 (Lowe's ratio test)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        logger.info(f"좋은 매칭점: {len(good_matches)}개")
        
        # 충분한 매칭점이 있는지 확인
        if len(good_matches) < min_matches:
            logger.warning(f"충분한 매칭점이 없습니다. (필요: {min_matches}, 발견: {len(good_matches)})")
            return None
        
        # 호모그래피로 위치 계산
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            logger.warning("호모그래피 행렬을 계산할 수 없습니다.")
            return None
        
        # 템플릿의 네 모서리를 원본 이미지 좌표로 변환
        h, w = template.shape[:2]
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        
        logger.info(f"호모그래피 변환 성공: {len(good_matches)}개 매칭점 사용")
        return dst
        
    except Exception as e:
        logger.error(f"특징점 매칭 중 오류 발생: {e}")
        return None

def calculate_crop_area(transformed_corners: np.ndarray) -> Tuple[int, int, int, int]:
    """변환된 모서리 좌표를 기반으로 크롭 영역 계산"""
    try:
        # 변환된 모서리 좌표를 정수로 변환
        corners = transformed_corners.reshape(-1, 2).astype(int)
        
        # 바운딩 박스 계산
        x_min = np.min(corners[:, 0])
        y_min = np.min(corners[:, 1])
        x_max = np.max(corners[:, 0])
        y_max = np.max(corners[:, 1])
        
        # 크롭 영역 계산
        crop_x = x_min
        crop_y = y_min
        crop_width = x_max - x_min
        crop_height = y_max - y_min
        
        logger.info(f"변환된 모서리 좌표: {corners.tolist()}")
        logger.info(f"크롭 영역: ({crop_x}, {crop_y}, {crop_width}, {crop_height})")
        
        return (crop_x, crop_y, crop_width, crop_height)
        
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
        output_filename = f"{base_name}_feature_crop.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 저장
        cv2.imwrite(output_path, cropped_image)
        logger.info(f"✅ 특징점 크롭 저장: {output_path}")
        logger.info(f"   크롭된 이미지 크기: {cropped_image.shape}")
        
    except Exception as e:
        logger.error(f"특징점 크롭 저장 중 오류: {e}")

def process_images(input_dir: str, output_dir: str, template_path: str, min_matches: int = 10) -> None:
    """모든 이미지 파일 처리 (특징점 매칭 방식)"""
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        logger.warning(f"{input_dir}에 이미지 파일이 없습니다.")
        return
    
    logger.info(f"처리할 이미지 파일 수: {len(image_files)}")
    logger.info(f"템플릿 파일: {template_path}")
    logger.info(f"최소 매칭점: {min_matches}")
    
    success_count = 0
    fail_count = 0
    
    # 각 이미지 처리
    for filename in image_files:
        try:
            logger.info(f"처리 중: {filename}")
            
            # 이미지 경로
            image_path = os.path.join(input_dir, filename)
            
            # 특징점 매칭
            transformed_corners = find_feature_matches(image_path, template_path, min_matches)
            
            if transformed_corners is not None:
                # 크롭 영역 계산
                crop_area = calculate_crop_area(transformed_corners)
                
                if crop_area[2] > 0 and crop_area[3] > 0:  # 유효한 크기인지 확인
                    # 크롭 및 저장
                    crop_and_save_table(image_path, crop_area, filename, output_dir)
                    success_count += 1
                else:
                    logger.warning(f"{filename}: 유효하지 않은 크롭 영역")
                    fail_count += 1
            else:
                logger.warning(f"{filename}: 특징점 매칭 실패")
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
        template_path = os.path.join(template_dir, 'case3_01.png')
        
        setup_directories(input_dir, output_dir, template_dir)
        
        # 설정 옵션들
        min_matches = 10  # 최소 매칭점 수 (더 높을수록 더 정확한 매칭)
        
        logger.info("=" * 60)
        logger.info("특징점 매칭을 사용한 테이블 감지 및 크롭 시작")
        logger.info("=" * 60)
        
        # 이미지 처리
        process_images(input_dir, output_dir, template_path, min_matches)
        
        logger.info("=" * 60)
        logger.info("모든 이미지 처리 완료!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {e}")
        raise

if __name__ == "__main__":
    main()
