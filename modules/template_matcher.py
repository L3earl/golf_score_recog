"""
case3 템플릿 매칭 및 크롭 모듈

의도: case2를 통과하지 못한 예외 파일들을 템플릿 매칭으로 처리
- raw_img에서 예외 파일들을 가져와서 template_img의 case3_01.png와 매칭
- 매칭된 이미지를 raw_template_crop 폴더에 저장하여 후속 처리 준비
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from config import RAW_IMG_FOLDER, RAW_TEMPLATE_CROP_FOLDER, DATA_FOLDER

# 로깅 설정
logger = logging.getLogger(__name__)

def setup_template_crop_directory():
    """raw_template_crop 디렉토리 설정
    
    의도: 템플릿 매칭 결과를 저장할 디렉토리를 안전하게 생성
    """
    os.makedirs(RAW_TEMPLATE_CROP_FOLDER, exist_ok=True)
    logger.info(f"템플릿 크롭 디렉토리: {RAW_TEMPLATE_CROP_FOLDER}")

def find_feature_matches(image_path: str, template_path: str, min_matches: int = 10) -> Optional[np.ndarray]:
    """특징점 매칭으로 템플릿 위치 찾기
    
    의도: ORB 특징점을 사용하여 템플릿과 이미지 간의 매칭점을 찾아 Affine 변환 행렬 계산
    
    Args:
        image_path: 매칭할 이미지 파일 경로
        template_path: 템플릿 이미지 파일 경로
        min_matches: 최소 매칭점 개수
    
    Returns:
        Affine 변환 행렬 또는 None (매칭 실패 시)
    """
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
        
        # ORB 특징점 검출기 생성
        orb = cv2.ORB_create()
        
        # 특징점과 디스크립터 검출
        kp1, des1 = orb.detectAndCompute(image, None)
        kp2, des2 = orb.detectAndCompute(template, None)
        
        logger.info(f"이미지 특징점: {len(kp1)}개")
        logger.info(f"템플릿 특징점: {len(kp2)}개")
        
        if des1 is None or des2 is None:
            logger.warning("특징점 디스크립터를 생성할 수 없습니다.")
            return None
        
        # BFMatcher로 매칭 (ORB는 Hamming 거리 사용)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des2, des1)
        
        # 거리 기준으로 정렬
        matches = sorted(matches, key=lambda x: x.distance)
        
        logger.info(f"총 매칭점: {len(matches)}개")
        
        # 충분한 매칭점이 있는지 확인
        if len(matches) < min_matches:
            logger.warning(f"충분한 매칭점이 없습니다. (필요: {min_matches}, 발견: {len(matches)})")
            return None
        
        # 상위 매칭점들만 사용 (거리가 작은 것들)
        good_matches = matches[:min(len(matches), min_matches * 2)]
        
        logger.info(f"사용할 매칭점: {len(good_matches)}개")
        
        # Affine 변환 행렬 계산 (4자유도: 이동, 회전, 균일 스케일)
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        
        if M is None:
            logger.warning("Affine 변환 행렬을 계산할 수 없습니다.")
            return None
        
        logger.info(f"Affine 변환 성공: {len(good_matches)}개 매칭점 사용")
        logger.info(f"변환 행렬:\n{M}")
        return M
        
    except Exception as e:
        logger.error(f"특징점 매칭 중 오류 발생: {e}")
        return None

def calculate_crop_area(affine_matrix: np.ndarray, template_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Affine 변환 행렬을 기반으로 크롭 영역 계산
    
    Args:
        affine_matrix: 2x3 Affine 변환 행렬
        template_size: 템플릿 크기 (width, height)
    
    Returns:
        크롭 영역 (x, y, width, height)
    """
    try:
        # 템플릿 크기
        templ_width, templ_height = template_size
        
        # Affine 변환 행렬에서 스케일과 이동값 추출
        # M = [[a, -b, tx],
        #      [b,  a, ty]]
        # 여기서 a = scale*cos(θ), b = scale*sin(θ)
        # 스케일 = sqrt(a² + b²)
        
        a = affine_matrix[0, 0]
        b = affine_matrix[1, 0]
        tx = affine_matrix[0, 2]
        ty = affine_matrix[1, 2]
        
        # 스케일 계산
        scale = np.sqrt(a*a + b*b)
        
        logger.info(f"Affine 변환 파라미터:")
        logger.info(f"  스케일: {scale:.3f}")
        logger.info(f"  이동: ({tx:.1f}, {ty:.1f})")
        
        # 스케일이 적용된 템플릿 크기
        scaled_width = int(templ_width * scale)
        scaled_height = int(templ_height * scale)
        
        # 크롭 영역 계산 (중심점 기준)
        crop_x = int(tx - scaled_width // 2)
        crop_y = int(ty - scaled_height // 2)
        
        logger.info(f"스케일 적용된 템플릿 크기: {scaled_width}x{scaled_height}")
        logger.info(f"크롭 영역: ({crop_x}, {crop_y}, {scaled_width}, {scaled_height})")
        
        return (crop_x, crop_y, scaled_width, scaled_height)
        
    except Exception as e:
        logger.error(f"크롭 영역 계산 중 오류 발생: {e}")
        return (0, 0, 0, 0)

def crop_and_save_template(image_path: str, crop_area: Tuple[int, int, int, int], 
                          filename: str, output_dir: str) -> bool:
    """계산된 영역을 크롭하여 저장"""
    try:
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"이미지를 다시 로드할 수 없습니다: {image_path}")
            return False
        
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
            return False
        
        # 크롭
        cropped_image = image[y:y+h, x:x+w]
        
        # 파일명 생성
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 저장
        cv2.imwrite(output_path, cropped_image)
        logger.info(f"✅ 템플릿 크롭 저장: {output_path}")
        logger.info(f"   크롭된 이미지 크기: {cropped_image.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"템플릿 크롭 저장 중 오류: {e}")
        return False

def process_case3_template_matching(exception_files: List[str], min_matches: int = 10) -> List[str]:
    """case3 템플릿 매칭 처리
    
    의도: case2를 통과하지 못한 예외 파일들을 템플릿 매칭으로 처리하여 크롭된 이미지 생성
    
    Args:
        exception_files: 처리할 예외 파일명 리스트
        min_matches: 최소 매칭점 개수
    
    Returns:
        성공적으로 처리된 파일명 리스트
    """
    try:
        # 템플릿 경로 설정
        template_path = os.path.join(DATA_FOLDER, "template_img", "case3_03.png")
        
        if not os.path.exists(template_path):
            logger.error(f"템플릿 파일이 존재하지 않습니다: {template_path}")
            return []
        
        # 출력 디렉토리 설정
        setup_template_crop_directory()
        
        logger.info(f"처리할 예외 파일 수: {len(exception_files)}")
        logger.info(f"템플릿 파일: {template_path}")
        logger.info(f"최소 매칭점: {min_matches}")
        
        success_files = []
        fail_files = []
        
        # 각 예외 파일 처리
        for filename in exception_files:
            try:
                logger.info(f"처리 중: {filename}")
                
                # 이미지 경로 (raw_img에서 찾기)
                image_path = os.path.join(RAW_IMG_FOLDER, f"{filename}.png")
                if not os.path.exists(image_path):
                    # 다른 확장자 시도
                    for ext in ['.jpg', '.jpeg', '.bmp', '.tiff']:
                        alt_path = os.path.join(RAW_IMG_FOLDER, f"{filename}{ext}")
                        if os.path.exists(alt_path):
                            image_path = alt_path
                            break
                    else:
                        logger.warning(f"이미지 파일을 찾을 수 없습니다: {filename}")
                        fail_files.append(filename)
                        continue
                
                # 특징점 매칭
                affine_matrix = find_feature_matches(image_path, template_path, min_matches)
                
                if affine_matrix is not None:
                    # 템플릿 크기 가져오기
                    template = cv2.imread(template_path)
                    if template is not None:
                        templ_height, templ_width = template.shape[:2]
                        template_size = (templ_width, templ_height)
                        
                        # 크롭 영역 계산
                        crop_area = calculate_crop_area(affine_matrix, template_size)
                        
                        if crop_area[2] > 0 and crop_area[3] > 0:  # 유효한 크기인지 확인
                            # 크롭 및 저장
                            if crop_and_save_template(image_path, crop_area, filename, RAW_TEMPLATE_CROP_FOLDER):
                                success_files.append(filename)
                            else:
                                fail_files.append(filename)
                        else:
                            logger.warning(f"{filename}: 유효하지 않은 크롭 영역")
                            fail_files.append(filename)
                    else:
                        logger.warning(f"{filename}: 템플릿을 로드할 수 없습니다")
                        fail_files.append(filename)
                else:
                    logger.warning(f"{filename}: 특징점 매칭 실패")
                    fail_files.append(filename)
                
            except Exception as e:
                logger.error(f"{filename} 처리 중 오류: {e}")
                fail_files.append(filename)
                continue
        
        # 처리 결과 요약
        logger.info("=" * 60)
        logger.info("case3 템플릿 매칭 처리 결과 요약")
        logger.info("=" * 60)
        logger.info(f"총 처리 파일: {len(exception_files)}")
        logger.info(f"성공: {len(success_files)}")
        logger.info(f"실패: {len(fail_files)}")
        logger.info(f"성공률: {(len(success_files) / len(exception_files) * 100):.1f}%")
        
        return success_files
        
    except Exception as e:
        logger.error(f"case3 템플릿 매칭 처리 중 오류: {e}")
        return []
