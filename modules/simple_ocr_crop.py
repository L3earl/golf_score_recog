"""
case3 OCR 기반 테이블 크롭 모듈

의도: case2를 통과하지 못한 예외 파일들을 EasyOCR을 사용하여 테이블 영역을 크롭
- raw_img에서 예외 파일들을 가져와서 EasyOCR로 HOLE과 T 문자를 찾아 테이블 영역 크롭
- 크롭된 이미지를 raw_table_crop 폴더에 저장하여 후속 처리 준비
"""

import os
import cv2
import numpy as np
import easyocr
import logging
from typing import List, Tuple, Optional, Dict, Any
from config import RAW_IMG_FOLDER, RAW_TABLE_CROP_FOLDER, IMAGE_EXTENSIONS

# 로깅 설정
logger = logging.getLogger(__name__)

def setup_table_crop_directory():
    """raw_table_crop 디렉토리 설정
    
    의도: OCR 크롭 결과를 저장할 디렉토리를 안전하게 생성
    """
    os.makedirs(RAW_TABLE_CROP_FOLDER, exist_ok=True)
    logger.info(f"테이블 크롭 디렉토리: {RAW_TABLE_CROP_FOLDER}")

def extract_text_with_easyocr(image_path: str, reader: easyocr.Reader) -> Optional[List[Tuple]]:
    """EasyOCR을 사용하여 이미지에서 텍스트 추출
    
    Args:
        image_path: 이미지 파일 경로
        reader: EasyOCR 리더 객체
    
    Returns:
        OCR 결과 리스트 또는 None (실패 시)
    """
    try:
        logger.info(f"텍스트 추출 중: {os.path.basename(image_path)}")
        
        # EasyOCR로 텍스트 추출
        results = reader.readtext(image_path)
        
        if not results:
            logger.warning("텍스트를 찾을 수 없습니다.")
            return None
        
        logger.info(f"추출된 텍스트 ({len(results)}개)")
        
        for i, (bbox, text, confidence) in enumerate(results, 1):
            logger.debug(f"{i:2d}. [{confidence:.3f}] {text}")
        
        return results
        
    except Exception as e:
        logger.error(f"텍스트 추출 중 오류 발생: {e}")
        return None

def find_hole_and_t_coordinates(results: List[Tuple]) -> Tuple[List[Dict], List[Dict]]:
    """OCR 결과에서 HOLE과 T 문자 좌표 찾기
    
    Args:
        results: EasyOCR 결과 리스트
    
    Returns:
        (holes, valid_hole_t_pairs) 튜플
    """
    holes = []
    t_chars = []
    
    # 모든 텍스트 요소를 수집
    all_texts = []
    for bbox, text, confidence in results:
        text_upper = text.upper().strip()
        center_x = (bbox[0][0] + bbox[2][0]) / 2
        center_y = (bbox[0][1] + bbox[2][1]) / 2
        
        all_texts.append({
            'text': text,
            'text_upper': text_upper,
            'center': (center_x, center_y),
            'bbox': bbox,
            'confidence': confidence
        })
    
    # HOLE과 T 문자 찾기
    for text_info in all_texts:
        text_upper = text_info['text_upper']
        
        # HOLE 문자 찾기 (혼자 있거나 문자열 내에 있거나)
        if 'HOLE' in text_upper:
            holes.append(text_info)
        
        # T 문자 찾기 (혼자 있거나 문자열 내에 있거나)
        elif 'T' in text_upper:
            t_chars.append(text_info)
    
    # HOLE들을 Y 좌표 순으로 정렬 (위에서 아래로)
    holes.sort(key=lambda x: x['center'][1])
    
    # 각 HOLE에 대해 가장 가까운 T를 찾고, 사이의 문자들을 한줄로 나열해서 확인
    valid_hole_t_pairs = []
    
    for hole in holes:
        hole_x, hole_y = hole['center']
        
        # 같은 Y 좌표 근처의 T 문자들 찾기 (Y 좌표 차이가 작은 것)
        nearby_ts = []
        for t_char in t_chars:
            t_x, t_y = t_char['center']
            y_diff = abs(t_y - hole_y)
            if y_diff < 100:  # Y 좌표 차이가 100픽셀 이내
                nearby_ts.append((t_char, y_diff))
        
        if not nearby_ts:
            continue
            
        # 가장 가까운 T 선택
        closest_t = min(nearby_ts, key=lambda x: x[1])[0]
        
        # HOLE과 T 사이의 모든 텍스트 요소들 찾기
        hole_x, hole_y = hole['center']
        t_x, t_y = closest_t['center']
        
        # HOLE과 T 사이의 텍스트 요소들 찾기 (X 좌표 기준)
        between_texts = []
        for text_info in all_texts:
            text_x, text_y = text_info['center']
            
            # HOLE과 T 사이에 있는지 확인 (X 좌표 기준)
            if min(hole_x, t_x) <= text_x <= max(hole_x, t_x):
                # Y 좌표도 비슷한 범위에 있는지 확인
                if abs(text_y - hole_y) < 100:
                    between_texts.append(text_info)
        
        # 사이의 텍스트들을 X 좌표 순으로 정렬
        between_texts.sort(key=lambda x: x['center'][0])
        
        # 모든 텍스트를 한줄로 합치기
        combined_text = ""
        for text_info in between_texts:
            combined_text += text_info['text'] + " "
        
        combined_text = combined_text.strip()
        
        # 합친 문자열에서 1~9 숫자 개수 확인
        digit_count = 0
        for char in combined_text:
            if char.isdigit() and '1' <= char <= '9':
                digit_count += 1
        
        # 숫자가 4개 이상 있는 경우만 유효한 쌍으로 추가
        if digit_count >= 4:
            valid_hole_t_pairs.append({
                'hole': hole,
                't_char': closest_t,
                'digit_count': digit_count,
                'combined_text': combined_text,
                'between_texts': between_texts
            })
    
    return holes, valid_hole_t_pairs

def calculate_crop_dimensions(first_hole: Dict, second_hole: Optional[Dict], t_char: Dict) -> Tuple[Optional[int], Optional[int]]:
    """크롭할 영역의 너비와 높이 계산
    
    Args:
        first_hole: 첫번째 HOLE 정보
        second_hole: 두번째 HOLE 정보 (없으면 None)
        t_char: T 문자 정보
    
    Returns:
        (width, height) 튜플 또는 (None, None) (실패 시)
    """
    if not first_hole or not t_char:
        return None, None
    
    # 너비: 첫번째 HOLE의 시작점과 T 문자 사이의 거리
    hole_bbox = first_hole['bbox']
    hole_start_x = hole_bbox[0][0]  # HOLE의 왼쪽 시작 X 좌표
    hole_y = first_hole['center'][1]  # HOLE의 Y는 중앙값 사용
    
    # T 문자의 오른쪽 끝 X 좌표 사용
    t_bbox = t_char['bbox']
    t_right_x = t_bbox[2][0]  # bbox의 오른쪽 끝 X 좌표
    t_y = (t_bbox[0][1] + t_bbox[2][1]) / 2  # T의 Y는 중앙값 사용
    
    width = abs(t_right_x - hole_start_x)*27/26
    
    # 높이: 첫번째와 두번째 HOLE 사이의 거리 * 2
    height = None
    if second_hole:
        second_hole_x, second_hole_y = second_hole['center']
        height = abs(second_hole_y - hole_y) * 2  # 높이 * 2
    else:
        # 두번째 HOLE이 없으면 첫번째 HOLE의 높이를 기준으로 추정
        hole_bbox = first_hole['bbox']
        hole_height = abs(hole_bbox[2][1] - hole_bbox[0][1])
        height = hole_height * 4  # 추정값 * 4
    
    return width, height

def crop_and_save_table(image_path: str, holes: List[Dict], valid_pairs: List[Dict], 
                       filename: str, output_dir: str, target_width: int = 1000) -> bool:
    """골프 스코어 영역 크롭 및 저장
    
    Args:
        image_path: 원본 이미지 경로
        holes: 발견된 HOLE 리스트
        valid_pairs: 유효한 HOLE-T 쌍 리스트
        filename: 파일명
        output_dir: 출력 디렉토리
        target_width: 목표 너비 (기본값: 1000)
    
    Returns:
        성공 여부
    """
    try:
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"이미지를 로드할 수 없습니다: {image_path}")
            return False
        
        # 유효한 HOLE-T 쌍이 있는지 확인
        if not valid_pairs:
            logger.warning("유효한 HOLE-T 쌍을 찾을 수 없습니다.")
            return False
        
        # 첫번째 유효한 쌍 사용
        first_pair = valid_pairs[0]
        first_hole = first_pair['hole']
        first_t = first_pair['t_char']
        
        # 두번째 쌍이 있으면 사용, 없으면 None
        second_pair = valid_pairs[1] if len(valid_pairs) > 1 else None
        second_hole = second_pair['hole'] if second_pair else None
        
        # 크롭 영역 계산
        width, height = calculate_crop_dimensions(first_hole, second_hole, first_t)
        
        if width is None or height is None or width <= 0 or height <= 0:
            logger.warning("크롭 영역을 계산할 수 없습니다.")
            return False
        
        # 첫번째 HOLE의 좌상단 좌표 계산
        hole_bbox = first_hole['bbox']
        start_x = int(hole_bbox[0][0])  # 좌상단 x
        start_y = int(hole_bbox[0][1])  # 좌상단 y
        
        # 크롭 영역 계산 (정수형으로 명시적 변환)
        end_x = int(start_x + int(width))
        end_y = int(start_y + int(height))
        
        # 이미지 경계 확인
        img_height, img_width = image.shape[:2]
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(img_width, end_x)
        end_y = min(img_height, end_y)
        
        logger.info(f"크롭 영역: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
        logger.info(f"크기: {end_x - start_x} x {end_y - start_y}")
        
        # 이미지 크롭
        cropped = image[start_y:end_y, start_x:end_x]
        
        if cropped.size == 0:
            logger.warning("잘못된 크롭 영역입니다.")
            return False
        
        # 가로 크기를 동일하게 맞추기
        if target_width is not None:
            current_height, current_width = cropped.shape[:2]
            # 비율을 유지하면서 리사이즈
            aspect_ratio = current_height / current_width
            new_height = int(target_width * aspect_ratio)
            
            # 정수형으로 명시적 변환
            target_width_int = int(target_width)
            new_height_int = int(new_height)
            
            # 크기가 유효한지 확인
            if target_width_int > 0 and new_height_int > 0:
                cropped = cv2.resize(cropped, (target_width_int, new_height_int))
                logger.info(f"리사이즈: {current_width}x{current_height} -> {target_width_int}x{new_height_int}")
            else:
                logger.warning(f"잘못된 리사이즈 크기: {target_width_int}x{new_height_int}")
        
        # 파일명 생성 및 저장
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, cropped)
        logger.info(f"✅ 테이블 크롭 저장: {output_path}")
        logger.info(f"   크롭된 이미지 크기: {cropped.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"테이블 크롭 저장 중 오류: {e}")
        return False

def process_case3_ocr_crop(exception_files: List[str], target_width: int = 1000) -> List[str]:
    """case3 OCR 기반 테이블 크롭 처리
    
    의도: case2를 통과하지 못한 예외 파일들을 EasyOCR을 사용하여 테이블 영역을 크롭
    
    Args:
        exception_files: 처리할 예외 파일명 리스트
        target_width: 목표 너비 (기본값: 1000)
    
    Returns:
        성공적으로 처리된 파일명 리스트
    """
    try:
        # 출력 디렉토리 설정
        setup_table_crop_directory()
        
        logger.info(f"처리할 예외 파일 수: {len(exception_files)}")
        logger.info(f"목표 너비: {target_width}px")
        
        # EasyOCR 리더 초기화 (한국어, 영어 지원)
        logger.info("EasyOCR 리더 초기화 중...")
        try:
            reader = easyocr.Reader(['ko', 'en'])
            logger.info("EasyOCR 리더 초기화 완료")
        except Exception as e:
            logger.error(f"EasyOCR 리더 초기화 실패: {e}")
            return []
        
        success_files = []
        fail_files = []
        
        # 각 예외 파일 처리
        for filename in exception_files:
            try:
                logger.info(f"처리 중: {filename}")
                
                # 이미지 경로 (raw_img에서 찾기)
                image_path = None
                for ext in IMAGE_EXTENSIONS:
                    alt_path = os.path.join(RAW_IMG_FOLDER, f"{filename}{ext}")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
                
                if not image_path:
                    logger.warning(f"이미지 파일을 찾을 수 없습니다: {filename}")
                    fail_files.append(filename)
                    continue
                
                # 텍스트 추출
                results = extract_text_with_easyocr(image_path, reader)
                
                if results is None:
                    logger.warning(f"{filename}: 텍스트 추출 실패")
                    fail_files.append(filename)
                    continue
                
                # HOLE과 T 문자 좌표 찾기
                holes, valid_pairs = find_hole_and_t_coordinates(results)
                
                logger.info(f"발견된 HOLE: {len(holes)}개")
                logger.info(f"유효한 HOLE-T 쌍: {len(valid_pairs)}개")
                
                # 골프 스코어 영역 크롭
                if crop_and_save_table(image_path, holes, valid_pairs, filename, RAW_TABLE_CROP_FOLDER, target_width):
                    success_files.append(filename)
                else:
                    fail_files.append(filename)
                
            except Exception as e:
                logger.error(f"{filename} 처리 중 오류: {e}")
                fail_files.append(filename)
                continue
        
        # 처리 결과 요약
        logger.info("=" * 60)
        logger.info("case3 OCR 크롭 처리 결과 요약")
        logger.info("=" * 60)
        logger.info(f"총 처리 파일: {len(exception_files)}")
        logger.info(f"성공: {len(success_files)}")
        logger.info(f"실패: {len(fail_files)}")
        logger.info(f"성공률: {(len(success_files) / len(exception_files) * 100):.1f}%")
        
        return success_files
        
    except Exception as e:
        logger.error(f"case3 OCR 크롭 처리 중 오류: {e}")
        return []
