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
from config import RAW_IMG_FOLDER, RAW_CROP_NUM_FOLDER, IMAGE_EXTENSIONS

# 로깅 설정
logger = logging.getLogger(__name__)

def setup_table_crop_directory():
    """raw_crop/case3 디렉토리 설정
    
    의도: OCR 크롭 결과를 저장할 디렉토리를 안전하게 생성
    """
    case3_crop_folder = os.path.join(RAW_CROP_NUM_FOLDER, "case3")
    os.makedirs(case3_crop_folder, exist_ok=True)
    logger.info(f"테이블 크롭 디렉토리: {case3_crop_folder}")
    return case3_crop_folder

def extract_text_with_easyocr(image_path: str, reader: easyocr.Reader) -> Optional[Tuple[List[Tuple], List[Dict]]]:
    """EasyOCR을 사용하여 이미지에서 텍스트 추출
    
    Args:
        image_path: 이미지 파일 경로
        reader: EasyOCR 리더 객체
    
    Returns:
        (results, all_texts) 튜플 또는 (None, None) (실패 시)
    """
    try:
        logger.info(f"텍스트 추출 중: {os.path.basename(image_path)}")
        
        # EasyOCR로 텍스트 추출
        results = reader.readtext(image_path)
        
        if not results:
            logger.warning("텍스트를 찾을 수 없습니다.")
            return None, None
        
        logger.info(f"추출된 텍스트 ({len(results)}개)")
        
        # 모든 텍스트 요소를 수집 (Y좌표 범위 포함)
        all_texts = []
        for bbox, text, confidence in results:
            text_upper = text.upper().strip()
            center_x = (bbox[0][0] + bbox[2][0]) / 2
            center_y = (bbox[0][1] + bbox[2][1]) / 2
            
            # Y좌표 범위 계산 (top, bottom)
            y_top = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
            y_bottom = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
            
            all_texts.append({
                'text': text,
                'text_upper': text_upper,
                'center': (center_x, center_y),
                'bbox': bbox,
                'confidence': confidence,
                'y_top': y_top,
                'y_bottom': y_bottom
            })
        
        for i, (bbox, text, confidence) in enumerate(results, 1):
            logger.debug(f"{i:2d}. [{confidence:.3f}] {text}")
        
        return results, all_texts
        
    except Exception as e:
        logger.error(f"텍스트 추출 중 오류 발생: {e}")
        return None, None

def find_hole_and_t_coordinates(results: List[Tuple], all_texts: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """OCR 결과에서 HOLE과 T 문자 좌표 찾기
    
    Args:
        results: EasyOCR 결과 리스트
        all_texts: Y좌표 범위가 포함된 텍스트 정보 리스트
    
    Returns:
        (holes, valid_hole_t_pairs) 튜플
    """
    holes = []
    t_chars = []
    
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

def group_texts_by_y_overlap(texts):
    """Y좌표 범위가 겹치는 텍스트들을 그룹화"""
    if not texts:
        return []
    
    groups = []
    
    for text in texts:
        # 현재 텍스트와 겹치는 그룹 찾기
        overlapping_groups = []
        for i, group in enumerate(groups):
            for group_text in group:
                # Y좌표 범위가 겹치는지 확인
                if (max(text['y_top'], group_text['y_top']) <= 
                    min(text['y_bottom'], group_text['y_bottom'])):
                    overlapping_groups.append(i)
                    break
        
        if overlapping_groups:
            # 겹치는 그룹들을 하나로 합치기
            merged_group = [text]
            for i in sorted(overlapping_groups, reverse=True):
                merged_group.extend(groups.pop(i))
            groups.append(merged_group)
        else:
            # 새로운 그룹 생성
            groups.append([text])
    
    return groups

def get_crop_groups_from_holes(holes, all_texts):
    """HOLE 2개 사이의 텍스트들을 Y좌표 범위별로 그룹화 (HOLE 포함 그룹 제외)"""
    if len(holes) < 2:
        return []
    
    # 첫번째와 두번째 HOLE의 Y좌표 범위
    first_hole = holes[0]
    second_hole = holes[1]
    
    # HOLE 2개 사이의 Y좌표 범위 (HOLE 제외)
    y_range_top = first_hole['y_bottom']      # 첫번째 HOLE의 하단
    y_range_bottom = second_hole['y_top']    # 두번째 HOLE의 상단
        
    # HOLE 2개 사이의 Y좌표 범위에 있는 텍스트들 필터링
    texts_in_range = []
    for text in all_texts:
        text_y_top = text['y_top']
        text_y_bottom = text['y_bottom']
        
        # 텍스트의 Y좌표 범위가 HOLE 2개 사이의 범위와 겹치는지 확인
        if (max(text_y_top, y_range_top) <= min(text_y_bottom, y_range_bottom)):
            texts_in_range.append(text)
    
    # Y좌표 범위가 겹치는 텍스트들을 그룹화
    groups = group_texts_by_y_overlap(texts_in_range)
    
    # 각 그룹에서 가장 위쪽과 아래쪽 텍스트 하나씩만 남기기
    crop_groups = []
    for group in groups:
        if len(group) == 1:
            crop_groups.append(group)
        else:
            # Y좌표 순으로 정렬
            group.sort(key=lambda x: x['y_top'])
            # 가장 위쪽과 아래쪽 하나씩만 남기기
            crop_groups.append([group[0], group[-1]])
    
    # HOLE과 직접적으로 연관된 그룹 제외
    filtered_groups = []
    hole_groups_count = 0
    
    # HOLE 1과 HOLE 2의 Y좌표 범위 계산
    first_hole_y_top = first_hole['y_top']
    first_hole_y_bottom = first_hole['y_bottom']
    second_hole_y_top = second_hole['y_top']
    second_hole_y_bottom = second_hole['y_bottom']
    
    for group in crop_groups:
        is_hole_related = False
        
        # 그룹의 Y좌표 범위 계산
        group_y_top = min(text_info['y_top'] for text_info in group)
        group_y_bottom = max(text_info['y_bottom'] for text_info in group)
        
        # HOLE 1과 Y좌표가 겹치는지 확인
        if (max(group_y_top, first_hole_y_top) <= min(group_y_bottom, first_hole_y_bottom)):
            is_hole_related = True
            logger.debug(f"HOLE 1과 연관된 그룹 제외: {[text['text'] for text in group]}")
        
        # HOLE 2와 Y좌표가 겹치는지 확인
        elif (max(group_y_top, second_hole_y_top) <= min(group_y_bottom, second_hole_y_bottom)):
            is_hole_related = True
            logger.debug(f"HOLE 2와 연관된 그룹 제외: {[text['text'] for text in group]}")
        
        # HOLE 텍스트가 직접 포함된 그룹인지 확인
        else:
            for text_info in group:
                if 'HOLE' in text_info['text_upper']:
                    is_hole_related = True
                    logger.debug(f"HOLE 텍스트 포함 그룹 제외: {[text['text'] for text in group]}")
                    break
        
        if is_hole_related:
            hole_groups_count += 1
        else:
            filtered_groups.append(group)
    
    logger.debug(f"필터링 결과: 전체 {len(crop_groups)}개 그룹 → HOLE 제외 후 {len(filtered_groups)}개 그룹 (제외된 HOLE 그룹: {hole_groups_count}개)")
    
    return filtered_groups

def get_crop_groups_after_second_hole(holes, all_texts, target_group_count):
    """HOLE 2 뒤쪽의 텍스트들을 Y좌표 범위별로 그룹화"""
    if len(holes) < 2:
        return []
    
    # 두번째 HOLE의 Y좌표 범위
    second_hole = holes[1]
    second_hole_y_top = second_hole['y_top']
    second_hole_y_bottom = second_hole['y_bottom']
    
    # HOLE 2 뒤쪽의 Y좌표 범위에 있는 텍스트들 필터링
    texts_after_second_hole = []
    for text in all_texts:
        text_y_top = text['y_top']
        text_y_bottom = text['y_bottom']
        
        # 텍스트가 HOLE 2보다 아래쪽에 있는지 확인
        if text_y_top > second_hole_y_bottom:
            texts_after_second_hole.append(text)
    
    # Y좌표 범위가 겹치는 텍스트들을 그룹화
    groups = group_texts_by_y_overlap(texts_after_second_hole)
    
    # 각 그룹에서 가장 위쪽과 아래쪽 텍스트 하나씩만 남기기
    crop_groups = []
    for group in groups:
        if len(group) == 1:
            crop_groups.append(group)
        else:
            # Y좌표 순으로 정렬
            group.sort(key=lambda x: x['y_top'])
            # 가장 위쪽과 아래쪽 하나씩만 남기기
            crop_groups.append([group[0], group[-1]])
    
    # Y좌표 순으로 정렬 (위에서 아래로)
    crop_groups.sort(key=lambda group: group[0]['y_top'])
    
    # 목표 그룹 개수만큼만 반환
    return crop_groups[:target_group_count]

def crop_single_group(image, group, first_hole, width, target_width, filename_base, filename):
    """단일 그룹을 크롭하는 헬퍼 함수"""
    try:
        # 그룹에서 가장 위쪽과 아래쪽 텍스트의 Y좌표 범위 계산
        group.sort(key=lambda x: x['y_top'])
        top_text = group[0]
        bottom_text = group[-1]
        
        # 크롭할 Y 범위 계산
        start_y = int(top_text['y_top'])
        end_y = int(bottom_text['y_bottom'])
        
        # 첫번째 HOLE의 오른쪽 끝 X 좌표를 시작점으로 사용
        hole_bbox = first_hole['bbox']
        start_x = int(max(hole_bbox[1][0], hole_bbox[2][0]))  # HOLE의 오른쪽 끝 x
        end_x = int(start_x + int(width))
        
        # 이미지 경계 확인
        img_height, img_width = image.shape[:2]
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(img_width, end_x)
        end_y = min(img_height, end_y)
        
        logger.debug(f"크롭 영역: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
        logger.debug(f"크기: {end_x - start_x} x {end_y - start_y}")
        
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
                logger.debug(f"리사이즈: {current_width}x{current_height} -> {target_width_int}x{new_height_int}")
            else:
                logger.warning(f"잘못된 리사이즈 크기: {target_width_int}x{new_height_int}")
        
        # case1, case2와 동일한 폴더 구조로 저장
        case3_crop_folder = os.path.join(RAW_CROP_NUM_FOLDER, "case3")
        filename_folder = os.path.join(case3_crop_folder, filename_base)
        os.makedirs(filename_folder, exist_ok=True)
        
        # 파일 저장 (case3_01 접두사 제거)
        output_path = os.path.join(filename_folder, filename)
        cv2.imwrite(output_path, cropped)
        logger.info(f"✅ 그룹 크롭 저장: {filename}")
        
        return True
        
    except Exception as e:
        logger.error(f"그룹 크롭 중 오류 발생: {e}")
        return False

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
    
    # 너비: 첫번째 HOLE과 T 문자 사이의 거리
    hole_x, hole_y = first_hole['center']
    
    # T 문자의 오른쪽 끝 X 좌표 사용
    t_bbox = t_char['bbox']
    t_right_x = t_bbox[2][0]  # bbox의 오른쪽 끝 X 좌표
    t_y = (t_bbox[0][1] + t_bbox[2][1]) / 2  # T의 Y는 중앙값 사용
    
    width = abs(t_right_x - hole_x)*0.87
    
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

def crop_golf_score_area(image_path, holes, valid_pairs, all_texts, target_width=None):
    """골프 스코어 영역 크롭 및 저장 (HOLE 1-2 사이 + HOLE 2 뒤쪽 그룹)"""
    try:
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"이미지 로드 실패: {image_path}")
            return False
        
        # 유효한 HOLE-T 쌍이 있는지 확인
        if not valid_pairs:
            logger.warning("유효한 HOLE-T 쌍을 찾을 수 없습니다.")
            return False
        
        # 첫번째 유효한 쌍 사용
        first_pair = valid_pairs[0]
        first_hole = first_pair['hole']
        first_t = first_pair['t_char']
        
        # 너비 계산 (HOLE과 T 사이의 거리)
        width, _ = calculate_crop_dimensions(first_hole, None, first_t)
        
        if width is None or width <= 0:
            logger.warning("크롭 영역을 계산할 수 없습니다.")
            return False
        
        # HOLE 1-2 사이의 텍스트들을 Y좌표 범위별로 그룹화
        crop_groups_between = get_crop_groups_from_holes(holes, all_texts)
        
        if not crop_groups_between:
            logger.warning("HOLE 1-2 사이의 크롭할 그룹을 찾을 수 없습니다.")
            return False
        
        logger.info(f"HOLE 1-2 사이 발견된 크롭 그룹: {len(crop_groups_between)}개")
        
        # 디버깅: 각 그룹의 내용 확인
        for i, group in enumerate(crop_groups_between):
            logger.debug(f"HOLE 1-2 사이 그룹 {i+1} 내용:")
            for text_info in group:
                has_hole = 'HOLE' in text_info['text_upper']
                logger.debug(f"  - '{text_info['text']}' (HOLE 포함: {has_hole})")
        
        # HOLE 2 뒤쪽의 텍스트들을 Y좌표 범위별로 그룹화 (동일한 개수만큼)
        crop_groups_after = get_crop_groups_after_second_hole(holes, all_texts, len(crop_groups_between))
        
        logger.info(f"HOLE 2 뒤쪽 발견된 크롭 그룹: {len(crop_groups_after)}개")
        
        # 디버깅: HOLE 2 뒤쪽 각 그룹의 내용 확인
        for i, group in enumerate(crop_groups_after):
            logger.debug(f"HOLE 2 뒤쪽 그룹 {i+1} 내용:")
            for text_info in group:
                has_hole = 'HOLE' in text_info['text_upper']
                logger.debug(f"  - '{text_info['text']}' (HOLE 포함: {has_hole})")
        
        # 각 그룹마다 크롭 수행
        success_count = 0
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # HOLE 1-2 사이 그룹들 크롭
        for i, group in enumerate(crop_groups_between):
            logger.info(f"HOLE 1-2 사이 그룹 {i+1} 크롭 중...")
            
            if crop_single_group(image, group, first_hole, width, target_width, 
                               base_name, f"between_group_{i+1}.png"):
                success_count += 1
        
        # HOLE 2 뒤쪽 그룹들 크롭
        for i, group in enumerate(crop_groups_after):
            logger.info(f"HOLE 2 뒤쪽 그룹 {i+1} 크롭 중...")
            
            if crop_single_group(image, group, first_hole, width, target_width, 
                               base_name, f"after_group_{i+1}.png"):
                success_count += 1
        
        logger.info(f"총 {success_count}개 그룹 크롭 완료")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"크롭 처리 중 오류 발생: {e}")
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
                results, all_texts = extract_text_with_easyocr(image_path, reader)
                
                if results is None or all_texts is None:
                    logger.warning(f"{filename}: 텍스트 추출 실패")
                    fail_files.append(filename)
                    continue
                
                # HOLE과 T 문자 좌표 찾기
                holes, valid_pairs = find_hole_and_t_coordinates(results, all_texts)
                
                logger.info(f"발견된 HOLE: {len(holes)}개")
                logger.info(f"유효한 HOLE-T 쌍: {len(valid_pairs)}개")
                
                # 골프 스코어 영역 크롭 (그룹화 방식)
                if crop_golf_score_area(image_path, holes, valid_pairs, all_texts, target_width):
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
