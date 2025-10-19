"""
EasyOCR 기반 텍스트 인식 및 골프 스코어 영역 크롭 테스트
- data/raw_img 폴더의 이미지들을 가져와서
- EasyOCR로 텍스트 데이터를 추출하여 HOLE과 T 문자 좌표 찾기
- 첫번째 HOLE과 T 문자 사이 거리를 너비로, 첫번째와 두번째 HOLE 사이 거리를 높이로 설정
- 해당 영역을 크롭해서 test 폴더에 저장
"""

import os
import glob
import cv2
import numpy as np
import easyocr

def create_test_folder():
    """test 폴더 생성"""
    test_dir = "test"
    os.makedirs(test_dir, exist_ok=True)
    print(f"✅ test 폴더 생성: {test_dir}")
    return test_dir

def get_image_files():
    """raw_img 폴더의 이미지 파일 목록 반환"""
    raw_img_dir = "data/raw_img"
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(raw_img_dir, ext)
        image_files.extend(glob.glob(pattern))
    
    print(f"📁 발견된 이미지 파일: {len(image_files)}개")
    for img in image_files:
        print(f"   - {os.path.basename(img)}")
    
    return image_files

def find_hole_and_t_coordinates(results, all_texts):
    """OCR 결과에서 HOLE과 T 문자 좌표 찾기"""
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
            print(f"   🚫 HOLE 1과 연관된 그룹 제외: {[text['text'] for text in group]}")
        
        # HOLE 2와 Y좌표가 겹치는지 확인
        elif (max(group_y_top, second_hole_y_top) <= min(group_y_bottom, second_hole_y_bottom)):
            is_hole_related = True
            print(f"   🚫 HOLE 2와 연관된 그룹 제외: {[text['text'] for text in group]}")
        
        # HOLE 텍스트가 직접 포함된 그룹인지 확인
        else:
            for text_info in group:
                if 'HOLE' in text_info['text_upper']:
                    is_hole_related = True
                    print(f"   🚫 HOLE 텍스트 포함 그룹 제외: {[text['text'] for text in group]}")
                    break
        
        if is_hole_related:
            hole_groups_count += 1
        else:
            filtered_groups.append(group)
    
    print(f"   📊 필터링 결과: 전체 {len(crop_groups)}개 그룹 → HOLE 제외 후 {len(filtered_groups)}개 그룹 (제외된 HOLE 그룹: {hole_groups_count}개)")
    
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

def calculate_crop_dimensions(first_hole, second_hole, t_char):
    """크롭할 영역의 너비와 높이 계산"""
    if not first_hole or not t_char:
        return None, None
    
    # 너비: 첫번째 HOLE과 T 문자 사이의 거리
    hole_x, hole_y = first_hole['center']
    
    # T 문자의 오른쪽 끝 X 좌표 사용
    t_bbox = t_char['bbox']
    t_right_x = t_bbox[2][0]  # bbox의 오른쪽 끝 X 좌표
    t_y = (t_bbox[0][1] + t_bbox[2][1]) / 2  # T의 Y는 중앙값 사용
    
    width = abs(t_right_x - hole_x)
    
    # 높이: 첫번째와 두번째 HOLE 사이의 거리 * 2
    height = None
    if second_hole:
        second_hole_x, second_hole_y = second_hole['center']
        height = abs(second_hole_y - hole_y) * 2  # 높이 * 2
    else:
        # 두번째 HOLE이 없으면 첫번째 HOLE의 높이를 기준으로 추정
        hole_bbox = first_hole['bbox']
        hole_height = abs(hole_bbox[2][1] - hole_bbox[0][1])
        height = hole_height * 4  # 추정값 * 2
    
    return width, height

def crop_golf_score_area(image_path, holes, valid_pairs, all_texts, output_dir, target_width=None):
    """골프 스코어 영역 크롭 및 저장 (HOLE 1-2 사이 + HOLE 2 뒤쪽 그룹)"""
    try:
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            return False
        
        # 유효한 HOLE-T 쌍이 있는지 확인
        if not valid_pairs:
            print("   ⚠️ 유효한 HOLE-T 쌍을 찾을 수 없습니다.")
            return False
        
        # 첫번째 유효한 쌍 사용
        first_pair = valid_pairs[0]
        first_hole = first_pair['hole']
        first_t = first_pair['t_char']
        
        # 너비 계산 (HOLE과 T 사이의 거리)
        width, _ = calculate_crop_dimensions(first_hole, None, first_t)
        
        if width is None or width <= 0:
            print("   ⚠️ 크롭 영역을 계산할 수 없습니다.")
            return False
        
        # HOLE 1-2 사이의 텍스트들을 Y좌표 범위별로 그룹화
        crop_groups_between = get_crop_groups_from_holes(holes, all_texts)
        
        if not crop_groups_between:
            print("   ⚠️ HOLE 1-2 사이의 크롭할 그룹을 찾을 수 없습니다.")
            return False
        
        print(f"   🔍 HOLE 1-2 사이 발견된 크롭 그룹: {len(crop_groups_between)}개")
        
        # 디버깅: 각 그룹의 내용 확인
        for i, group in enumerate(crop_groups_between):
            print(f"   🔍 HOLE 1-2 사이 그룹 {i+1} 내용:")
            for text_info in group:
                has_hole = 'HOLE' in text_info['text_upper']
                print(f"      - '{text_info['text']}' (HOLE 포함: {has_hole})")
        
        # HOLE 2 뒤쪽의 텍스트들을 Y좌표 범위별로 그룹화 (동일한 개수만큼)
        crop_groups_after = get_crop_groups_after_second_hole(holes, all_texts, len(crop_groups_between))
        
        print(f"   🔍 HOLE 2 뒤쪽 발견된 크롭 그룹: {len(crop_groups_after)}개")
        
        # 디버깅: HOLE 2 뒤쪽 각 그룹의 내용 확인
        for i, group in enumerate(crop_groups_after):
            print(f"   🔍 HOLE 2 뒤쪽 그룹 {i+1} 내용:")
            for text_info in group:
                has_hole = 'HOLE' in text_info['text_upper']
                print(f"      - '{text_info['text']}' (HOLE 포함: {has_hole})")
        
        # 각 그룹마다 크롭 수행
        success_count = 0
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # HOLE 1-2 사이 그룹들 크롭
        for i, group in enumerate(crop_groups_between):
            print(f"   📐 HOLE 1-2 사이 그룹 {i+1} 크롭 중...")
            
            if crop_single_group(image, group, first_hole, width, target_width, 
                               output_dir, f"{base_name}_between_group_{i+1}.png"):
                success_count += 1
        
        # HOLE 2 뒤쪽 그룹들 크롭
        for i, group in enumerate(crop_groups_after):
            print(f"   📐 HOLE 2 뒤쪽 그룹 {i+1} 크롭 중...")
            
            if crop_single_group(image, group, first_hole, width, target_width, 
                               output_dir, f"{base_name}_after_group_{i+1}.png"):
                success_count += 1
        
        print(f"   📊 총 {success_count}개 그룹 크롭 완료")
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 크롭 처리 중 오류 발생: {e}")
        return False

def crop_single_group(image, group, first_hole, width, target_width, output_dir, filename):
    """단일 그룹을 크롭하는 헬퍼 함수"""
    try:
        # 그룹에서 가장 위쪽과 아래쪽 텍스트의 Y좌표 범위 계산
        group.sort(key=lambda x: x['y_top'])
        top_text = group[0]
        bottom_text = group[-1]
        
        # 크롭할 Y 범위 계산
        start_y = int(top_text['y_top'])
        end_y = int(bottom_text['y_bottom'])
        
        # 첫번째 HOLE의 X 좌표를 시작점으로 사용
        hole_bbox = first_hole['bbox']
        start_x = int(hole_bbox[0][0])  # 좌상단 x
        end_x = int(start_x + int(width))
        
        # 이미지 경계 확인
        img_height, img_width = image.shape[:2]
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(img_width, end_x)
        end_y = min(img_height, end_y)
        
        print(f"      📐 크롭 영역: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
        print(f"      📏 크기: {end_x - start_x} x {end_y - start_y}")
        
        # 이미지 크롭
        cropped = image[start_y:end_y, start_x:end_x]
        
        if cropped.size == 0:
            print(f"      ⚠️ 잘못된 크롭 영역입니다.")
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
                print(f"      🔄 리사이즈: {current_width}x{current_height} -> {target_width_int}x{new_height_int}")
            else:
                print(f"      ⚠️ 잘못된 리사이즈 크기: {target_width_int}x{new_height_int}")
        
        # 파일 저장
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cropped)
        print(f"      ✅ 저장 완료: {filename}")
        
        return True
        
    except Exception as e:
        print(f"      ❌ 그룹 크롭 중 오류 발생: {e}")
        return False

def extract_text_with_easyocr(image_path, reader):
    """EasyOCR을 사용하여 이미지에서 텍스트 추출"""
    try:
        print(f"\n🔍 텍스트 추출 중: {os.path.basename(image_path)}")
        
        # EasyOCR로 텍스트 추출
        results = reader.readtext(image_path)
        
        if not results:
            print("   ⚠️ 텍스트를 찾을 수 없습니다.")
            return None, None
        
        print(f"   📝 추출된 텍스트 ({len(results)}개):")
        
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
            print(f"   {i:2d}. [{confidence:.3f}] {text}")
        
        return results, all_texts
        
    except Exception as e:
        print(f"❌ 텍스트 추출 중 오류 발생: {e}")
        return None, None

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("EasyOCR 기반 골프 스코어 영역 크롭 테스트 시작")
    print("=" * 60)
    
    # test 폴더 생성
    test_dir = create_test_folder()
    
    # EasyOCR 리더 초기화 (한국어, 영어 지원)
    print("🔄 EasyOCR 리더 초기화 중...")
    try:
        reader = easyocr.Reader(['ko', 'en'])
        print("✅ EasyOCR 리더 초기화 완료")
    except Exception as e:
        print(f"❌ EasyOCR 리더 초기화 실패: {e}")
        return
    
    # 이미지 파일 목록 가져오기
    image_files = get_image_files()
    
    if not image_files:
        print("❌ 처리할 이미지 파일이 없습니다.")
        return
    
    # 각 이미지에 대해 텍스트 추출 및 크롭 수행
    success_count = 0
    total_count = len(image_files)
    crop_widths = []  # 모든 크롭된 이미지의 너비 저장
    
    print("\n" + "=" * 60)
    print("골프 스코어 영역 크롭 처리 시작")
    print("=" * 60)
    
    # 1단계: 모든 이미지를 크롭하고 너비 수집
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{total_count}] 처리 중: {os.path.basename(image_path)}")
        
        # 텍스트 추출
        results, all_texts = extract_text_with_easyocr(image_path, reader)
        
        if results is None or all_texts is None:
            continue
        
        # HOLE과 T 문자 좌표 찾기
        holes, valid_pairs = find_hole_and_t_coordinates(results, all_texts)
        
        print(f"   🔍 발견된 HOLE: {len(holes)}개")
        for j, hole in enumerate(holes):
            print(f"      {j+1}. {hole['text']} at ({hole['center'][0]:.1f}, {hole['center'][1]:.1f})")
        
        print(f"   🔍 유효한 HOLE-T 쌍: {len(valid_pairs)}개")
        for j, pair in enumerate(valid_pairs):
            hole = pair['hole']
            t_char = pair['t_char']
            digit_count = pair['digit_count']
            combined_text = pair['combined_text']
            print(f"      {j+1}. HOLE '{hole['text']}' -> T '{t_char['text']}' (사이 문자열: '{combined_text}', 숫자: {digit_count}개)")
        
        # 골프 스코어 영역 크롭 (임시로 너비만 수집)
        if valid_pairs:
            first_pair = valid_pairs[0]
            first_hole = first_pair['hole']
            first_t = first_pair['t_char']
            
            width, _ = calculate_crop_dimensions(first_hole, None, first_t)
            if width is not None:
                crop_widths.append(width)
                success_count += 1
    
    # 2단계: 고정 너비 설정
    if crop_widths:
        target_width = 1000  # 고정 너비 1000px
        print(f"\n📏 모든 이미지의 가로 크기를 {target_width}px로 통일합니다.")
        
        # 3단계: 모든 이미지를 동일한 너비로 리사이즈하여 다시 저장
        print("\n" + "=" * 60)
        print("가로 크기 통일 처리")
        print("=" * 60)
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{total_count}] 리사이즈 중: {os.path.basename(image_path)}")
            
            # 텍스트 추출
            results, all_texts = extract_text_with_easyocr(image_path, reader)
            
            if results is None or all_texts is None:
                continue
            
            # HOLE과 T 문자 좌표 찾기
            holes, valid_pairs = find_hole_and_t_coordinates(results, all_texts)
            
            # 골프 스코어 영역 크롭 (동일한 너비로 리사이즈)
            if crop_golf_score_area(image_path, holes, valid_pairs, all_texts, test_dir, target_width=target_width):
                pass  # 이미 success_count는 위에서 계산됨
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("처리 결과 요약")
    print("=" * 60)
    print(f"총 처리 파일: {total_count}")
    print(f"성공: {success_count}")
    print(f"실패: {total_count - success_count}")
    print(f"성공률: {(success_count / total_count * 100):.1f}%")
    print(f"결과 저장 위치: {test_dir}")

if __name__ == "__main__":
    main()
