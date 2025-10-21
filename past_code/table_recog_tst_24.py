"""
EasyOCR 기반 텍스트 인식 및 골프 스코어 영역 크롭 테스트 (숫자가 아닌 문자 기준)
- data/raw_img 폴더의 이미지들을 가져와서
- EasyOCR로 텍스트 데이터를 추출하여 HOLE과 T 문자 좌표 찾기
- 첫번째 HOLE과 T 문자 사이 거리를 너비로 설정
- T 이후 텍스트에서 숫자가 아닌 문자를 찾아서 각각의 Y좌표 범위로 높이 계산
- 각 숫자가 아닌 문자마다 개별적으로 crop하여 여러 개의 이미지로 저장
- 중복된 좌표는 제거하여 불필요한 crop 방지
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

def find_hole_and_t_coordinates(results):
    """OCR 결과에서 HOLE과 T 문자 좌표 찾기"""
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

def find_non_digit_characters_after_t(valid_pairs, all_texts):
    """T 이후 텍스트에서 숫자가 아닌 문자들의 좌표 찾기"""
    print(f"   📋 find_non_digit_characters_after_t 시작")
    print(f"      - valid_pairs 개수: {len(valid_pairs) if valid_pairs else 0}")
    print(f"      - all_texts 개수: {len(all_texts) if all_texts else 0}")
    
    if not valid_pairs:
        print("   ❌ valid_pairs가 비어있습니다.")
        return []
    
    # 첫번째 유효한 쌍 사용
    first_pair = valid_pairs[0]
    first_t = first_pair['t_char']
    
    # T의 좌표
    t_x, t_y = first_t['center']
    print(f"   📍 T 문자 좌표: ({t_x:.1f}, {t_y:.1f})")
    
    # all_texts를 X 좌표 순으로 정렬
    sorted_texts = sorted(all_texts, key=lambda x: x['center'][0])
    print(f"   📊 정렬된 텍스트 개수: {len(sorted_texts)}")
    
    # 정렬된 텍스트들 출력 (디버깅용)
    print(f"   📝 정렬된 텍스트 목록:")
    for i, text_info in enumerate(sorted_texts):
        print(f"      {i:2d}. '{text_info['text']}' at ({text_info['center'][0]:.1f}, {text_info['center'][1]:.1f})")
    
    # T를 찾기 (좌표로 비교)
    t_found = False
    t_index = -1
    print(f"   🔍 T 문자 검색 시작...")
    
    for i, text_info in enumerate(sorted_texts):
        text_x, text_y = text_info['center']
        x_diff = abs(text_x - t_x)
        y_diff = abs(text_y - t_y)
        
        print(f"      [{i:2d}] '{text_info['text']}' - X차이: {x_diff:.1f}, Y차이: {y_diff:.1f}")
        
        # T와 동일한 텍스트 찾기 (좌표로 비교)
        if x_diff < 10 and y_diff < 10:
            t_found = True
            t_index = i
            print(f"   ✅ T 문자 발견! (인덱스 {i}): '{text_info['text']}' at ({text_info['center'][0]:.1f}, {text_info['center'][1]:.1f})")
            break
    
    if not t_found:
        print("   ❌ T 문자를 찾을 수 없습니다.")
        return []
    
    # T 이후의 텍스트들에서 숫자가 아닌 문자 찾기
    non_digit_chars = []
    print(f"   🔍 T 이후 텍스트 검색 시작 (인덱스 {t_index + 1}부터)...")
    
    # T 이후의 텍스트들만 순회 (t_index + 1부터)
    for i, text_info in enumerate(sorted_texts[t_index + 1:], start=t_index + 1):
        text_x, text_y = text_info['center']
        text = text_info['text'].strip()
        y_diff = abs(text_y - t_y)
        
        print(f"      [{i:2d}] '{text}' at ({text_x:.1f}, {text_y:.1f}) - Y차이: {y_diff:.1f}")
        
        # 모든 텍스트 확인 (Y 좌표 제한 없음)
        print(f"         ✅ 텍스트 확인")
        
        # 숫자가 아닌 문자 찾기
        print(f"         🔍 문자별 분석:")
        for j, char in enumerate(text):
            is_alpha = char.isalpha()
            is_special = char in ['-', '+', '=', '*', '/', '(', ')', '[', ']', '{', '}', '!', '@', '#', '$', '%', '^', '&', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?', '~', '`', 'P']
            is_non_digit = is_alpha or is_special
            
            print(f"            [{j}] '{char}' - isalpha: {is_alpha}, is_special: {is_special}, is_non_digit: {is_non_digit}")
            
            if is_non_digit:
                # 중복 제거 로직
                char_y = text_info['center'][1]
                is_duplicate = False
                
                for existing_char in non_digit_chars:
                    existing_y = existing_char['center'][1]
                    if abs(char_y - existing_y) < 20:  # Y 좌표 차이가 20픽셀 이내면 중복으로 간주
                        is_duplicate = True
                        print(f"               ⚠️ 중복 문자 (기존 Y: {existing_y:.1f}, 현재 Y: {char_y:.1f})")
                        break
                
                if not is_duplicate:
                    non_digit_chars.append({
                        'char': char,
                        'text_info': text_info,
                        'center': text_info['center'],
                        'bbox': text_info['bbox']
                    })
                    print(f"               ✅ 숫자가 아닌 문자 추가: '{char}' (텍스트: '{text}') at ({text_info['center'][0]:.1f}, {text_info['center'][1]:.1f})")
                else:
                    print(f"               ❌ 중복으로 제외: '{char}'")
    
    print(f"   📊 최종 결과: {len(non_digit_chars)}개의 숫자가 아닌 문자 발견")
    for i, char_info in enumerate(non_digit_chars):
        print(f"      {i+1}. '{char_info['char']}' at ({char_info['center'][0]:.1f}, {char_info['center'][1]:.1f})")
    
    return non_digit_chars

def calculate_crop_height_for_char(char_info):
    """숫자가 아닌 문자의 Y좌표 범위로 높이 계산"""
    bbox = char_info['bbox']
    char_height = abs(bbox[2][1] - bbox[0][1])  # 문자의 실제 높이
    return char_height

def calculate_crop_dimensions(first_hole, second_hole, t_char):
    """크롭할 영역의 너비와 높이 계산"""
    if not first_hole or not t_char:
        return None, None
    
    # 너비: 첫번째 HOLE의 오른쪽 끝과 T 문자 사이의 거리
    hole_bbox = first_hole['bbox']
    hole_right_x = hole_bbox[2][0]  # HOLE의 오른쪽 끝 X 좌표
    hole_y = (hole_bbox[0][1] + hole_bbox[2][1]) / 2  # HOLE의 Y는 중앙값 사용
    
    # T 문자의 오른쪽 끝 X 좌표 사용
    t_bbox = t_char['bbox']
    t_right_x = t_bbox[2][0]  # bbox의 오른쪽 끝 X 좌표
    t_y = (t_bbox[0][1] + t_bbox[2][1]) / 2  # T의 Y는 중앙값 사용
    
    width = abs(t_right_x - hole_right_x) * 0.95
    
    # 높이: Hole 문자의 (최대Y값 - 최소Y값) = Hole 문자의 실제 높이
    hole_bbox = first_hole['bbox']
    hole_height = abs(hole_bbox[2][1] - hole_bbox[0][1])  # Hole 문자의 실제 높이
    
    return width, hole_height

def crop_golf_score_area(image_path, holes, valid_pairs, output_dir, all_texts, target_width=None):
    """골프 스코어 영역 크롭 및 저장 (숫자가 아닌 문자 기준 crop 방식)"""
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
        
        # 크롭 영역 계산 (너비만)
        width, _ = calculate_crop_dimensions(first_hole, None, first_t)
        
        if width is None or width <= 0:
            print("   ⚠️ 크롭 영역을 계산할 수 없습니다.")
            return False
        
        # 첫번째 HOLE의 오른쪽 끝 X 좌표에서 시작
        hole_bbox = first_hole['bbox']
        start_x = int(hole_bbox[2][0])
        
        # 이미지 경계 확인
        img_height, img_width = image.shape[:2]
        start_x = max(0, start_x)
        end_x = min(img_width, int(start_x + int(width)))
        
        print(f"   📐 크롭 설정:")
        print(f"      - 너비: {width:.1f}px")
        
        # T 이후의 숫자가 아닌 문자들 찾기 (전체 OCR 결과 사용)
        non_digit_chars = find_non_digit_characters_after_t(valid_pairs, all_texts)
        
        if not non_digit_chars:
            print("   ⚠️ T 이후 숫자가 아닌 문자를 찾을 수 없습니다.")
            return False
        
        print(f"   🔍 발견된 숫자가 아닌 문자: {len(non_digit_chars)}개")
        
        # 각 숫자가 아닌 문자에 대해 crop 수행
        crop_count = 0
        
        for i, char_info in enumerate(non_digit_chars):
            char = char_info['char']
            char_center = char_info['center']
            char_bbox = char_info['bbox']
            
            # 해당 문자의 높이 계산
            char_height = calculate_crop_height_for_char(char_info)
            
            # 실제 세로 길이 = 문자 높이 * 2 (중심에서 위아래로 각각 높이만큼)
            actual_height = int(char_height * 2)
            
            # 문자 중심을 기준으로 crop 영역 계산
            char_center_y = char_center[1]
            crop_start_y = int(char_center_y - char_height)
            crop_end_y = crop_start_y + actual_height
            
            # 이미지 경계 확인
            crop_start_y = max(0, crop_start_y)
            crop_end_y = min(img_height, crop_end_y)
            
            # 크롭 영역이 너무 작으면 건너뛰기
            if crop_end_y - crop_start_y < char_height:
                print(f"   ⚠️ Crop #{i+1} ({char}): 크롭 영역이 너무 작아서 건너뛰기")
                continue
            
            crop_count += 1
            print(f"   📐 Crop #{crop_count} ({char}): ({start_x}, {crop_start_y}) -> ({end_x}, {crop_end_y})")
            print(f"      - 문자 높이: {char_height:.1f}px, 실제 세로 길이: {actual_height}px")
            
            # 이미지 크롭
            cropped = image[crop_start_y:crop_end_y, start_x:end_x]
            
            if cropped.size == 0:
                print(f"   ⚠️ Crop #{crop_count} ({char}): 잘못된 크롭 영역입니다.")
                continue
            
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
                    print(f"   🔄 Crop #{crop_count} ({char}) 리사이즈: {current_width}x{current_height} -> {target_width_int}x{new_height_int}")
                else:
                    print(f"   ⚠️ Crop #{crop_count} ({char}) 잘못된 리사이즈 크기: {target_width_int}x{new_height_int}")
            
            # 파일명 생성 및 저장
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_crop_{crop_count:02d}_{char}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, cropped)
            print(f"   ✅ Crop #{crop_count} ({char}) 저장 완료: {output_filename}")
        
        print(f"   📊 총 {crop_count}개의 crop 이미지 생성 완료")
        return crop_count > 0
        
    except Exception as e:
        print(f"❌ 크롭 처리 중 오류 발생: {e}")
        return False

def extract_text_with_easyocr(image_path, reader):
    """EasyOCR을 사용하여 이미지에서 텍스트 추출"""
    try:
        print(f"\n🔍 텍스트 추출 중: {os.path.basename(image_path)}")
        
        # EasyOCR로 텍스트 추출
        results = reader.readtext(image_path)
        
        if not results:
            print("   ⚠️ 텍스트를 찾을 수 없습니다.")
            return None
        
        print(f"   📝 추출된 텍스트 ({len(results)}개):")
        
        for i, (bbox, text, confidence) in enumerate(results, 1):
            print(f"   {i:2d}. [{confidence:.3f}] {text}")
        
        return results
        
    except Exception as e:
        print(f"❌ 텍스트 추출 중 오류 발생: {e}")
        return None

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
    total_crop_count = 0  # 총 생성된 crop 이미지 수
    
    print("\n" + "=" * 60)
    print("골프 스코어 영역 크롭 처리 시작 (숫자가 아닌 문자 기준)")
    print("=" * 60)
    
    # 각 이미지에 대해 반복 crop 수행
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{total_count}] 처리 중: {os.path.basename(image_path)}")
        
        # 텍스트 추출
        results = extract_text_with_easyocr(image_path, reader)
        
        if results is None:
            continue
        
        # HOLE과 T 문자 좌표 찾기
        holes, valid_pairs = find_hole_and_t_coordinates(results)
        
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
        
        # 전체 OCR 결과를 all_texts 형태로 변환
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
        
        # 골프 스코어 영역 반복 크롭 (고정 너비 1000px로 리사이즈)
        target_width = 1000  # 고정 너비 1000px
        crop_result = crop_golf_score_area(image_path, holes, valid_pairs, test_dir, all_texts, target_width=target_width)
        if crop_result:
            success_count += 1
            total_crop_count += crop_result  # 생성된 crop 수 추가
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("처리 결과 요약")
    print("=" * 60)
    print(f"총 처리 파일: {total_count}")
    print(f"성공한 파일: {success_count}")
    print(f"실패한 파일: {total_count - success_count}")
    print(f"파일 성공률: {(success_count / total_count * 100):.1f}%")
    print(f"총 생성된 crop 이미지: {total_crop_count}개")
    print(f"결과 저장 위치: {test_dir}")

if __name__ == "__main__":
    main()
