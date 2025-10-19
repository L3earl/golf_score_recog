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

def crop_golf_score_area(image_path, holes, valid_pairs, output_dir, target_width=None):
    """골프 스코어 영역 크롭 및 저장"""
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
        
        # 두번째 쌍이 있으면 사용, 없으면 None
        second_pair = valid_pairs[1] if len(valid_pairs) > 1 else None
        second_hole = second_pair['hole'] if second_pair else None
        
        # 크롭 영역 계산
        width, height = calculate_crop_dimensions(first_hole, second_hole, first_t)
        
        if width is None or height is None or width <= 0 or height <= 0:
            print("   ⚠️ 크롭 영역을 계산할 수 없습니다.")
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
        
        print(f"   📐 크롭 영역: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
        print(f"   📏 크기: {end_x - start_x} x {end_y - start_y}")
        
        # 이미지 크롭
        cropped = image[start_y:end_y, start_x:end_x]
        
        if cropped.size == 0:
            print("   ⚠️ 잘못된 크롭 영역입니다.")
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
                print(f"   🔄 리사이즈: {current_width}x{current_height} -> {target_width_int}x{new_height_int}")
            else:
                print(f"   ⚠️ 잘못된 리사이즈 크기: {target_width_int}x{new_height_int}")
        
        # 파일명 생성 및 저장
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{base_name}_golf_score.png"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, cropped)
        print(f"   ✅ 저장 완료: {output_filename}")
        
        return True
        
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
    crop_widths = []  # 모든 크롭된 이미지의 너비 저장
    
    print("\n" + "=" * 60)
    print("골프 스코어 영역 크롭 처리 시작")
    print("=" * 60)
    
    # 1단계: 모든 이미지를 크롭하고 너비 수집
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
        
        # 골프 스코어 영역 크롭 (임시로 너비만 수집)
        if valid_pairs:
            first_pair = valid_pairs[0]
            first_hole = first_pair['hole']
            first_t = first_pair['t_char']
            second_pair = valid_pairs[1] if len(valid_pairs) > 1 else None
            second_hole = second_pair['hole'] if second_pair else None
            
            width, height = calculate_crop_dimensions(first_hole, second_hole, first_t)
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
            results = extract_text_with_easyocr(image_path, reader)
            
            if results is None:
                continue
            
            # HOLE과 T 문자 좌표 찾기
            holes, valid_pairs = find_hole_and_t_coordinates(results)
            
            # 골프 스코어 영역 크롭 (동일한 너비로 리사이즈)
            if crop_golf_score_area(image_path, holes, valid_pairs, test_dir, target_width=target_width):
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
