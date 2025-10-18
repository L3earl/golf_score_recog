"""
심플한 템플릿 매칭 테스트
- data/raw_img 폴더의 이미지들을 가져와서
- data/template_img/case3_01.png와 매칭
- 매칭된 영역을 crop하여 test 폴더에 저장
"""

import os
import cv2
import numpy as np
import glob

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

def apply_canny_edge(image):
    """Canny 엣지 검출 적용"""
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러 적용 (노이즈 제거)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny 엣지 검출
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

def multi_scale_template_match(image, template, scales=None, max_matches=2):
    """다중 스케일 템플릿 매칭 (60% ~ 160%, 1%씩 증가, 최대 2개 매칭)"""
    # 기본 스케일 설정: 60%부터 160%까지 1%씩 증가
    if scales is None:
        scales = [i/100.0 for i in range(60, 161)]  # 0.6, 0.61, 0.62, ..., 1.6
    
    all_matches = []  # 모든 매칭 결과 저장
    
    img_height, img_width = image.shape[:2]
    templ_height, templ_width = template.shape[:2]
    
    print(f"   🔍 총 {len(scales)}개 스케일로 매칭 테스트 중...")
    
    for scale in scales:
        # 템플릿 크기 조정
        new_width = int(templ_width * scale)
        new_height = int(templ_height * scale)
        
        # 스케일이 너무 작거나 큰 경우 스킵
        if new_width < 10 or new_height < 10:
            continue
        if new_width > img_width or new_height > img_height:
            continue
        
        # 템플릿 리사이징
        scaled_template = cv2.resize(template, (new_width, new_height))
        
        # 템플릿 매칭 수행
        result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
        
        # 최대 매칭 위치 찾기
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 매칭 결과 저장
        all_matches.append({
            'score': max_val,
            'location': max_loc,
            'scale': scale,
            'size': (new_width, new_height)
        })
    
    # 매칭 점수 기준으로 정렬
    all_matches.sort(key=lambda x: x['score'], reverse=True)
    
    # 상위 max_matches개 선택 (중복 제거)
    selected_matches = []
    for match in all_matches:
        if len(selected_matches) >= max_matches:
            break
        
        # 중복 체크 (이미 선택된 매칭과 너무 가까운지 확인)
        is_duplicate = False
        for selected in selected_matches:
            # 거리 계산 (유클리드 거리)
            dist = np.sqrt((match['location'][0] - selected['location'][0])**2 + 
                          (match['location'][1] - selected['location'][1])**2)
            # 템플릿 크기의 절반보다 가까우면 중복으로 간주
            min_distance = min(match['size'][0], match['size'][1]) // 2
            if dist < min_distance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            selected_matches.append(match)
    
    print(f"   ✅ {len(selected_matches)}개 매칭 발견")
    return selected_matches

def template_match_and_crop(image_path, template_path, output_dir):
    """템플릿 매칭 후 crop하여 저장 (Canny 엣지 + 다중 스케일 적용, 최대 2개 매칭)"""
    try:
        # 이미지와 템플릿 로드
        image = cv2.imread(image_path)
        template = cv2.imread(template_path)
        
        if image is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            return False
        
        if template is None:
            print(f"❌ 템플릿 로드 실패: {template_path}")
            return False
        
        # 이미지와 템플릿 크기 확인
        img_height, img_width = image.shape[:2]
        templ_height, templ_width = template.shape[:2]
        
        print(f"   이미지 크기: {img_width}x{img_height}")
        print(f"   템플릿 크기: {templ_width}x{templ_height}")
        
        # Canny 엣지 검출 적용
        image_edges = apply_canny_edge(image)
        template_edges = apply_canny_edge(template)
        
        print(f"   🔍 Canny 엣지 검출 적용 완료")
        
        # 다중 스케일 템플릿 매칭 수행 (최대 2개 매칭)
        matches = multi_scale_template_match(image_edges, template_edges, max_matches=2)
        
        if not matches:
            print(f"   ⚠️ 매칭을 찾을 수 없습니다.")
            return False
        
        # 매칭 점수가 너무 낮은 매칭들 필터링 (임계값: 0.1)
        valid_matches = [match for match in matches if match['score'] >= 0.1]
        
        if not valid_matches:
            print(f"   ⚠️ 유효한 매칭이 없습니다. (매칭 점수가 0.1 미만)")
            return False
        
        print(f"   📊 유효한 매칭: {len(valid_matches)}개")
        
        # 매칭된 2개 위치 사이의 영역을 하나로 crop
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        if len(valid_matches) >= 2:
            # 2개 매칭이 있는 경우: 두 위치 사이의 영역을 하나로 crop
            match1 = valid_matches[0]
            match2 = valid_matches[1]
            
            print(f"   매칭 1: 점수 {match1['score']:.3f}, 스케일 {match1['scale']:.2f}, 위치 ({match1['location'][0]}, {match1['location'][1]})")
            print(f"   매칭 2: 점수 {match2['score']:.3f}, 스케일 {match2['scale']:.2f}, 위치 ({match2['location'][0]}, {match2['location'][1]})")
            
            # 두 매칭 위치의 좌표
            x1, y1 = match1['location']
            x2, y2 = match2['location']
            
            # 두 매칭의 크기
            w1, h1 = match1['size']
            w2, h2 = match2['size']
            
            # 두 매칭을 포함하는 영역 계산
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1 + w1, x2 + w2)
            bottom = max(y1 + h1, y2 + h2)
            
            # 이미지 경계 확인
            img_height, img_width = image.shape[:2]
            left = max(0, left)
            top = max(0, top)
            right = min(img_width, right)
            bottom = min(img_height, bottom)
            
            crop_width = right - left
            crop_height = bottom - top
            
            print(f"   📐 통합 영역: ({left}, {top}) ~ ({right}, {bottom}), 크기: {crop_width}x{crop_height}")
            
            # 통합 영역에서 크롭
            cropped = image[top:bottom, left:right]
            
            # 파일명 생성 및 저장
            output_filename = f"{base_name}_combined.png"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, cropped)
            print(f"   ✅ 저장 완료: {output_filename}")
            
        elif len(valid_matches) == 1:
            # 1개 매칭만 있는 경우: 해당 영역만 crop
            match = valid_matches[0]
            score = match['score']
            x, y = match['location']
            scale = match['scale']
            crop_width, crop_height = match['size']
            
            print(f"   매칭 1: 점수 {score:.3f}, 스케일 {scale:.2f}, 위치 ({x}, {y})")
            
            # 원본 이미지에서 크롭
            cropped = image[y:y+crop_height, x:x+crop_width]
            
            # 파일명 생성 및 저장
            output_filename = f"{base_name}_single.png"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, cropped)
            print(f"   ✅ 저장 완료: {output_filename}")
        
        return len(valid_matches) > 0
        
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("심플한 템플릿 매칭 테스트 시작")
    print("=" * 60)
    
    # test 폴더 생성
    test_dir = create_test_folder()
    
    # 템플릿 경로 설정
    template_path = "data/template_img/case3_02.png"
    
    if not os.path.exists(template_path):
        print(f"❌ 템플릿 파일이 존재하지 않습니다: {template_path}")
        return
    
    print(f"🎯 사용할 템플릿: {template_path}")
    
    # 이미지 파일 목록 가져오기
    image_files = get_image_files()
    
    if not image_files:
        print("❌ 처리할 이미지 파일이 없습니다.")
        return
    
    # 각 이미지에 대해 템플릿 매칭 수행
    success_count = 0
    total_count = len(image_files)
    
    print("\n" + "=" * 60)
    print("템플릿 매칭 처리 시작")
    print("=" * 60)
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{total_count}] 처리 중: {os.path.basename(image_path)}")
        
        if template_match_and_crop(image_path, template_path, test_dir):
            success_count += 1
    
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
