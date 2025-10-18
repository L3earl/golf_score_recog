"""
LoFTR + OpenCV 기반 특징점 매칭 테스트
- data/raw_img 폴더의 이미지들을 가져와서
- data/template_img/case3_01.png와 LoFTR, OpenCV 특징점으로 매칭
- 매칭된 이미지를 tst 폴더에 crop하여 저장
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import glob
import kornia as K
from kornia.feature import LoFTR

def create_tst_folder():
    """tst 폴더 생성"""
    tst_dir = "tst"
    os.makedirs(tst_dir, exist_ok=True)
    print(f"✅ tst 폴더 생성: {tst_dir}")
    return tst_dir

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

def load_loftr_model():
    """LoFTR 모델 로드"""
    print("🔄 LoFTR 모델 로딩 중...")
    
    try:
        # LoFTR 모델 로드
        loftr = LoFTR(pretrained=True)
        print("✅ LoFTR 모델 로드 완료")
        return loftr
    except Exception as e:
        print(f"❌ LoFTR 모델 로드 실패: {e}")
        return None

def preprocess_image(image_path):
    """이미지 전처리 (텐서 변환)"""
    try:
        # 이미지 로드 및 전처리
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # BGR to RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환
        pil_image = Image.fromarray(image_rgb)
        
        # 텐서로 변환 (H, W, C) -> (1, C, H, W)
        tensor_image = K.image_to_tensor(np.array(pil_image), keepdim=True).float() / 255.0
        
        return tensor_image, image
    except Exception as e:
        print(f"❌ 이미지 전처리 실패: {e}")
        return None, None

def match_with_loftr(image_tensor, template_tensor):
    """LoFTR를 사용한 특징점 매칭"""
    try:
        loftr = load_loftr_model()
        if loftr is None:
            return None
        
        # LoFTR 매칭 수행
        with torch.no_grad():
            input_dict = {
                "image0": template_tensor,
                "image1": image_tensor
            }
            match_dict = loftr(input_dict)
        
        # 매칭 결과 추출
        kpts0 = match_dict["keypoints0"].cpu().numpy()
        kpts1 = match_dict["keypoints1"].cpu().numpy()
        matches = match_dict["matches"].cpu().numpy()
        
        # 유효한 매칭만 필터링
        valid_matches = matches > -1
        if np.sum(valid_matches) < 10:  # 최소 10개 매칭 필요
            return None
        
        matched_kpts0 = kpts0[valid_matches]
        matched_kpts1 = kpts1[matches[valid_matches]]
        
        return matched_kpts0, matched_kpts1
    except Exception as e:
        print(f"❌ LoFTR 매칭 실패: {e}")
        return None

def match_with_opencv(image, template):
    """OpenCV SIFT를 사용한 특징점 매칭"""
    try:
        # SIFT 특징점 검출기 생성
        sift = cv2.SIFT_create()
        
        # 특징점과 디스크립터 추출
        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(image, None)
        
        if des1 is None or des2 is None:
            return None
        
        # FLANN 매처 사용
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 매칭 수행
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test 적용
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:  # 최소 10개 매칭 필요
            return None
        
        # 매칭된 특징점 좌표 추출
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        return src_pts.reshape(-1, 2), dst_pts.reshape(-1, 2)
    except Exception as e:
        print(f"❌ OpenCV 매칭 실패: {e}")
        return None

def find_homography_and_crop(kpts0, kpts1, template_image, source_image, output_dir, base_name, method_name):
    """호모그래피 계산 후 crop"""
    try:
        if len(kpts0) < 4:
            return False
        
        # RANSAC으로 호모그래피 계산
        H, mask = cv2.findHomography(kpts0, kpts1, cv2.RANSAC, 5.0)
        
        if H is None:
            return False
        
        # 템플릿 이미지 크기
        h, w = template_image.shape[:2]
        
        # 템플릿의 네 모서리 점들
        template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        # 호모그래피를 사용해 소스 이미지에서 해당 영역 찾기
        source_corners = cv2.perspectiveTransform(template_corners, H)
        
        # 바운딩 박스 계산
        x_coords = source_corners[:, 0, 0]
        y_coords = source_corners[:, 0, 1]
        
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        # 이미지 경계 내로 제한
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(source_image.shape[1], x_max)
        y_max = min(source_image.shape[0], y_max)
        
        # crop 수행
        cropped = source_image[y_min:y_max, x_min:x_max]
        
        if cropped.size == 0:
            return False
        
        # 저장
        output_filename = f"{base_name}_{method_name}_matched.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, cropped)
        print(f"   ✅ 저장 완료: {output_filename}")
        
        return True
    except Exception as e:
        print(f"❌ 호모그래피 계산 실패: {e}")
        return False

def match_and_crop(image_path, template_path, output_dir):
    """이미지와 템플릿 매칭 후 crop"""
    try:
        print(f"   처리 중: {os.path.basename(image_path)}")
        
        # 이미지 전처리
        image_tensor, image_cv = preprocess_image(image_path)
        template_tensor, template_cv = preprocess_image(template_path)
        
        if image_tensor is None or template_tensor is None:
            return False
        
        # 그레이스케일 변환 (OpenCV 매칭용)
        image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_cv, cv2.COLOR_BGR2GRAY)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        success = False
        
        # LoFTR로 매칭 시도
        print("   🔍 LoFTR 매칭 시도...")
        loftr_result = match_with_loftr(image_tensor, template_tensor)
        if loftr_result is not None:
            kpts0, kpts1 = loftr_result
            if find_homography_and_crop(kpts0, kpts1, template_cv, image_cv, output_dir, base_name, "loftr"):
                success = True
                print("   ✅ LoFTR 매칭 성공")
        
        # OpenCV SIFT로 매칭 시도
        print("   🔍 OpenCV SIFT 매칭 시도...")
        opencv_result = match_with_opencv(image_gray, template_gray)
        if opencv_result is not None:
            kpts0, kpts1 = opencv_result
            if find_homography_and_crop(kpts0, kpts1, template_cv, image_cv, output_dir, base_name, "opencv"):
                success = True
                print("   ✅ OpenCV SIFT 매칭 성공")
        
        return success
        
    except Exception as e:
        print(f"❌ 매칭 처리 중 오류 발생: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("LoFTR + OpenCV SIFT 기반 특징점 매칭 테스트 시작")
    print("=" * 60)
    
    # tst 폴더 생성
    tst_dir = create_tst_folder()
    
    # 템플릿 경로 설정
    template_path = "data/template_img/case3_01.png"
    
    if not os.path.exists(template_path):
        print(f"❌ 템플릿 파일이 존재하지 않습니다: {template_path}")
        return
    
    print(f"🎯 사용할 템플릿: {template_path}")
    
    # 이미지 파일 목록 가져오기
    image_files = get_image_files()
    
    if not image_files:
        print("❌ 처리할 이미지 파일이 없습니다.")
        return
    
    # 각 이미지에 대해 매칭 수행
    success_count = 0
    total_count = len(image_files)
    
    print("\n" + "=" * 60)
    print("특징점 매칭 처리 시작")
    print("=" * 60)
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{total_count}] 처리 중: {os.path.basename(image_path)}")
        
        if match_and_crop(image_path, template_path, tst_dir):
            success_count += 1
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("처리 결과 요약")
    print("=" * 60)
    print(f"총 처리 파일: {total_count}")
    print(f"성공: {success_count}")
    print(f"실패: {total_count - success_count}")
    print(f"성공률: {(success_count / total_count * 100):.1f}%")
    print(f"결과 저장 위치: {tst_dir}")

if __name__ == "__main__":
    main()
