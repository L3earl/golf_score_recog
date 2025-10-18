#!/usr/bin/env python3
"""
간결한 특징점 매칭 기반 테이블 크롭

data/raw_img의 이미지에서 case3_02.png 템플릿을 2개 찾아
위쪽과 아래쪽 사이의 거리를 계산하여 영역을 크롭합니다.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple

def find_two_matches(image_path: str, template_path: str, min_matches: int = 1) -> List[Tuple[int, int]]:
    """특징점 매칭으로 2개의 위치 찾기"""
    # 이미지 로드
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)
    
    if image is None or template is None:
        return []
    
    # SIFT 특징점 검출
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(template, None)
    
    if des1 is None or des2 is None:
        return []
    
    # 특징점 매칭
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des2, des1, k=2)
    
    # 좋은 매칭점 필터링 (임계값을 0.75로 완화)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    # 호모그래피 계산을 위해 최소 4개의 매칭점 필요
    if len(good_matches) < max(4, min_matches):
        return []
    
    # 템플릿과 이미지의 매칭된 좌표 추출
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 호모그래피 행렬 계산
    M, mask_homo = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is None:
        return []
    
    # 템플릿의 4개 모서리 좌표 정의
    h, w = template.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    
    # 템플릿 모서리를 이미지 좌표로 변환
    dst_corners1 = cv2.perspectiveTransform(pts, M)
    
    # 변환된 4개 모서리로 경계 사각형 계산
    x1, y1, w1, h1 = cv2.boundingRect(dst_corners1)
    top_left1 = (x1, y1)
    
    # 첫 번째 매칭 영역 마스킹 (실제 변환된 모서리 사용)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(dst_corners1)], 255)
    
    # 마스킹 영역을 크게 확대 (템플릿 크기의 1.5배)
    kernel_size = max(h, w) // 2
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    masked_image = image.copy()
    masked_image[mask > 0] = 0
    
    # 두 번째 매칭 찾기
    kp1_masked, des1_masked = sift.detectAndCompute(masked_image, None)
    
    if des1_masked is None or len(kp1_masked) < min_matches:
        return []
    
    matches2 = bf.knnMatch(des2, des1_masked, k=2)
    good_matches2 = [m for m, n in matches2 if m.distance < 0.75 * n.distance]
    
    # 호모그래피 계산을 위해 최소 4개의 매칭점 필요
    if len(good_matches2) < max(4, min_matches):
        return []
    
    # 두 번째 매칭의 호모그래피 계산
    src_pts2 = np.float32([kp2[m.queryIdx].pt for m in good_matches2]).reshape(-1, 1, 2)
    dst_pts2 = np.float32([kp1_masked[m.trainIdx].pt for m in good_matches2]).reshape(-1, 1, 2)
    
    M2, mask_homo2 = cv2.findHomography(src_pts2, dst_pts2, cv2.RANSAC, 5.0)
    
    if M2 is None:
        return []
    
    # 두 번째 템플릿 모서리를 이미지 좌표로 변환
    dst_corners2 = cv2.perspectiveTransform(pts, M2)
    
    # 두 번째 경계 사각형 계산
    x2, y2, w2, h2 = cv2.boundingRect(dst_corners2)
    top_left2 = (x2, y2)
    
    return [top_left1, top_left2]

def calculate_crop_area(matches: List[Tuple[int, int]], template_path: str) -> Tuple[int, int, int, int]:
    """크롭 영역 계산: (x, y, width, height)"""
    if len(matches) < 2:
        return (0, 0, 0, 0)
    
    # 템플릿 크기
    template = cv2.imread(template_path)
    template_height, template_width = template.shape[:2]
    
    # y 좌표 기준으로 정렬
    sorted_matches = sorted(matches, key=lambda x: x[1])
    top_match = sorted_matches[0]
    bottom_match = sorted_matches[-1]
    
    # x좌표와 y좌표 보정
    x_diff = abs(top_match[0] - bottom_match[0])
    
    # x좌표 보정 (두 HOLE 헤더는 같은 테이블이므로 x좌표가 비슷해야 함)
    if x_diff > template_width * 0.2:  # 템플릿 너비의 20% 이상 차이
        print(f"  [!] x좌표 차이 감지 ({x_diff}px), 보정 중...")
        
        # 음수가 아닌 더 합리적인 x좌표 선택
        x_values = [top_match[0], bottom_match[0]]
        # 0 이상이고 너무 크지 않은 값 필터링
        valid_x = [x for x in x_values if 0 <= x < template_width * 2]
        
        if valid_x:
            # 유효한 값 중 더 작은 값(왼쪽) 사용
            crop_x = min(valid_x)
        else:
            # 모두 유효하지 않으면 평균 사용
            crop_x = sum(x_values) // 2
    else:
        # x좌표가 비슷하면 위쪽 매칭의 x좌표 사용
        crop_x = top_match[0]
    
    # y좌표 보정: sorted_matches를 다시 정렬 (이번에는 유효한 값만)
    # 음수가 아니고 너무 크지 않은 y좌표만 선택
    valid_matches = []
    for match in matches:
        x, y = match
        if y >= 0:  # 음수가 아닌 y좌표만
            valid_matches.append(match)
    
    if len(valid_matches) < 2:
        print(f"  [!] 유효한 매칭이 부족함 ({len(valid_matches)}개)")
        return (0, 0, 0, 0)
    
    # 유효한 매칭들을 y좌표로 재정렬
    valid_sorted = sorted(valid_matches, key=lambda x: x[1])
    top_valid = valid_sorted[0]  # 가장 위쪽
    bottom_valid = valid_sorted[-1]  # 가장 아래쪽
    
    # 재정렬된 매칭으로 y좌표 설정
    crop_y = top_valid[1]
    bottom_y = bottom_valid[1]
    
    height = bottom_y - crop_y
    width = template_width
    
    # 크롭 영역
    crop_width = width
    crop_height = height
    
    print(f"  위쪽 매칭: {top_match}, 아래쪽 매칭: {bottom_match}")
    print(f"  높이: {height}, 너비: {width}")
    print(f"  크롭 영역: ({crop_x}, {crop_y}, {crop_width}, {crop_height})")
    
    return (crop_x, crop_y, crop_width, crop_height)

def crop_and_save(image_path: str, crop_area: Tuple[int, int, int, int], 
                  filename: str, output_dir: str) -> None:
    """이미지 크롭 및 저장"""
    image = cv2.imread(image_path)
    if image is None:
        return
    
    x, y, w, h = crop_area
    
    # 이미지 범위 확인
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w))
    y = max(0, min(y, img_h))
    w = max(0, min(w, img_w - x))
    h = max(0, min(h, img_h - y))
    
    if w <= 0 or h <= 0:
        print(f"  [!] 유효하지 않은 크롭 영역")
        return
    
    # 크롭 및 저장
    cropped = image[y:y+h, x:x+w]
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{base_name}_crop.png")
    
    cv2.imwrite(output_path, cropped)
    print(f"  [OK] 저장: {output_path} (크기: {cropped.shape})")

def main():
    """메인 실행"""
    # 경로 설정
    input_dir = 'data/raw_img'
    output_dir = 'test'
    template_path = 'data/template_img/case3_02.png'
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 파일 목록 (case3만 필터링)
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith('case3')]
    
    print(f"처리할 이미지: {len(image_files)}개")
    print(f"템플릿: {template_path}\n")
    
    success = 0
    fail = 0
    
    # 각 이미지 처리
    for filename in image_files:
        print(f"처리 중: {filename}")
        image_path = os.path.join(input_dir, filename)
        
        # 2개의 매칭 찾기
        matches = find_two_matches(image_path, template_path)
        
        if len(matches) >= 2:
            # 크롭 영역 계산
            crop_area = calculate_crop_area(matches, template_path)
            
            if crop_area[2] > 0 and crop_area[3] > 0:
                # 크롭 및 저장
                crop_and_save(image_path, crop_area, filename, output_dir)
                success += 1
            else:
                print(f"  [X] 유효하지 않은 크롭 영역")
                fail += 1
        else:
            print(f"  [X] 2개의 매칭을 찾을 수 없음 (발견: {len(matches)}개)")
            fail += 1
        
        print()
    
    # 결과 요약
    print("=" * 50)
    print(f"완료! 성공: {success}개, 실패: {fail}개")
    print(f"성공률: {(success / len(image_files) * 100):.1f}%")

if __name__ == "__main__":
    main()

