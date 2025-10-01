import os
import cv2
import numpy as np
from PIL import Image

# 설정 변수들
INPUT_FOLDER = "raw_crop_num"
OUTPUT_FOLDER = "raw_clean_num"

# 전처리 파라미터
DENOISE_H = 15  # 디노이징 강도 (높을수록 강함, 10~30)
ADAPTIVE_BLOCK_SIZE = 15  # 적응형 임계값 블록 크기 (홀수, 11~21)
ADAPTIVE_C = 3  # 적응형 임계값 상수 (2~10)
MORPH_OPEN_KERNEL = (2, 2)  # Opening 커널 크기 (작은 얼룩 제거용)
MORPH_CLOSE_KERNEL = (3, 3)  # Closing 커널 크기 (큰 얼룩 제거용)
MIN_CONTOUR_AREA = 5  # 최소 컨투어 면적 (이보다 작은 얼룩 제거)

def preprocess_image(image_path):
    """이미지 전처리: 얼룩 및 노이즈 제거 (강화 버전)"""
    # 이미지 읽기
    img = cv2.imread(image_path)
    
    # 그레이스케일 변환
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 1. 디노이징 (노이즈 제거) - 강도 증가
    denoised = cv2.fastNlMeansDenoising(gray, h=DENOISE_H)
    
    # 2. 적응형 임계값 적용 (얼룩 제거)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )
    
    # 3. 모폴로지 연산 - Opening (작은 얼룩 제거)
    kernel_open = np.ones(MORPH_OPEN_KERNEL, np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    # 4. 모폴로지 연산 - Closing (구멍 메우기)
    kernel_close = np.ones(MORPH_CLOSE_KERNEL, np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    
    # 5. 컨투어 기반 작은 얼룩 제거
    # 이미지 반전 (흰색 배경, 검은색 전경)
    inverted = cv2.bitwise_not(closed)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 작은 컨투어 제거
    mask = np.ones_like(inverted) * 255
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            cv2.drawContours(mask, [contour], -1, 0, -1)
    
    # 마스크 적용
    cleaned = cv2.bitwise_and(inverted, mask)
    
    # 다시 반전 (검은색 배경, 흰색 전경)
    final = cv2.bitwise_not(cleaned)
    
    return final

def process_all_images():
    """모든 이미지 처리"""
    # 출력 폴더가 없으면 생성
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # 입력 폴더의 모든 하위 폴더 처리
    for folder_name in os.listdir(INPUT_FOLDER):
        folder_path = os.path.join(INPUT_FOLDER, folder_name)
        
        if os.path.isdir(folder_path):
            print(f"처리 중: {folder_name}")
            
            # 출력 폴더 생성 (같은 구조)
            output_folder_path = os.path.join(OUTPUT_FOLDER, folder_name)
            os.makedirs(output_folder_path, exist_ok=True)
            
            # 폴더 내 모든 이미지 처리
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for image_file in image_files:
                try:
                    input_image_path = os.path.join(folder_path, image_file)
                    output_image_path = os.path.join(output_folder_path, image_file)
                    
                    # 이미지 전처리
                    cleaned_image = preprocess_image(input_image_path)
                    
                    # 전처리된 이미지 저장
                    cv2.imwrite(output_image_path, cleaned_image)
                    
                except Exception as e:
                    print(f"  처리 실패 {image_file}: {e}")
                    continue
            
            print(f"  완료: {len(image_files)}개 이미지 처리")

if __name__ == "__main__":
    print("이미지 전처리 시작...")
    process_all_images()
    print("모든 처리 완료!")
