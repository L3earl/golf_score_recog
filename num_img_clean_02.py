import os
import cv2
import numpy as np

# 설정 변수들
INPUT_FOLDER = "raw_crop_num"
OUTPUT_FOLDER = "raw_clean_num"

# 색상 필터링 파라미터
BLACK_THRESHOLD = 50  # 검은색 임계값 (0~255, 낮을수록 더 어두운 색만 검은색으로 인식)
EXCLUDE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 45, 46, 47, 48, 49, 50, 51, 52, 53, 36, 37, 38, 39, 40, 41, 42, 43, 44, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104]

def is_black_color(pixel, threshold=BLACK_THRESHOLD):
    """픽셀이 검은색인지 판단"""
    if len(pixel.shape) == 0:  # 그레이스케일
        return pixel < threshold
    else:  # 컬러
        return np.all(pixel < threshold)

def remove_non_black_colors(image_path):
    """검은색 외의 색상 제거 (얼룩 제거)"""
    # 이미지 읽기
    img = cv2.imread(image_path)
    
    # 그레이스케일 변환
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 검은색 마스크 생성
    black_mask = gray < BLACK_THRESHOLD
    
    # 검은색이 아닌 픽셀을 흰색으로 변경
    cleaned = gray.copy()
    cleaned[~black_mask] = 255  # 검은색이 아닌 픽셀을 흰색으로
    
    return cleaned

def should_process_image(image_name):
    """이미지 번호가 처리 대상인지 확인"""
    try:
        # 파일명에서 번호 추출 (예: "0.png" -> 0)
        image_number = int(image_name.split('.')[0])
        return image_number not in EXCLUDE_INDICES
    except:
        return False

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
            
            processed_count = 0
            skipped_count = 0
            
            for image_file in image_files:
                try:
                    input_image_path = os.path.join(folder_path, image_file)
                    output_image_path = os.path.join(output_folder_path, image_file)
                    
                    # 처리 대상 이미지인지 확인
                    if should_process_image(image_file):
                        # 색상 필터링 적용
                        cleaned_image = remove_non_black_colors(input_image_path)
                        cv2.imwrite(output_image_path, cleaned_image)
                        processed_count += 1
                    else:
                        # 처리 대상이 아닌 경우 원본 복사
                        original_img = cv2.imread(input_image_path)
                        cv2.imwrite(output_image_path, original_img)
                        skipped_count += 1
                    
                except Exception as e:
                    print(f"  처리 실패 {image_file}: {e}")
                    continue
            
            print(f"  완료: {processed_count}개 색상 필터링, {skipped_count}개 원본 복사")

if __name__ == "__main__":
    print("색상 기반 얼룩 제거 시작...")
    print(f"검은색 임계값: {BLACK_THRESHOLD}")
    print(f"제외할 이미지 번호: {EXCLUDE_INDICES}")
    process_all_images()
    print("모든 처리 완료!")
