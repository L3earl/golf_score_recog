import os
from PIL import Image
from crop_coordinates import CASE3_1_COORDINATES, CASE3_2_COORDINATES, CASE3_3_COORDINATES, CASE3_4_COORDINATES

def crop_and_save_images():
    """raw_img 폴더의 모든 이미지를 크롭하여 저장"""
    # raw_img 폴더 경로
    raw_img_dir = 'data/raw_table_crop'
    output_dir = 'raw_crop_num'
    
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # raw_img 폴더의 모든 이미지 파일 처리
    for filename in os.listdir(raw_img_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"처리 중: {filename}")
            
            # 이미지 열기
            img_path = os.path.join(raw_img_dir, filename)
            try:
                image = Image.open(img_path)
            except Exception as e:
                print(f"이미지 열기 실패 {filename}: {e}")
                continue
            
            # 이미지 크기 확인
            img_width, img_height = image.size
            print(f"  이미지 크기: {img_width}x{img_height}")
            
            # 가로:세로 비율 계산
            height_ratio = img_height / img_width
            print(f"  세로/가로 비율: {height_ratio:.3f} ({height_ratio*100:.1f}%)")
            
            # 케이스 분류 및 좌표/기준높이 설정
            if height_ratio >= 1.0:  # 100% 이상
                case_name = "4인"
                coordinates = CASE3_4_COORDINATES
                base_height_ratio = 1.122  # 가로 대비 112.2%
            elif height_ratio >= 0.8:  # 80%~90%
                case_name = "3인"
                coordinates = CASE3_3_COORDINATES
                base_height_ratio = 0.905  # 가로 대비 90.5%
            elif height_ratio >= 0.6:  # 60%~80%
                case_name = "2인"
                coordinates = CASE3_2_COORDINATES
                base_height_ratio = 0.723  # 가로 대비 72.3%
            elif height_ratio >= 0.4:  # 40%~60%
                case_name = "1인"
                coordinates = CASE3_1_COORDINATES
                base_height_ratio = 0.5  # 가로 대비 50%
            else:
                print(f"  경고: 비율이 너무 낮습니다 ({height_ratio*100:.1f}%). 기본값(4인) 사용")
                case_name = "4인(기본)"
                coordinates = CASE3_4_COORDINATES
                base_height_ratio = 1.122
            
            # 기준 높이 계산
            base_height = int(img_width * base_height_ratio)
            print(f"  분류: {case_name} 케이스")
            print(f"  기준 높이: {base_height}px (가로 {img_width}px × {base_height_ratio})")
            
            # 높이 비율 계산 (실제 높이와 기준 높이의 비율)
            height_scale_ratio = img_height / base_height
            print(f"  높이 스케일 비율: {height_scale_ratio:.3f}")
            
            # 이미지 이름으로 폴더 생성 (확장자 제거)
            img_name = os.path.splitext(filename)[0]
            img_output_dir = os.path.join(output_dir, img_name)
            os.makedirs(img_output_dir, exist_ok=True)
            
            # 각 좌표에 대해 크롭 및 저장
            for i, (left, top, right, bottom) in enumerate(coordinates):
                try:
                    # Y축 좌표를 비율에 맞춰 조정
                    adjusted_top = int(top * height_scale_ratio)
                    adjusted_bottom = int(bottom * height_scale_ratio)
                    
                    # 조정된 좌표로 이미지 크롭
                    cropped = image.crop((left, adjusted_top, right, adjusted_bottom))
                    
                    # 저장 (0.png, 1.png, 2.png...)
                    output_filename = f"{i}.png"
                    output_path = os.path.join(img_output_dir, output_filename)
                    cropped.save(output_path)
                    
                    print(f"  저장됨: {output_path} (원본: {left},{top},{right},{bottom} -> 조정: {left},{adjusted_top},{right},{adjusted_bottom})")
                    
                except Exception as e:
                    print(f"  크롭 실패 {filename} 좌표 {i}: {e}")
                    continue

if __name__ == "__main__":
    crop_and_save_images()
    print("크롭 작업 완료!")
