import os
from PIL import Image
from crop_coordinates import CASE3_4_COORDINATES, CASE3_3_COORDINATES, CASE3_2_COORDINATES, CASE3_1_COORDINATES

def crop_and_save_images():
    """raw_img 폴더의 모든 이미지를 크롭하여 저장"""
    # 좌표 정보 불러오기
    coordinates = CASE3_3_COORDINATES
    
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
            
            # 이미지 이름으로 폴더 생성 (확장자 제거)
            img_name = os.path.splitext(filename)[0]
            img_output_dir = os.path.join(output_dir, img_name)
            os.makedirs(img_output_dir, exist_ok=True)
            
            # 각 좌표에 대해 크롭 및 저장
            for i, (left, top, right, bottom) in enumerate(coordinates):
                try:
                    # 이미지 크롭
                    cropped = image.crop((left, top, right, bottom))
                    
                    # 저장 (0.png, 1.png, 2.png...)
                    output_filename = f"{i}.png"
                    output_path = os.path.join(img_output_dir, output_filename)
                    cropped.save(output_path)
                    
                    print(f"  저장됨: {output_path}")
                    
                except Exception as e:
                    print(f"  크롭 실패 {filename} 좌표 {i}: {e}")
                    continue

if __name__ == "__main__":
    crop_and_save_images()
    print("크롭 작업 완료!")
