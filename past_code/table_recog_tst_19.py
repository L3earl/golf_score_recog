import os
import subprocess
import glob

def main():
    # 이미지 폴더 경로
    # image_folder = "data/raw_table_crop"
    image_folder = "test_clean"
    
    # 이미지 파일들 가져오기
    image_files = glob.glob(os.path.join(image_folder, "*.png"))
    
    if not image_files:
        print("이미지 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(image_files)}개의 이미지를 처리합니다.\n")
    
    # 각 이미지에 대해 Tesseract OCR 실행
    for image_file in image_files:
        filename = os.path.basename(image_file)
        print(f"=== {filename} ===")
        
        try:
            # tesseract 명령어 실행
            result = subprocess.run(
                # ["tesseract", image_file, "stdout", "--psm", "7"],
                ["tesseract", image_file, "stdout", "--psm", "7", "-c", "tessedit_char_whitelist=-2-1012345 "],
                capture_output=True,
                text=True,
                check=True
            )
            
            # 결과 출력
            if result.stdout.strip():
                print(result.stdout.strip())
            else:
                print("인식된 텍스트가 없습니다.")
                
        except subprocess.CalledProcessError as e:
            print(f"오류 발생: {e}")
        except FileNotFoundError:
            print("Tesseract가 설치되지 않았거나 PATH에 없습니다.")
            break
        
        print("-" * 50)

if __name__ == "__main__":
    main()
