"""
EasyOCR 기반 숫자 인식 테스트
- test_clean 폴더의 이미지들을 가져와서
- EasyOCR로 숫자들을 인식하고 테이블 형태로 출력
"""

import os
import glob
import easyocr

def main():
    # 이미지 폴더 경로
    image_folder = "test_clean"
    
    # 이미지 파일들 가져오기
    image_files = glob.glob(os.path.join(image_folder, "*.png"))
    
    if not image_files:
        print("이미지 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(image_files)}개의 이미지를 처리합니다.\n")
    
    # EasyOCR 리더 초기화
    print("EasyOCR 초기화 중...")
    try:
        reader = easyocr.Reader(['en'])
        print("EasyOCR 초기화 완료\n")
    except Exception as e:
        print(f"EasyOCR 초기화 실패: {e}")
        return
    
    # 각 이미지에 대해 EasyOCR 실행
    for image_file in image_files:
        filename = os.path.basename(image_file)
        print(f"=== {filename} ===")
        
        try:
            # EasyOCR로 텍스트 추출
            results = reader.readtext(image_file)
            
            if not results:
                print("인식된 텍스트가 없습니다.")
            else:
                # 숫자만 필터링
                numbers = []
                for bbox, text, confidence in results:
                    text = text.strip()
                    # 숫자만 추출 (정수, 소수, 음수 포함)
                    if text.replace('.', '').replace('-', '').isdigit():
                        numbers.append(text)
                
                if numbers:
                    print(f"발견된 숫자 ({len(numbers)}개):")
                    
                    # 9개씩 한 행으로 테이블 형태로 출력
                    for i in range(0, len(numbers), 9):
                        row_numbers = numbers[i:i+9]
                        row_text = "   "
                        for j, num in enumerate(row_numbers):
                            row_text += f"{num:>3}"
                            if j < len(row_numbers) - 1:
                                row_text += " | "
                        print(row_text)
                else:
                    print("숫자를 찾을 수 없습니다.")
                    
        except Exception as e:
            print(f"오류 발생: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
