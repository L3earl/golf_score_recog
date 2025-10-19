"""
PaddleOCR 기반 숫자 인식 테스트
- test 폴더의 이미지들을 가져와서
- case1 clean 과정으로 이미지 정리 (검은색만 남기기)
- PaddleOCR로 숫자들을 인식하고 출력
"""

import os
import glob
import cv2
import numpy as np
import paddleocr

def clean_image_case1(image_path, output_path):
    """case1 방식으로 이미지 정리 (검은색만 남기기)
    
    Args:
        image_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
    
    Returns:
        정리 성공 여부
    """
    try:
        # 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            print(f"   [오류] 이미지 로드 실패: {image_path}")
            return False
        
        # 그레이스케일 변환
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 검은색 마스크 생성 (임계값 50 사용)
        black_threshold = 50
        black_mask = gray < black_threshold
        
        # 검은색이 아닌 픽셀을 흰색으로 변경
        cleaned = gray.copy()
        cleaned[~black_mask] = 255
        
        # 정리된 이미지 저장
        cv2.imwrite(output_path, cleaned)
        print(f"   [정리] 이미지 정리 완료: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"   [오류] 이미지 정리 실패: {e}")
        return False

def clean_all_images():
    """test 폴더의 모든 이미지를 정리하여 test_clean 폴더에 저장"""
    test_dir = "test"
    test_clean_dir = "test_clean"
    
    if not os.path.exists(test_dir):
        print(f"[오류] {test_dir} 폴더가 존재하지 않습니다.")
        return []
    
    # test_clean 폴더 생성
    os.makedirs(test_clean_dir, exist_ok=True)
    
    # 이미지 파일 목록 가져오기
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(test_dir, ext)
        image_files.extend(glob.glob(pattern))
    
    print(f"[정리] 발견된 이미지 파일: {len(image_files)}개")
    
    cleaned_files = []
    for image_path in image_files:
        filename = os.path.basename(image_path)
        output_path = os.path.join(test_clean_dir, filename)
        
        print(f"\n[정리] 처리 중: {filename}")
        if clean_image_case1(image_path, output_path):
            cleaned_files.append(output_path)
    
    print(f"\n[정리] 정리 완료: {len(cleaned_files)}개 파일")
    return cleaned_files

def get_test_images():
    """test_clean 폴더의 이미지 파일 목록 반환"""
    test_dir = "test_clean"
    if not os.path.exists(test_dir):
        print(f"[오류] {test_dir} 폴더가 존재하지 않습니다.")
        return []
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(test_dir, ext)
        image_files.extend(glob.glob(pattern))
    
    print(f"[폴더] 발견된 이미지 파일: {len(image_files)}개")
    for img in image_files:
        print(f"   - {os.path.basename(img)}")
    
    return image_files

def extract_numbers_with_paddleocr(image_path, ocr):
    """PaddleOCR을 사용하여 이미지에서 숫자 추출"""
    try:
        print(f"\n[인식] 숫자 인식 중: {os.path.basename(image_path)}")
        
        # PaddleOCR로 텍스트 추출
        results = ocr.predict(image_path)
        
        if not results or not results[0]:
            print("   [경고] 텍스트를 찾을 수 없습니다.")
            return []
        
        # 결과에서 텍스트와 신뢰도 추출
        result = results[0]
        texts = result.get('rec_texts', [])
        scores = result.get('rec_scores', [])
        boxes = result.get('rec_polys', [])
        
        # 숫자만 필터링
        numbers = []
        for i, text in enumerate(texts):
            text = text.strip()
            
            # 숫자만 추출 (정수, 소수, 음수 포함)
            if text.replace('.', '').replace('-', '').isdigit():
                confidence = scores[i] if i < len(scores) else 1.0
                bbox = boxes[i] if i < len(boxes) else None
                
                numbers.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        print(f"   [결과] 발견된 숫자 ({len(numbers)}개):")
        
        # 9개씩 한 행으로 테이블 형태로 출력
        for i in range(0, len(numbers), 9):
            row_numbers = numbers[i:i+9]
            row_text = "   "
            for j, num in enumerate(row_numbers):
                row_text += f"{num['text']:>3}"
                if j < len(row_numbers) - 1:
                    row_text += " | "
            print(row_text)
        
        # 신뢰도 정보도 별도로 출력 (선택사항)
        print(f"   [신뢰도] 평균: {sum(num['confidence'] for num in numbers) / len(numbers):.3f}")
        
        return numbers
        
    except Exception as e:
        print(f"[오류] 숫자 인식 중 오류 발생: {e}")
        return []

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("PaddleOCR 기반 숫자 인식 테스트 시작")
    print("=" * 60)
    
    # 1단계: 이미지 정리 (case1 방식)
    print("\n" + "=" * 60)
    print("1단계: 이미지 정리 (case1 방식)")
    print("=" * 60)
    
    cleaned_files = clean_all_images()
    if not cleaned_files:
        print("[오류] 이미지 정리 실패")
        return
    
    # 2단계: 정리된 이미지 파일들 가져오기
    print("\n" + "=" * 60)
    print("2단계: 정리된 이미지에서 숫자 인식")
    print("=" * 60)
    
    image_files = get_test_images()
    
    if not image_files:
        print("[오류] 처리할 이미지 파일이 없습니다.")
        return
    
    print("\n[초기화] PaddleOCR 초기화 중...")
    try:
        ocr = paddleocr.PaddleOCR(use_textline_orientation=True, lang='en')
        print("[완료] PaddleOCR 초기화 완료")
    except Exception as e:
        print(f"[오류] PaddleOCR 초기화 실패: {e}")
        return
    
    # 각 이미지에 대해 숫자 인식 수행
    print("\n" + "=" * 60)
    print("숫자 인식 처리 시작")
    print("=" * 60)
    
    total_numbers = 0
    processed_count = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 처리 중: {os.path.basename(image_path)}")
        
        numbers = extract_numbers_with_paddleocr(image_path, ocr)
        total_numbers += len(numbers)
        processed_count += 1
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("처리 결과 요약")
    print("=" * 60)
    print(f"총 처리 파일: {processed_count}")
    print(f"총 발견된 숫자: {total_numbers}개")
    print(f"파일당 평균 숫자: {(total_numbers / processed_count):.1f}개" if processed_count > 0 else "파일당 평균 숫자: 0개")

if __name__ == "__main__":
    main()