"""
test_clean 폴더의 이미지들을 TrOCR 모델로 숫자 인식하는 테스트
- microsoft/trocr-small-printed 모델 사용
- 간단한 OCR 처리 및 결과 출력
"""

import os
import glob
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def get_clean_images():
    """test_clean 폴더의 이미지 파일 목록 반환"""
    clean_dir = "test_clean_2"
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(clean_dir, ext)
        image_files.extend(glob.glob(pattern))
    
    print(f"📁 발견된 클린 이미지: {len(image_files)}개")
    return sorted(image_files)

def load_trocr_model():
    """TrOCR 모델 로드"""
    print("🔄 TrOCR 모델 로딩 중...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
    print("✅ TrOCR 모델 로딩 완료")
    return processor, model

def recognize_text(image_path, processor, model):
    """이미지에서 텍스트 인식"""
    try:
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # OCR 수행
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values, num_beams=10, num_return_sequences=10)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 후처리: 불필요한 공백 제거
        cleaned_texts = []
        for text in generated_texts:
            cleaned_text = text.strip()
            cleaned_text = cleaned_text.replace("- ", "-")
            # cleaned_text = cleaned_text.replace(" -", "-")
            cleaned_texts.append(cleaned_text)
        
        # 첫 번째 결과가 9개 미만이면 9개인 결과를 찾아서 맨 앞으로 이동
        if cleaned_texts and len(cleaned_texts[0].split()) < 9:
            print(f"      첫 결과 부족 ({len(cleaned_texts[0].split())}개), 9개 문자 결과 찾는 중...")
            
            # 9개 문자가 있는 결과 찾기
            nine_char_results = []
            other_results = []
            
            for text in cleaned_texts:
                if len(text.split()) == 9:
                    nine_char_results.append(text)
                else:
                    other_results.append(text)
            
            if nine_char_results:
                # 9개 문자 결과를 맨 앞에, 나머지는 뒤에
                cleaned_texts = nine_char_results + other_results
                print(f"      ✅ 9개 문자 결과 {len(nine_char_results)}개 발견, 재정렬 완료")
            else:
                print(f"      ⚠️ 9개 문자 결과 없음, 원래 순서 유지")
        
        return cleaned_texts
    except Exception as e:
        print(f"   ❌ OCR 실패: {e}")
        return ""

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("TrOCR 기반 숫자 인식 테스트")
    print("=" * 60)
    
    # test_clean 폴더 확인
    if not os.path.exists("test_clean"):
        print("❌ test_clean 폴더가 존재하지 않습니다.")
        return
    
    # 이미지 파일 목록 가져오기
    image_files = get_clean_images()
    
    if not image_files:
        print("❌ 처리할 이미지 파일이 없습니다.")
        return
    
    # TrOCR 모델 로드
    processor, model = load_trocr_model()
    
    # 각 이미지에 대해 OCR 수행
    print("\n" + "=" * 60)
    print("OCR 처리 시작")
    print("=" * 60)
    
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        print(f"\n[{i}/{len(image_files)}] 처리 중: {filename}")
        
        # OCR 수행
        results = recognize_text(image_path, processor, model)
        
        if results and results != "":
            print(f"   📝 인식 결과 ({len(results)}개):")
            for i, result in enumerate(results, 1):
                if result:
                    # 공백으로 분리된 문자 개수 계산
                    char_count = len(result.split())
                    print(f"      {i}. '{result}' ({char_count}개 문자)")
                else:
                    print(f"      {i}. (빈 결과)")
        else:
            print(f"   ⚠️ 텍스트를 인식할 수 없습니다.")
    
    print("\n" + "=" * 60)
    print("OCR 처리 완료")
    print("=" * 60)

if __name__ == "__main__":
    main()
