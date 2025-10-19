"""
test_clean 폴더의 이미지들을 Florence-2 모델로 숫자 인식하는 테스트
- microsoft/Florence-2-large 모델 사용
- <OCR> 프롬프트로 전체 텍스트 추출
- 간단한 OCR 처리 및 결과 출력
"""

import os
import glob
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

def get_clean_images():
    """test_clean 폴더의 이미지 파일 목록 반환"""
    clean_dir = "test_clean"
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(clean_dir, ext)
        image_files.extend(glob.glob(pattern))
    
    print(f"📁 발견된 클린 이미지: {len(image_files)}개")
    return sorted(image_files)

def load_florence2_model():
    """Florence-2 모델 로드"""
    print("🔄 Florence-2 모델 로딩 중...")
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    print("✅ Florence-2 모델 로딩 완료")
    return processor, model

def recognize_text_with_florence2(image_path, processor, model):
    """Florence-2 모델로 이미지에서 텍스트 인식"""
    try:
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # <OCR> 프롬프트로 전체 텍스트 추출
        prompt = "<OCR>"
        
        # 모델에 이미지와 프롬프트 전달
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        # 텍스트 생성
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            do_sample=False
        )
        
        # 결과 디코딩
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 프롬프트 부분 제거하고 실제 OCR 결과만 추출
        if generated_text.startswith(prompt):
            ocr_result = generated_text[len(prompt):].strip()
        else:
            ocr_result = generated_text.strip()
        
        return ocr_result
    except Exception as e:
        print(f"   ❌ OCR 실패: {e}")
        return ""

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Florence-2 기반 숫자 인식 테스트")
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
    
    # Florence-2 모델 로드
    processor, model = load_florence2_model()
    
    # 각 이미지에 대해 OCR 수행
    print("\n" + "=" * 60)
    print("OCR 처리 시작")
    print("=" * 60)
    
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        print(f"\n[{i}/{len(image_files)}] 처리 중: {filename}")
        
        # Florence-2로 OCR 수행
        result = recognize_text_with_florence2(image_path, processor, model)
        
        if result:
            print(f"   📝 인식 결과: '{result}'")
        else:
            print(f"   ⚠️ 텍스트를 인식할 수 없습니다.")
    
    print("\n" + "=" * 60)
    print("OCR 처리 완료")
    print("=" * 60)

if __name__ == "__main__":
    main()
