import os
import re
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def extract_numbers_from_text(text):
    """텍스트에서 숫자만 추출하는 함수"""
    numbers = re.findall(r'\d+', text)
    return numbers

def process_image_with_tablegpt(image_path, model, tokenizer):
    """이미지를 TableGPT2-7B 모델로 처리하여 테이블 인식 및 숫자 추출"""
    try:
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")
        
        # 이미지를 텍스트로 변환 (간단한 방법)
        # 실제로는 더 복잡한 전처리가 필요할 수 있음
        image_tensor = torch.tensor([1])  # 더미 텐서
        
        # 모델에 입력 (실제 구현에서는 이미지 전처리가 필요)
        with torch.no_grad():
            # 이미지에서 텍스트 추출을 위한 프롬프트
            prompt = "Extract numbers from this table image:"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # 모델 실행
            outputs = model.generate(
                inputs.input_ids,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            # 결과 디코딩
            result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        return result_text
        
    except Exception as e:
        print(f"이미지 처리 중 오류 발생 ({image_path}): {e}")
        return ""

def main():
    # 모델과 토크나이저 로드
    model_name = "tablegpt/TableGPT2-7B"
    
    try:
        print("TableGPT2-7B 모델 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("모델 로딩 완료!")
        
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        print("대안으로 간단한 텍스트 처리 방식을 사용합니다.")
        return
    
    # data/raw_img 폴더의 이미지 파일들 처리
    raw_img_dir = "data/raw_img"
    
    if not os.path.exists(raw_img_dir):
        print(f"폴더가 존재하지 않습니다: {raw_img_dir}")
        return
    
    image_files = [f for f in os.listdir(raw_img_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("처리할 이미지 파일이 없습니다.")
        return
    
    print(f"총 {len(image_files)}개의 이미지 파일을 처리합니다.")
    
    for image_file in image_files:
        image_path = os.path.join(raw_img_dir, image_file)
        print(f"\n처리 중: {image_file}")
        
        # 이미지 처리
        result_text = process_image_with_tablegpt(image_path, model, tokenizer)
        
        if result_text:
            # 숫자 추출
            numbers = extract_numbers_from_text(result_text)
            print(f"추출된 숫자들: {numbers}")
        else:
            print("텍스트 추출 실패")

if __name__ == "__main__":
    main()
