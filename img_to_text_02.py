"""
Image-to-Text 추론 스크립트
raw_img 폴더의 모든 이미지를 Image-to-Text 모델로 처리하여 텍스트로 변환
"""

# ==================== 사용자 설정 변수 ====================
# 허깅페이스 모델 설정
MODEL_NAME = "microsoft/layoutlmv3-base"  # Image-to-Text 모델
DEVICE = "cuda"  # "cpu" 또는 "cuda"

# 폴더 경로 설정
INPUT_FOLDER = "raw_img"
OUTPUT_FOLDER = "result_imgToTxt"

# ==================== 라이브러리 임포트 ====================
import os
from PIL import Image
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from datetime import datetime

# ==================== 모델 로딩 함수 ====================
def load_model():
    """Image-to-Text 모델을 로드합니다."""
    print("모델 로딩 중...")
    
    processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME)
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_NAME)
    
    if torch.cuda.is_available() and DEVICE == "cuda":
        model = model.cuda()
    
    print("모델 로딩 완료!")
    
    # 전체 파라미터 수 계산
    num_params = model.num_parameters()
    
    # 백만 단위(M)로 변환하여 출력
    print(f"'{MODEL_NAME}' 모델의 파라미터 수: {num_params / 1_000_000:.2f}M")
    
    return processor, model

# ==================== 이미지 처리 함수 ====================
def process_image(image_path, processor, model):
    """이미지를 처리하여 텍스트를 추출합니다."""
    # 이미지 로드
    image = Image.open(image_path).convert('RGB')
    
    # LayoutLMv3 처리
    inputs = processor(image, return_tensors="pt")
    
    if torch.cuda.is_available() and DEVICE == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_labels = torch.argmax(predictions, dim=-1)
        
        # 토큰을 텍스트로 변환
        tokens = processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        predicted_text = processor.tokenizer.convert_tokens_to_string(tokens)
    
    return predicted_text.strip()

# ==================== 결과 저장 함수 ====================
def save_results(results, output_file):
    """결과를 파일에 저장합니다."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Image-to-Text 추론 결과\n")
        f.write(f"모델: {MODEL_NAME}\n")
        f.write(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, (image_name, text) in enumerate(results, 1):
            f.write(f"이미지 {i}: {image_name}\n")
            f.write(f"텍스트: {text}\n")
            f.write("-" * 30 + "\n\n")

# ==================== 메인 실행 함수 ====================
def main():
    """메인 실행 함수"""
    print("Image-to-Text 추론 시작...")
    
    # 모델 로딩
    processor, model = load_model()
    
    # 입력 폴더 확인
    if not os.path.exists(INPUT_FOLDER):
        print(f"입력 폴더가 존재하지 않습니다: {INPUT_FOLDER}")
        return
    
    # 이미지 파일 목록 가져오기
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(INPUT_FOLDER):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(INPUT_FOLDER, file))
    
    if not image_files:
        print(f"입력 폴더에 이미지 파일이 없습니다: {INPUT_FOLDER}")
        return
    
    print(f"처리할 이미지 수: {len(image_files)}")
    
    # 모든 이미지 처리
    results = []
    
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        print(f"처리 중: {image_name}")
        
        try:
            text = process_image(image_path, processor, model)
            results.append((image_name, text))
            print(f"  → {text}")
        except Exception as e:
            print(f"  → 오류: {e}")
            results.append((image_name, f"오류: {e}"))
    
    # 결과 저장
    output_file = os.path.join(OUTPUT_FOLDER, f"{MODEL_NAME.split('/')[-1]}.txt")
    save_results(results, output_file)
    
    print(f"\n처리 완료! 결과가 저장되었습니다: {output_file}")
    print(f"총 {len(results)}개의 이미지가 처리되었습니다.")

if __name__ == "__main__":
    main()
