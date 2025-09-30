"""
Image-to-Text 추론 스크립트
raw_img 폴더의 모든 이미지를 Image-to-Text 모델로 처리하여 텍스트로 변환
"""

# ==================== 사용자 설정 변수 ====================
# 허깅페이스 모델 설정
MODEL_NAME = "microsoft/table-transformer-structure-recognition-v1.1-all"  # Image-to-Text 모델
DEVICE = "cuda"  # "cpu" 또는 "cuda"

# 폴더 경로 설정
INPUT_FOLDER = "raw_img"
OUTPUT_FOLDER = "result_imgToTxt"

# ==================== 라이브러리 임포트 ====================
import os
from PIL import Image
import torch
from transformers import TableTransformerForObjectDetection, AutoImageProcessor
import torchvision.transforms as transforms
from datetime import datetime

# ==================== 모델 로딩 함수 ====================
def load_model():
    """Image-to-Text 모델을 로드합니다."""
    print("모델 로딩 중...")
    
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = TableTransformerForObjectDetection.from_pretrained(MODEL_NAME)
    
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
    """이미지를 처리하여 테이블 구조를 추출합니다."""
    # 이미지 로드
    image = Image.open(image_path).convert('RGB')
    
    # 이미지 크기 조정 (TableTransformer 요구사항에 맞게)
    target_size = (800, 600)  # (width, height)
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # torchvision transforms를 사용한 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 이미지를 텐서로 변환
    image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가
    
    # TableTransformer 처리 (processor 대신 직접 텐서 사용)
    inputs = {"pixel_values": image_tensor}
    
    if torch.cuda.is_available() and DEVICE == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        # 디버깅: 출력 구조 확인
        print(f"Outputs type: {type(outputs)}")
        print(f"Outputs keys/attrs: {dir(outputs)}")
        
        # TableTransformer 출력 처리
        results = outputs[0]
        print(f"Results type: {type(results)}")
        print(f"Results keys/attrs: {dir(results)}")
        
        # 다양한 방식으로 결과 추출 시도
        detected_objects = []
        
        # 방법 1: DetectionOutput 객체로 접근
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes.cpu().numpy()
            scores = results.scores.cpu().numpy()
            labels = results.pred_labels.cpu().numpy()
            
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                if score > 0.5:
                    x1, y1, x2, y2 = box
                    detected_objects.append(f"객체 {i+1}: 라벨 {label}, 신뢰도 {score:.3f}, 위치 ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        
        # 방법 2: 딕셔너리로 접근
        elif isinstance(results, dict):
            if 'boxes' in results and results['boxes'] is not None:
                boxes = results['boxes'].cpu().numpy()
                scores = results['scores'].cpu().numpy()
                labels = results['labels'].cpu().numpy()
                
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    if score > 0.5:
                        x1, y1, x2, y2 = box
                        detected_objects.append(f"객체 {i+1}: 라벨 {label}, 신뢰도 {score:.3f}, 위치 ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        
        # 방법 3: 직접 텐서 접근
        else:
            try:
                # outputs에서 직접 추출
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    print(f"Logits shape: {logits.shape}")
                    predicted_text = f"모델 출력 - Logits shape: {logits.shape}"
                else:
                    predicted_text = f"모델 출력 - Type: {type(outputs)}, Attrs: {dir(outputs)}"
            except Exception as e:
                predicted_text = f"모델 출력 처리 오류: {e}"
        
        if detected_objects:
            predicted_text = "\n".join(detected_objects)
        elif 'predicted_text' not in locals():
            predicted_text = "감지된 객체가 없습니다."
    
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
