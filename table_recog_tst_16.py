"""
Table-Transformer Structure Recognition 기반 테이블 감지 테스트
- data/raw_img 폴더의 이미지들을 가져와서
- microsoft/table-transformer-structure-recognition 모델로 테이블 감지 후
- 감지된 이미지를 tst 폴더에 crop하여 저장
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import glob
from transformers import TableTransformerForObjectDetection, DetrImageProcessor

def create_tst_folder():
    """tst 폴더 생성"""
    tst_dir = "tst"
    os.makedirs(tst_dir, exist_ok=True)
    print(f"✅ tst 폴더 생성: {tst_dir}")
    return tst_dir

def get_image_files():
    """raw_img 폴더의 이미지 파일 목록 반환"""
    raw_img_dir = "data/raw_img"
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(raw_img_dir, ext)
        image_files.extend(glob.glob(pattern))
    
    print(f"📁 발견된 이미지 파일: {len(image_files)}개")
    for img in image_files:
        print(f"   - {os.path.basename(img)}")
    
    return image_files

def load_table_transformer_model():
    """Table-Transformer Structure Recognition 모델 로드"""
    print("🔄 Table-Transformer Structure Recognition 모델 로딩 중...")
    
    # Table-Transformer 구조 인식 모델 사용
    model_name = "microsoft/table-transformer-structure-recognition"
    
    try:
        # 모델과 프로세서 로드
        processor = DetrImageProcessor.from_pretrained(model_name)
        model = TableTransformerForObjectDetection.from_pretrained(model_name)
        
        print(f"✅ 모델 로드 완료: {model_name}")
        return processor, model
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None, None

def detect_tables(image_path, processor, model, output_dir):
    """이미지에서 테이블 감지 후 crop하여 저장"""
    try:
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")
        image_cv = cv2.imread(image_path)
        
        if image_cv is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            return False
        
        print(f"   이미지 크기: {image.size}")
        
        # 이미지 전처리
        inputs = processor(images=image, return_tensors="pt")
        
        # 테이블 감지 수행
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 결과 후처리
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]
        
        # 감지된 테이블들 처리
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.5:  # 더 엄격한 confidence threshold
                x1, y1, x2, y2 = map(int, box)
                
                # 크기 필터링 추가
                box_area = (x2 - x1) * (y2 - y1)
                image_area = image.size[0] * image.size[1]
                area_ratio = box_area / image_area
                
                # 적절한 크기의 테이블만 선택 (이미지의 5%~80% 범위)
                if 0.05 <= area_ratio <= 0.8:
                    detections.append({
                        'score': score.item(),
                        'box': box.tolist(),
                        'label': model.config.id2label[label.item()]
                    })
        
        print(f"   🔍 감지된 테이블: {len(detections)}개")
        
        if not detections:
            print(f"   ⚠️ 테이블을 찾을 수 없습니다.")
            return False
        
        # 각 감지된 테이블에 대해 crop 및 저장
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        success_count = 0
        
        for i, detection in enumerate(detections, 1):
            score = detection['score']
            box = detection['box']  # [x1, y1, x2, y2]
            
            # bounding box 좌표 추출
            x1, y1, x2, y2 = map(int, box)
            
            # 좌표 유효성 검사
            if x1 >= x2 or y1 >= y2:
                continue
            
            print(f"   테이블 {i}: 점수 {score:.3f}, 좌표 ({x1}, {y1}, {x2}, {y2})")
            
            # 이미지 crop
            cropped = image_cv[y1:y2, x1:x2]
            
            if cropped.size == 0:
                print(f"   ⚠️ 잘못된 crop 영역: {i}")
                continue
            
            # 파일명 생성 및 저장
            output_filename = f"{base_name}_table{i}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, cropped)
            print(f"   ✅ 저장 완료: {output_filename}")
            success_count += 1
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Table-Transformer Structure Recognition 기반 테이블 감지 테스트 시작")
    print("=" * 60)
    
    # tst 폴더 생성
    tst_dir = create_tst_folder()
    
    # Table-Transformer 모델 로드
    processor, model = load_table_transformer_model()
    
    if processor is None or model is None:
        print("❌ 모델 로드에 실패했습니다.")
        return
    
    # 이미지 파일 목록 가져오기
    image_files = get_image_files()
    
    if not image_files:
        print("❌ 처리할 이미지 파일이 없습니다.")
        return
    
    # 각 이미지에 대해 테이블 감지 수행
    success_count = 0
    total_count = len(image_files)
    
    print("\n" + "=" * 60)
    print("테이블 감지 처리 시작")
    print("=" * 60)
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{total_count}] 처리 중: {os.path.basename(image_path)}")
        
        if detect_tables(image_path, processor, model, tst_dir):
            success_count += 1
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("처리 결과 요약")
    print("=" * 60)
    print(f"총 처리 파일: {total_count}")
    print(f"성공: {success_count}")
    print(f"실패: {total_count - success_count}")
    print(f"성공률: {(success_count / total_count * 100):.1f}%")
    print(f"결과 저장 위치: {tst_dir}")

if __name__ == "__main__":
    main()
