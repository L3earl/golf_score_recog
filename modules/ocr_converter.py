"""
OCR 변환 모듈
- TrOCR 모델을 사용한 텍스트 추출
- 골프 스코어카드 구조로 데이터 정리
- CSV 파일로 저장
"""

import os
import time
import torch
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TROCR_MODEL_NAME, 
    RAW_CLEAN_NUM_FOLDER, 
    RESULT_CONVERT_NUM_FOLDER, 
    SCORECARD_COLUMNS, 
    NUM_PLAYERS, 
    NUM_IMAGES, 
    CSV_ENCODING
)

class OCRConverter:
    """OCR 변환 클래스"""
    
    def __init__(self):
        """OCR 변환기 초기화"""
        self.model_name = TROCR_MODEL_NAME
        self.input_folder = RAW_CLEAN_NUM_FOLDER
        self.output_folder = RESULT_CONVERT_NUM_FOLDER
        self.column_names = SCORECARD_COLUMNS
        self.num_players = NUM_PLAYERS
        self.num_images = NUM_IMAGES
        self.csv_encoding = CSV_ENCODING
        
        # 모델 로드
        self.processor, self.model, self.device = self._load_trocr_model()
    
    def _get_device(self):
        """GPU 사용 가능 여부 확인 및 device 설정"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("  ⚠️  CPU 사용")
        return device
    
    def _load_trocr_model(self):
        """TrOCR 모델 로드"""
        try:
            processor = TrOCRProcessor.from_pretrained(self.model_name)
            model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            device = self._get_device()
            model = model.to(device)
            return processor, model, device
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def _extract_text_from_image(self, image_path):
        """이미지에서 텍스트 추출"""
        try:
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
        except:
            return ""
    
    def _organize_data(self, extracted_texts):
        """추출된 텍스트를 골프 스코어카드 구조로 정리"""
        
        # 각 플레이어별 데이터 구조 정의
        organized_data = {}
        
        for player_idx in range(self.num_players):
            row_name = f'row_{player_idx + 1}'
            row_data = []
            
            # 홀별 점수 추가 (18개)
            hole_indices = (list(range(player_idx * 9, (player_idx + 1) * 9)) + 
                          list(range(45 + player_idx * 9, 45 + (player_idx + 1) * 9)))
            
            for idx in hole_indices:
                if idx in extracted_texts and extracted_texts[idx].strip():
                    row_data.append(extracted_texts[idx].strip())
                else:
                    row_data.append(None)
            
            # total1, total2, sum 추가
            total1_idx = 90 + player_idx
            total2_idx = 95 + player_idx
            sum_idx = 100 + player_idx 
            
            # total1 추가
            if total1_idx in extracted_texts and extracted_texts[total1_idx].strip():
                row_data.append(extracted_texts[total1_idx].strip())
            else:
                row_data.append(None)
            
            # total2 추가
            if total2_idx in extracted_texts and extracted_texts[total2_idx].strip():
                row_data.append(extracted_texts[total2_idx].strip())
            else:
                row_data.append(None)
            
            # sum 추가
            if sum_idx and sum_idx in extracted_texts and extracted_texts[sum_idx].strip():
                row_data.append(extracted_texts[sum_idx].strip())
            else:
                row_data.append(None)
            
            organized_data[row_name] = row_data
        
        return organized_data, self.column_names
    
    def _process_folder(self, folder_path):
        """폴더 내 모든 이미지 처리"""
        folder_name = os.path.basename(folder_path)
        start_time = time.time()
        
        # 모든 이미지에서 텍스트 추출
        extracted_texts = {}
        for i in range(self.num_images):
            image_path = os.path.join(folder_path, f"{i}.png")
            if os.path.exists(image_path):
                text = self._extract_text_from_image(image_path)
                extracted_texts[i] = text
            else:
                extracted_texts[i] = ""
        
        # 데이터 구조화 및 CSV 저장
        organized_data, column_names = self._organize_data(extracted_texts)
        output_path = os.path.join(self.output_folder, f"{folder_name}.csv")
        data_rows = list(organized_data.values())
        df = pd.DataFrame(data_rows, columns=column_names)
        df.to_csv(output_path, index=False, encoding=self.csv_encoding)
        
        elapsed_time = time.time() - start_time
        print(f"  ✓ {folder_name}: {elapsed_time:.1f}초")
        
        return organized_data
    
    def convert_all_folders(self):
        """모든 폴더의 이미지를 OCR 변환"""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        if not os.path.exists(self.input_folder):
            print(f"❌ 입력 폴더 없음: {self.input_folder}")
            return False
        
        processed_folders = 0
        
        for folder_name in os.listdir(self.input_folder):
            folder_path = os.path.join(self.input_folder, folder_name)
            if os.path.isdir(folder_path):
                try:
                    self._process_folder(folder_path)
                    processed_folders += 1
                except Exception as e:
                    print(f"  ❌ {folder_name}: {e}")
                    continue
        
        return processed_folders > 0
