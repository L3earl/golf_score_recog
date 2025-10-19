"""
OCR 변환 모듈

의도: 이미지를 TrOCR 모델로 처리하여 텍스트 데이터 추출
- case1, case2, case3별로 다른 처리 방식 적용
- 기호 검출과 숫자 인식을 통합하여 최종 결과 생성
- CSV 파일로 결과 저장
"""

import os
import time
import torch
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils import ensure_directory, get_image_files
from modules.symbol_detector import SymbolDetector
from config import (
    TROCR_MODEL_NAME, 
    RAW_CLEAN_NUM_FOLDER, 
    RESULT_CONVERT_NUM_FOLDER, 
    SCORECARD_COLUMNS, 
    NUM_PLAYERS, 
    NUM_IMAGES, 
    CSV_ENCODING,
    get_case_folder,
    CASE2_NUM_RANGE,
    CASE2_SIGN_RANGE,
    CASE2_SYMBOL_MAP
)

logger = logging.getLogger(__name__)

class OCRConverter:
    """OCR 변환 클래스
    
    의도: TrOCR 모델을 사용하여 이미지에서 텍스트를 추출하고 구조화된 데이터로 변환
    """
    
    # 클래스 변수로 모델 캐싱 (싱글톤 패턴)
    _model_cache = {}
    
    def __init__(self, case):
        """OCR 변환기 초기화
        
        의도: 케이스별 설정을 로드하고 TrOCR 모델을 초기화
        
        Args:
            case: 처리 케이스 ('case1', 'case2', 'case3')
        """
        self.case = case
        self.model_name = TROCR_MODEL_NAME
        self.input_folder = get_case_folder(RAW_CLEAN_NUM_FOLDER, case)
        self.output_folder = get_case_folder(RESULT_CONVERT_NUM_FOLDER, case)
        self.column_names = SCORECARD_COLUMNS
        self.num_players = NUM_PLAYERS
        self.num_images = NUM_IMAGES
        self.csv_encoding = CSV_ENCODING
        
        # 모델 로드 (싱글톤 패턴)
        if 'trocr' not in self._model_cache:
            self._model_cache['trocr'] = self._load_trocr_model()
        self.processor, self.model, self.device = self._model_cache['trocr']
        logger.debug(f"OCRConverter 초기화 완료 (케이스: {case})")
        
        # case2용 기호 검출기
        if case == "case2":
            self.symbol_detector = SymbolDetector.get_instance()
    
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
    
    def _extract_text_from_image_enhanced(self, image_path):
        """이미지에서 텍스트 추출 (9개 문자 우선 선택)"""
        try:
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            # 여러 후보 생성
            generated_ids = self.model.generate(
                pixel_values, 
                num_beams=10, 
                num_return_sequences=10
            )
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 후처리
            cleaned_texts = []
            for text in generated_texts:
                cleaned_text = text.strip()
                cleaned_text = cleaned_text.replace("- ", "-")
                cleaned_texts.append(cleaned_text)
            
            # 9개 문자인 결과 우선 선택
            if cleaned_texts and len(cleaned_texts[0].split()) < 9:
                nine_char_results = []
                other_results = []
                
                for text in cleaned_texts:
                    if len(text.split()) == 9:
                        nine_char_results.append(text)
                    else:
                        other_results.append(text)
                
                if nine_char_results:
                    cleaned_texts = nine_char_results + other_results
            
            # 첫 번째 결과 반환 (9개 문자 우선)
            return cleaned_texts[0] if cleaned_texts else ""
        except:
            return ""
    
    def _organize_data_case1(self, extracted_texts):
        """case1 데이터 구조화"""
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
    
    def _organize_data_case2(self, extracted_texts, symbol_values):
        """case2 데이터 구조화 (숫자 + 기호)"""
        organized_data = {}
        
        # 1행: 기호 데이터 (21~38번)
        row1_data = []
        for i in range(21, 39):  # 21~38번
            sign_value = symbol_values.get(i, 4)
            row1_data.append(sign_value)
        
        # total1: 21~29번 기호의 합
        total1 = sum(symbol_values.get(i, 4) for i in range(21, 30))
        # total2: 30~38번 기호의 합  
        total2 = sum(symbol_values.get(i, 4) for i in range(30, 39))
        # sum: total1 + total2
        sum_value = total1 + total2
        
        row1_data.extend([total1, total2, sum_value])
        organized_data['row_1'] = row1_data
        
        # 2행: 숫자 데이터 (0~20번) - 단순 나열
        row2_data = []
        for i in range(21):  # 0~20번 (18홀+각 점수들)
            num_value = extracted_texts.get(i, "").strip()
            row2_data.append(num_value)
        
        organized_data['row_2'] = row2_data
        
        return organized_data, self.column_names
    
    def _organize_data_case3(self, extracted_texts, actual_players):
        """case3 데이터 구조화 (par + 동적 플레이어 수)"""
        print(f"\n--- _organize_data_case3 디버깅 ---")
        print(f"actual_players: {actual_players}")
        print(f"extracted_texts: {extracted_texts}")
        
        organized_data = {}
        
        # 먼저 Par 데이터 처리
        print(f"\nPar 정보 (row_0) 처리:")
        par_row_data = []
        
        # Par 전반 9홀 점수
        par_front_scores = []
        par_between_key = "between_group_1"
        print(f"  par_between_key: {par_between_key}")
        print(f"  키 존재 여부: {par_between_key in extracted_texts}")
        
        if par_between_key in extracted_texts and extracted_texts[par_between_key].strip():
            par_front_text = extracted_texts[par_between_key].strip()
            print(f"  Par 전반 텍스트: '{par_front_text}'")
            par_front_scores = par_front_text.split()[:9]
            print(f"  Par 전반 점수들: {par_front_scores}")
        else:
            print(f"  ❌ Par 전반 데이터 없음")
        
        # Par 후반 9홀 점수
        par_back_scores = []
        par_after_key = "after_group_1"
        print(f"  par_after_key: {par_after_key}")
        print(f"  키 존재 여부: {par_after_key in extracted_texts}")
        
        if par_after_key in extracted_texts and extracted_texts[par_after_key].strip():
            par_back_text = extracted_texts[par_after_key].strip()
            print(f"  Par 후반 텍스트: '{par_back_text}'")
            par_back_scores = par_back_text.split()[:9]
            print(f"  Par 후반 점수들: {par_back_scores}")
        else:
            print(f"  ❌ Par 후반 데이터 없음")
        
        # Par 18홀 점수 조합
        par_all_scores = par_front_scores + par_back_scores
        while len(par_all_scores) < 18:
            par_all_scores.append(None)
        
        print(f"  Par 전체 18홀 점수: {par_all_scores}")
        par_row_data.extend(par_all_scores[:18])
        
        # Par total1, total2, sum 계산 (절대 점수)
        par_front_total = sum(int(score) for score in par_front_scores if score and score.isdigit())
        par_back_total = sum(int(score) for score in par_back_scores if score and score.isdigit())
        par_total_sum = par_front_total + par_back_total
        
        print(f"  Par total1 (전반): {par_front_total}")
        print(f"  Par total2 (후반): {par_back_total}")
        print(f"  Par sum: {par_total_sum}")
        
        par_row_data.extend([par_front_total, par_back_total, par_total_sum])
        print(f"  Par 최종 row_data: {par_row_data}")
        
        organized_data['row_0'] = par_row_data
        
        # 플레이어들 처리 (상대 점수 + 절대 점수 계산)
        for player_idx in range(1, actual_players + 1):
            row_name = f'row_{player_idx}'
            print(f"\n플레이어 {player_idx} ({row_name}) 처리:")
            
            row_data = []
            
            # 플레이어 전반 9홀 점수 (상대 점수)
            front_scores = []
            between_key = f"between_group_{player_idx + 1}"
            print(f"  between_key: {between_key}")
            print(f"  키 존재 여부: {between_key in extracted_texts}")
            
            if between_key in extracted_texts and extracted_texts[between_key].strip():
                front_text = extracted_texts[between_key].strip()
                print(f"  전반 텍스트: '{front_text}'")
                front_scores = front_text.split()[:9]
                print(f"  전반 점수들 (상대): {front_scores}")
            else:
                print(f"  ❌ 전반 데이터 없음")
            
            # 플레이어 후반 9홀 점수 (상대 점수)
            back_scores = []
            after_key = f"after_group_{player_idx + 1}"
            print(f"  after_key: {after_key}")
            print(f"  키 존재 여부: {after_key in extracted_texts}")
            
            if after_key in extracted_texts and extracted_texts[after_key].strip():
                back_text = extracted_texts[after_key].strip()
                print(f"  후반 텍스트: '{back_text}'")
                back_scores = back_text.split()[:9]
                print(f"  후반 점수들 (상대): {back_scores}")
            else:
                print(f"  ❌ 후반 데이터 없음")
            
            # 플레이어 18홀 점수 조합 (상대 점수 그대로 유지)
            all_scores = front_scores + back_scores
            while len(all_scores) < 18:
                all_scores.append(None)
            
            print(f"  전체 18홀 점수 (상대): {all_scores}")
            row_data.extend(all_scores[:18])
            
            # 플레이어 total1, total2, sum 계산 (Par + 상대점수 = 절대점수)
            front_total = 0
            back_total = 0
            
            # 전반 9홀 절대 점수 계산
            for i, (par_score, player_score) in enumerate(zip(par_front_scores, front_scores)):
                if par_score and player_score:
                    try:
                        absolute_score = int(par_score) + int(player_score)
                        front_total += absolute_score
                        print(f"    홀{i+1}: Par({par_score}) + 플레이어({player_score}) = {absolute_score}")
                    except ValueError:
                        print(f"    홀{i+1}: 숫자 변환 실패 - Par({par_score}), 플레이어({player_score})")
            
            # 후반 9홀 절대 점수 계산
            for i, (par_score, player_score) in enumerate(zip(par_back_scores, back_scores)):
                if par_score and player_score:
                    try:
                        absolute_score = int(par_score) + int(player_score)
                        back_total += absolute_score
                        print(f"    홀{i+10}: Par({par_score}) + 플레이어({player_score}) = {absolute_score}")
                    except ValueError:
                        print(f"    홀{i+10}: 숫자 변환 실패 - Par({par_score}), 플레이어({player_score})")
            
            total_sum = front_total + back_total
            
            print(f"  플레이어 total1 (전반 절대): {front_total}")
            print(f"  플레이어 total2 (후반 절대): {back_total}")
            print(f"  플레이어 sum: {total_sum}")
            
            row_data.extend([front_total, back_total, total_sum])
            print(f"  플레이어 최종 row_data: {row_data}")
            
            organized_data[row_name] = row_data
        
        print(f"\n최종 organized_data: {organized_data}")
        return organized_data, self.column_names
    
    def _process_folder_case1(self, folder_path):
        """case1 폴더 처리"""
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
        organized_data, column_names = self._organize_data_case1(extracted_texts)
        output_path = os.path.join(self.output_folder, f"{folder_name}.csv")
        data_rows = list(organized_data.values())
        df = pd.DataFrame(data_rows, columns=column_names)
        df.to_csv(output_path, index=False, encoding=self.csv_encoding)
        
        elapsed_time = time.time() - start_time
        logger.info(f"{folder_name}: {elapsed_time:.1f}초")
        
        return organized_data
    
    def _process_folder_case2(self, folder_path):
        """case2 폴더 처리 (숫자 + 기호 분리)"""
        folder_name = os.path.basename(folder_path)
        start_time = time.time()
        
        # 숫자 이미지 처리 (0-20번)
        extracted_texts = {}
        for i in range(CASE2_NUM_RANGE[0], CASE2_NUM_RANGE[1]):
            image_path = os.path.join(folder_path, f"{i}.png")
            if os.path.exists(image_path):
                text = self._extract_text_from_image(image_path)
                extracted_texts[i] = text
            else:
                extracted_texts[i] = ""
        
        # 기호 이미지 처리 (21-38번)
        symbol_values = {}
        for i in range(CASE2_SIGN_RANGE[0], CASE2_SIGN_RANGE[1]):
            image_path = os.path.join(folder_path, f"{i}.png")
            if os.path.exists(image_path):
                symbol_value = self.symbol_detector.detect(image_path)
                symbol_values[i] = symbol_value
            else:
                symbol_values[i] = 4  # 기호 없음
        
        # 데이터 구조화 및 CSV 저장
        organized_data, column_names = self._organize_data_case2(extracted_texts, symbol_values)
        output_path = os.path.join(self.output_folder, f"{folder_name}.csv")
        data_rows = list(organized_data.values())
        df = pd.DataFrame(data_rows, columns=column_names)
        df.to_csv(output_path, index=False, encoding=self.csv_encoding)
        
        elapsed_time = time.time() - start_time
        logger.info(f"{folder_name}: {elapsed_time:.1f}초")
        
        return organized_data
    
    def _process_folder_case3(self, folder_path):
        """case3 폴더 처리 (동적 플레이어 수 감지)"""
        folder_name = os.path.basename(folder_path)
        start_time = time.time()
        
        print(f"\n=== {folder_name} 디버깅 시작 ===")
        
        # 실제 이미지 파일 개수로 플레이어 수 동적 결정
        all_files = os.listdir(folder_path)
        print(f"폴더 내 모든 파일: {all_files}")
        
        between_files = [f for f in all_files if f.startswith('between_group_') and f.endswith('.png')]
        after_files = [f for f in all_files if f.startswith('after_group_') and f.endswith('.png')]
        
        print(f"between_group_ 파일들: {between_files}")
        print(f"after_group_ 파일들: {after_files}")
        
        # 플레이어 수 = 이미지 개수 - 1 (par 제외)
        actual_players = min(len(between_files), len(after_files)) - 1
        
        print(f"감지된 플레이어 수: {actual_players}")
        
        if actual_players <= 0:
            print(f"❌ {folder_name}: 플레이어 이미지가 없습니다")
            logger.warning(f"{folder_name}: 플레이어 이미지가 없습니다")
            return {}
        
        logger.info(f"{folder_name}: {actual_players}명 플레이어 감지")
        
        # between_group_n, after_group_n 이미지에서 텍스트 추출 (par + 플레이어들)
        extracted_texts = {}
        for player_idx in range(actual_players + 1):  # 0~actual_players (par + 플레이어들)
            if player_idx == 0:
                print(f"\n--- Par 정보 처리 ---")
            else:
                print(f"\n--- 플레이어 {player_idx} 처리 ---")
            
            # between_group_n.png (전반 9홀)
            between_path = os.path.join(folder_path, f"between_group_{player_idx + 1}.png")
            print(f"between_group_{player_idx + 1}.png 경로: {between_path}")
            print(f"파일 존재 여부: {os.path.exists(between_path)}")
            
            if os.path.exists(between_path):
                text = self._extract_text_from_image_enhanced(between_path)
                char_count = len(text.split()) if text else 0
                print(f"OCR 결과: '{text}' ({char_count}개 문자)")
                extracted_texts[f"between_group_{player_idx + 1}"] = text
            else:
                print(f"❌ 파일 없음: {between_path}")
                extracted_texts[f"between_group_{player_idx + 1}"] = ""
            
            # after_group_n.png (후반 9홀)
            after_path = os.path.join(folder_path, f"after_group_{player_idx + 1}.png")
            print(f"after_group_{player_idx + 1}.png 경로: {after_path}")
            print(f"파일 존재 여부: {os.path.exists(after_path)}")
            
            if os.path.exists(after_path):
                text = self._extract_text_from_image_enhanced(after_path)
                char_count = len(text.split()) if text else 0
                print(f"OCR 결과: '{text}' ({char_count}개 문자)")
                extracted_texts[f"after_group_{player_idx + 1}"] = text
            else:
                print(f"❌ 파일 없음: {after_path}")
                extracted_texts[f"after_group_{player_idx + 1}"] = ""
        
        print(f"\n추출된 텍스트 전체: {extracted_texts}")
        
        # 데이터 구조화 및 CSV 저장
        print(f"\n--- 데이터 구조화 시작 ---")
        organized_data, column_names = self._organize_data_case3(extracted_texts, actual_players)
        print(f"구조화된 데이터: {organized_data}")
        
        output_path = os.path.join(self.output_folder, f"{folder_name}.csv")
        print(f"CSV 저장 경로: {output_path}")
        
        data_rows = list(organized_data.values())
        print(f"CSV 데이터 행: {data_rows}")
        
        df = pd.DataFrame(data_rows, columns=column_names)
        df.to_csv(output_path, index=False, encoding=self.csv_encoding)
        
        print(f"✅ CSV 저장 완료: {output_path}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"{folder_name}: {elapsed_time:.1f}초")
        
        return organized_data
    
    def convert_all_folders(self, target_files=None):
        """모든 폴더의 이미지를 OCR 변환
        
        의도: 입력 폴더의 모든 하위 폴더를 케이스별로 처리하여 CSV 파일 생성
        
        Args:
            target_files: 처리할 폴더명 리스트 (None이면 전체)
        
        Returns:
            변환 성공 여부
        """
        ensure_directory(self.output_folder)
        
        if not os.path.exists(self.input_folder):
            logger.error(f"입력 폴더 없음: {self.input_folder}")
            return False
        
        try:
            logger.info(f"{self.case} OCR 변환 시작: {self.input_folder}")
            processed_folders = 0
            
            for folder_name in os.listdir(self.input_folder):
                # target_files가 지정되면 해당 파일만 처리
                if target_files and folder_name not in target_files:
                    continue
                    
                folder_path = os.path.join(self.input_folder, folder_name)
                if os.path.isdir(folder_path):
                    try:
                        if self.case == "case1":
                            self._process_folder_case1(folder_path)
                        elif self.case == "case2":
                            self._process_folder_case2(folder_path)
                        elif self.case == "case3":
                            self._process_folder_case3(folder_path)
                        else:
                            raise ValueError(f"Invalid case: {self.case}")
                        
                        processed_folders += 1
                    except Exception as e:
                        logger.error(f"{folder_name}: {e}")
                        continue
            
            logger.info(f"{self.case} OCR 변환 완료: {processed_folders}개 폴더 처리")
            return processed_folders > 0
        except Exception as e:
            logger.error(f"{self.case} OCR 변환 중 오류 발생: {e}")
            return False
