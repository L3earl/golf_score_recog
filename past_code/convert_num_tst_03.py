import os
import time
import pandas as pd
import easyocr

# 설정 변수들
INPUT_FOLDER = "raw_clean_num"
OUTPUT_FOLDER = "result_convert_num"
LANGUAGES = ['ko','en']  # 언어 설정 (영어)
USE_GPU = True  # GPU 사용 여부

def load_easyocr_reader():
    """EasyOCR 리더 로드"""
    print("EasyOCR 모델 로딩 중...")
    reader = easyocr.Reader(LANGUAGES, gpu=USE_GPU)
    print("EasyOCR 모델 로딩 완료!")
    return reader

def extract_text_from_image(image_path, reader):
    """이미지에서 텍스트 추출 (EasyOCR 사용)"""
    try:
        # EasyOCR로 텍스트 추출 (detail=0: 텍스트만 반환)
        results = reader.readtext(image_path, detail=0)
        
        # 디버깅: 원본 결과 출력
        print(f"  EasyOCR 원본 결과: {results}")
        
        # results는 이미 텍스트 리스트이므로 바로 처리
        extracted_texts = []
        for text in results:
            if text and text.strip():  # 빈 텍스트가 아닌 경우만
                extracted_texts.append(text.strip())
        
        # 여러 텍스트가 있으면 공백 없이 연결 (음수 기호 문제 해결)
        if len(extracted_texts) == 1:
            final_text = extracted_texts[0]
        elif len(extracted_texts) > 1:
            # 공백 없이 연결하여 음수 기호 문제 해결
            final_text = ''.join(extracted_texts)
        else:
            final_text = ""
        
        # 디버깅: 최종 결과 출력
        print(f"  최종 텍스트: '{final_text}'")
        
        return final_text
    except Exception as e:
        print(f"텍스트 추출 실패 {image_path}: {e}")
        return ""

def organize_data(extracted_texts):
    """추출된 텍스트를 골프 스코어카드 구조로 정리"""
    # 컬럼명 생성: 1홀~18홀, total1, total2, sum (총 21개)
    column_names = [f"{i}홀" for i in range(1, 19)] + ["total1", "total2", "sum"]
    
    # 디버깅: extracted_texts 확인
    print(f"extracted_texts 길이: {len(extracted_texts)}")
    non_empty_count = sum(1 for v in extracted_texts.values() if v.strip())
    print(f"비어있지 않은 텍스트 개수: {non_empty_count}")
    
    # 각 플레이어별 데이터 구조 정의 (5개 플레이어)
    organized_data = {}
    
    for player_idx in range(5):  # 5개 플레이어
        row_name = f'row_{player_idx + 1}'
        row_data = []
        
        # 홀별 점수 추가 (18개)
        hole_indices = list(range(player_idx * 9, (player_idx + 1) * 9)) + list(range(45 + player_idx * 9, 45 + (player_idx + 1) * 9))
        
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
        
        # 디버깅: 각 row의 데이터 확인
        non_none_count = sum(1 for v in row_data if v is not None)
        print(f"{row_name}: {non_none_count}개 데이터, 총 {len(row_data)}개")
    
    return organized_data, column_names

def process_folder(folder_path, reader):
    """폴더 내 모든 이미지 처리"""
    folder_name = os.path.basename(folder_path)
    print(f"처리 시작: {folder_name}")
    start_time = time.time()
    
    # 모든 이미지에서 텍스트 추출
    extracted_texts = {}
    for i in range(123):  # 0~122
        image_path = os.path.join(folder_path, f"{i}.png")
        if os.path.exists(image_path):
            text = extract_text_from_image(image_path, reader)
            extracted_texts[i] = text
        else:
            extracted_texts[i] = ""
    
    # 데이터 구조화
    organized_data, column_names = organize_data(extracted_texts)
    
    # 디버깅: DataFrame 생성 전 데이터 확인
    print(f"organized_data 키 개수: {len(organized_data)}")
    print(f"column_names 개수: {len(column_names)}")
    
    # CSV로 저장
    output_path = os.path.join(OUTPUT_FOLDER, f"{folder_name}.csv")
    
    # DataFrame 생성 (컬럼명 지정)
    # organized_data를 리스트의 리스트로 변환
    data_rows = list(organized_data.values())
    df = pd.DataFrame(data_rows, columns=column_names)
    
    # 디버깅: DataFrame 확인
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame 컬럼: {list(df.columns)}")
    print(f"DataFrame 행 개수: {len(df)}")
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # 작업 시간 계산 및 출력
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"저장 완료: {output_path}")
    print(f"작업 시간: {elapsed_time:.2f}초")
    print("-" * 50)
    
    return organized_data

def main():
    """메인 실행 함수"""
    # 출력 폴더 생성
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # EasyOCR 리더 로드
    reader = load_easyocr_reader()
    
    # 전체 작업 시작 시간
    total_start_time = time.time()
    
    # 입력 폴더의 모든 하위 폴더 처리
    for folder_name in os.listdir(INPUT_FOLDER):
        folder_path = os.path.join(INPUT_FOLDER, folder_name)
        if os.path.isdir(folder_path):
            try:
                process_folder(folder_path, reader)
            except Exception as e:
                print(f"폴더 처리 실패 {folder_name}: {e}")
                continue
    
    # 전체 작업 시간 계산 및 출력
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print("모든 처리 완료!")
    print(f"전체 작업 시간: {total_elapsed_time:.2f}초")

if __name__ == "__main__":
    main()