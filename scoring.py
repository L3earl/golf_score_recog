import pandas as pd
import os
from pathlib import Path
import gspread
from google.oauth2.service_account import Credentials
import json
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import TROCR_MODEL_NAME, DEFAULT_CLAUDE_MODEL

def _download_answers_from_sheet():
    """구글 스프레드시트 '정답지' 시트(A~V)를 읽어
    A열(이미지번호)별로 묶어 data/answer/{이미지번호}.csv로 저장합니다.
    - 1행은 헤더로 처리
    - 저장 시 A열(이미지번호)은 제거
    - 인증/접근 실패 시 조용히 스킵
    """
    try:
        load_dotenv()
        json_key = os.getenv('GCP_SERVICE_ACCOUNT_JSON')
        if not json_key:
            return  # 키 없으면 스킵

        service_account_info = json.loads(json_key)
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
        client = gspread.authorize(creds)

        worksheet = None

        # 1) 스프레드시트 이름이 '정답지'인 경우 시도
        try:
            spreadsheet = client.open('정답지')
            try:
                worksheet = spreadsheet.worksheet('정답지')
            except gspread.WorksheetNotFound:
                # 첫 번째 시트 사용 (대체)
                worksheet = spreadsheet.get_worksheet(0)
        except gspread.SpreadsheetNotFound:
            # 2) 대체: 기존에 사용하는 'result'에서 '정답지' 워크시트 시도
            try:
                spreadsheet = client.open('result')
                worksheet = spreadsheet.worksheet('정답지')
            except Exception:
                return  # 없으면 스킵

        if worksheet is None:
            return

        # A~V 범위 데이터 읽기
        values = worksheet.get('A:V')
        if not values or len(values) < 2:
            return  # 데이터 없음

        header = values[0][:22]  # 최대 V(22열)
        rows = values[1:]

        # 각 행 길이를 22열에 맞춤
        trimmed_rows = [r[:22] + [''] * (22 - len(r)) if len(r) < 22 else r[:22] for r in rows]

        df = pd.DataFrame(trimmed_rows, columns=header)
        if df.empty:
            return

        image_col = df.columns[0]  # 첫 컬럼이 이미지번호

        out_dir = Path("data/answer")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 이미지번호별 CSV 저장 (A열 제거)
        for image_id, group in df.groupby(image_col):
            image_str = str(image_id).strip()
            if not image_str:
                continue
            out_path = out_dir / f"{image_str}.csv"
            group.drop(columns=[image_col], errors='ignore').to_csv(out_path, index=False, encoding='utf-8-sig')

    except Exception:
        # 어떤 오류든 조용히 스킵
        return

# 파일 로드시 한 번 시도 (main과 독립)
_download_answers_from_sheet()

def load_csv_data(file_path):
    """CSV 파일을 로드하고 전처리"""
    try:
        df = pd.read_csv(file_path)
        # 빈 행 제거
        df = df.dropna(how='all')
        
        # 헤더가 있는지 확인 (첫 번째 행이 숫자가 아닌 경우)
        first_row_numeric = True
        try:
            # 첫 번째 행의 모든 값을 숫자로 변환 시도
            df.iloc[0].apply(pd.to_numeric)
        except (ValueError, TypeError):
            first_row_numeric = False
        
        if first_row_numeric:
            # 첫 번째 행부터 모두 데이터인 경우
            numeric_df = df.apply(pd.to_numeric, errors='coerce')
        else:
            # 첫 번째 행이 헤더인 경우 (기존 로직)
            numeric_df = df.iloc[1:].apply(pd.to_numeric, errors='coerce')
        
        return numeric_df.dropna()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_data(answer_df, result_df):
    """정답과 결과를 비교해서 정답률 계산"""
    if answer_df is None or result_df is None:
        return 0.0
    
    # 행과 열 수 맞추기
    min_rows = min(len(answer_df), len(result_df))
    min_cols = min(len(answer_df.columns), len(result_df.columns))
    
    answer_data = answer_df.iloc[:min_rows, :min_cols]
    result_data = result_df.iloc[:min_rows, :min_cols]
    
    # numpy 배열로 변환하여 비교 (컬럼명 무시)
    answer_array = answer_data.values.astype(int)
    result_array = result_data.values.astype(int)
    
    # 맞춘 개수 계산
    correct = (answer_array == result_array).sum()
    total = min_rows * min_cols
    
    return correct / total if total > 0 else 0.0

def get_ocr_model():
    """사용된 OCR 모델명을 동적으로 가져옵니다."""
    # result_convert 폴더가 있으면 TrOCR 모델 사용
    if Path("data/result_convert").exists():
        return TROCR_MODEL_NAME.split("/")[-1]  # microsoft/trocr-large-printed -> trocr-large-printed
    
    # result_claude 폴더가 있으면 Claude 모델 사용
    if Path("data/result_claude").exists():
        return DEFAULT_CLAUDE_MODEL
    
    # 기본값
    return "unknown"

def upload_to_google_sheets(all_results, ocr_model):
    """구글 시트에 결과 업로드"""
    try:
        # .env 파일 로드
        load_dotenv()
        
        # 환경변수에서 JSON 키 가져오기
        json_key = os.getenv('GCP_SERVICE_ACCOUNT_JSON')
        if not json_key:
            print("GCP_SERVICE_ACCOUNT_JSON 환경변수가 설정되지 않았습니다.")
            return None
        
        # JSON 문자열을 딕셔너리로 변환
        service_account_info = json.loads(json_key)
        
        # 구글 시트 인증
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
        client = gspread.authorize(creds)
        
        # 스프레드시트 열기 또는 생성
        try:
            spreadsheet = client.open('result_detail')
        except gspread.SpreadsheetNotFound:
            spreadsheet = client.create('result_detail')
        
        # result 시트 업로드
        upload_rate_sheet(spreadsheet, all_results, ocr_model)
        
        # result_detail 구글 시트 생성 및 업로드
        upload_detail_sheet(client)
        
        print(f"Results uploaded to Google Sheets: {spreadsheet.url}")
        return spreadsheet.url
        
    except Exception as e:
        print(f"Error uploading to Google Sheets: {e}")
        return None

def upload_rate_sheet(spreadsheet, all_results, ocr_model):
    """result 시트에 정답률 데이터 업로드 (같은 파일명, OCR모델이라도 Accuracy가 달라졌으면 업데이트)"""
    try:
        # result 시트 열기 또는 생성
        try:
            worksheet = spreadsheet.worksheet('result')
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet('result', rows=1000, cols=10)
        
        # 기존 데이터 가져오기
        existing_data = worksheet.get_all_records()
        
        # 헤더가 없으면 추가
        if not existing_data:
            worksheet.append_row(['Case', 'File', 'Accuracy', 'OCR_model'])
            existing_data = []  # 헤더만 추가된 상태로 초기화
        
        # 기존 데이터를 딕셔너리로 변환 (키: (Case, File, OCR_model), 값: (행번호, Accuracy))
        existing_dict = {}
        for i, row in enumerate(existing_data, start=2):  # 2부터 시작 (헤더 다음 행)
            if 'Case' in row and 'File' in row and 'OCR_model' in row:
                key = (row['Case'], row['File'], row['OCR_model'])
                existing_dict[key] = (i, row.get('Accuracy', ''))
        
        # 새로운 데이터 처리
        for case_name, results in all_results.items():
            for result in results:
                new_key = (case_name, result['file'], ocr_model)
                new_accuracy = result['accuracy']
                
                if new_key in existing_dict:
                    # 기존 행이 있는 경우
                    row_num, existing_accuracy = existing_dict[new_key]
                    
                    # Accuracy가 다르면 업데이트
                    if existing_accuracy != new_accuracy:
                        worksheet.update_cell(row_num, 3, new_accuracy)  # 3번째 컬럼이 Accuracy
                        print(f"Updated {case_name}/{result['file']}: {existing_accuracy} -> {new_accuracy}")
                    else:
                        print(f"Skipped {case_name}/{result['file']}: same accuracy ({new_accuracy})")
                else:
                    # 새로운 행 추가
                    new_row = [case_name, result['file'], new_accuracy, ocr_model]
                    worksheet.append_row(new_row)
                    print(f"Added {case_name}/{result['file']}: {new_accuracy}")
            
    except Exception as e:
        print(f"Error uploading result sheet: {e}")

def upload_detail_sheet(client):
    """result_detail 구글 시트에 모든 case 상세 비교 데이터 업로드"""
    try:
        # result_detail 스프레드시트 열기 또는 생성
        try:
            spreadsheet = client.open('result_detail')
        except gspread.SpreadsheetNotFound:
            spreadsheet = client.create('result_detail')
        
        # 각 case별로 처리
        for case_name in ["case1", "case2", "case3", "case99"]:
            upload_case_detail(client, spreadsheet, case_name)
            
    except Exception as e:
        print(f"Error uploading detail sheet: {e}")

def upload_case_detail(client, spreadsheet, case_name):
    """특정 case의 상세 비교 데이터 업로드"""
    try:
        # case 시트 열기 또는 생성
        try:
            worksheet = spreadsheet.worksheet(case_name)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(case_name, rows=1000, cols=100)
        
        # case 데이터 처리
        answer_dir = Path("data/answer")
        result_dir = Path(f"data/result_convert/{case_name}")
        
        if not answer_dir.exists() or not result_dir.exists():
            return
        
        answer_files = list(answer_dir.glob("*.csv"))
        if not answer_files:
            return
        
        # W열부터 BM열까지 지우기 (W=23, BM=65)
        worksheet.batch_clear(['W1:BM1000'])
        
        # 모든 데이터를 한 번에 준비
        all_data = []
        
        # 각 파일별 데이터 처리
        for answer_file in answer_files:
            filename = answer_file.name
            result_file = result_dir / filename
            
            if not result_file.exists():
                continue
            
            # 데이터 로드
            answer_df = load_csv_data(answer_file)
            result_df = load_csv_data(result_file)
            
            if answer_df is None or result_df is None:
                continue
            
            # 파일명 헤더 추가
            all_data.append([f"=== {filename} ==="] + [''] * 42)  # W~BM = 43열
            
            # 최대 행 수 계산
            max_rows = max(len(answer_df), len(result_df))
            
            # 헤더 행 생성
            headers = ['파일명']
            if len(answer_df.columns) > 0:
                headers.extend([f'정답_{col}' for col in answer_df.columns])
            if len(result_df.columns) > 0:
                headers.extend([f'예측_{col}' for col in result_df.columns])
            
            # 헤더를 43열로 맞추기
            while len(headers) < 43:
                headers.append('')
            all_data.append(headers)
            
            # 데이터 행 추가
            for i in range(max_rows):
                row_data = [f"행{i+1}"]
                
                # 정답 데이터 추가
                if i < len(answer_df):
                    row_data.extend(answer_df.iloc[i].astype(str).tolist())
                else:
                    row_data.extend([''] * len(answer_df.columns))
                
                # 예측 데이터 추가
                if i < len(result_df):
                    row_data.extend(result_df.iloc[i].astype(str).tolist())
                else:
                    row_data.extend([''] * len(result_df.columns))
                
                # 43열로 맞추기
                while len(row_data) < 43:
                    row_data.append('')
                
                all_data.append(row_data)
            
            # 빈 행 추가 (구분용)
            all_data.append([''] * 43)
        
        # 한 번에 모든 데이터 업로드
        if all_data:
            worksheet.update('W1', all_data)
            
    except Exception as e:
        print(f"Error uploading {case_name} detail: {e}")

def main():
    """메인 실행 함수"""
    answer_dir = Path("data/answer")
    result_dir = Path("data/result_convert")
    all_results = {}
    
    # answer 폴더의 모든 CSV 파일 찾기
    answer_files = list(answer_dir.glob("*.csv"))
    
    for case_name in ["case1", "case2", "case3", "case99"]:
        case_results = []
        
        for answer_file in answer_files:
            filename = answer_file.name
            result_file = result_dir / case_name / filename
            
            if not result_file.exists():
                print(f"Result file not found: {result_file}")
                continue
            
            # 데이터 로드
            answer_df = load_csv_data(answer_file)
            result_df = load_csv_data(result_file)
            
            # 정답률 계산
            accuracy = compare_data(answer_df, result_df)
            
            case_results.append({
                'file': filename,
                'accuracy': f"{accuracy:.4f}"
            })
            
            print(f"{case_name}/{filename}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        all_results[case_name] = case_results
    
    # 구글 시트에 업로드
    ocr_model = get_ocr_model()  # OCR 모델명 동적 가져오기
    upload_to_google_sheets(all_results, ocr_model)

if __name__ == "__main__":
    main()
