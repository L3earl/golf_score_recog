import pandas as pd
import os
from pathlib import Path

def load_csv_data(file_path):
    """CSV 파일을 로드하고 전처리"""
    try:
        df = pd.read_csv(file_path)
        # 빈 행 제거
        df = df.dropna(how='all')
        # 숫자 데이터만 선택 (헤더 제외)
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
    
    # 정수로 변환하여 비교
    answer_int = answer_data.astype(int)
    result_int = result_data.astype(int)
    
    # 맞춘 개수 계산
    correct = (answer_int == result_int).sum().sum()
    total = min_rows * min_cols
    
    return correct / total if total > 0 else 0.0

def create_score_report(case_name, results):
    """점수 보고서 생성"""
    score_dir = Path(f"data/score/{case_name}")
    score_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = score_dir / "score_report.csv"
    
    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results)
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    
    print(f"Score report saved: {report_path}")
    return report_path

def main():
    """메인 실행 함수"""
    answer_dir = Path("data/answer")
    result_dir = Path("data/result_convert_num")
    
    # answer 폴더의 모든 CSV 파일 찾기
    answer_files = list(answer_dir.glob("*.csv"))
    
    for case_name in ["case1", "case2"]:
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
                'accuracy': f"{accuracy:.4f}",
                'percentage': f"{accuracy*100:.2f}%"
            })
            
            print(f"{case_name}/{filename}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 전체 평균 계산
        if case_results:
            avg_accuracy = sum(float(r['accuracy']) for r in case_results) / len(case_results)
            case_results.append({
                'file': 'AVERAGE',
                'accuracy': f"{avg_accuracy:.4f}",
                'percentage': f"{avg_accuracy*100:.2f}%"
            })
            
            print(f"{case_name} Average: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        
        # 보고서 생성
        create_score_report(case_name, case_results)

if __name__ == "__main__":
    main()
