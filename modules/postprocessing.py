"""
후처리 및 예외처리 모듈

의도: OCR 결과를 정제하고 검증하여 최종 스코어카드 데이터 생성
- 빈 행 제거: total1, total2가 임계값 이하인 행 제거
- Sum 계산: total1 + total2를 sum 컬럼에 입력
- 예외처리: 숫자 데이터 검증, 타수 일치성 검증, 빈 데이터 검증
"""

import os
import pandas as pd
import numpy as np
import re
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils import is_numeric, convert_to_numeric
from config import (
    RESULT_CONVERT_NUM_FOLDER,
    SCORECARD_COLUMNS,
    NUM_PLAYERS,
    CSV_ENCODING,
    get_case_folder,
    MIN_TOTAL_THRESHOLD
)

logger = logging.getLogger(__name__)

class PostProcessor:
    """후처리 및 예외처리 클래스
    
    의도: OCR 결과를 정제하고 검증하여 최종 스코어카드 데이터 생성
    """
    
    def __init__(self, case="case1"):
        """후처리기 초기화
        
        의도: 케이스별 설정을 로드하여 후처리기 인스턴스 생성
        
        Args:
            case: 처리 케이스 ('case1', 'case2', 'case3')
        """
        self.case = case
        self.input_folder = get_case_folder(RESULT_CONVERT_NUM_FOLDER, case)
        self.column_names = SCORECARD_COLUMNS
        self.num_players = NUM_PLAYERS
        self.csv_encoding = CSV_ENCODING
        
        # 예외처리 기준값
        self.min_total_threshold = MIN_TOTAL_THRESHOLD
        self.hole_columns = [f"{i}홀" for i in range(1, 19)]  # 1홀~18홀 컬럼
        self.total1_col = "total1"
        self.total2_col = "total2"
        self.sum_col = "sum"
        logger.debug(f"PostProcessor 초기화 완료 (케이스: {case})")
    
    def _calculate_hole_sum(self, row, par_row=None):
        """홀별 점수의 합 계산 (par + 플레이어점수)"""
        hole_sum = 0
        valid_holes = 0
        
        for hole_col in self.hole_columns:
            if hole_col in row:
                player_value = convert_to_numeric(row[hole_col])
                if player_value is not None:
                    # par 값이 있으면 더하기
                    if par_row is not None and hole_col in par_row:
                        par_value = convert_to_numeric(par_row[hole_col])
                        if par_value is not None:
                            hole_sum += par_value + player_value
                            valid_holes += 1
                        else:
                            hole_sum += player_value
                            valid_holes += 1
                    else:
                        hole_sum += player_value
                        valid_holes += 1
        
        return hole_sum, valid_holes
    
    def _detect_exceptions(self, df, folder_name):
        """예외 케이스 감지"""
        exceptions = []
        
        print(f"\n🔍 {folder_name} 예외 감지 시작:")
        print(f"  - DataFrame shape: {df.shape}")
        
        # 1. 데이터가 부족한 경우 (컬럼만 있거나 1행만 있는 경우)
        if df.empty or len(df) == 0 or len(df) == 1:
            print(f"  ❌ 데이터 부족 감지")
            exceptions.append({
                'type': 'insufficient_data',
                'message': '데이터 부족 (컬럼만 존재하거나 1행만 존재)',
                'severity': 'high'
            })
            return exceptions
        
        # par 행 (행0) 가져오기
        par_row = None
        if len(df) > 0:
            par_row = df.iloc[0]  # 첫 번째 행이 par
        
        # 2. 각 행별 검사 (행1~4만 검사)
        for idx, row in df.iterrows():
            # 행0(par)은 건너뛰기
            if idx == 0:
                continue
                
            row_exceptions = []
            
            # 디버깅: 각 행의 total1, total2 값 확인
            total1 = convert_to_numeric(row.get(self.total1_col))
            total2 = convert_to_numeric(row.get(self.total2_col))
            print(f"  행{idx}: total1={total1} (타입: {type(total1)}), total2={total2} (타입: {type(total2)})")
            
            # 2-1. 숫자가 아닌 데이터 감지
            non_numeric_data = []
            for col in self.hole_columns + [self.total1_col, self.total2_col, self.sum_col]:
                if col in row and not is_numeric(row[col]):
                    non_numeric_data.append(f"{col}: {row[col]}")
            
            if non_numeric_data:
                print(f"    ❌ 숫자가 아닌 데이터: {non_numeric_data}")
                row_exceptions.append({
                    'type': 'non_numeric',
                    'message': f"숫자가 아닌 데이터: {', '.join(non_numeric_data)}",
                    'severity': 'medium',
                    'row': idx
                })
            
            # 2-2. 타수 일치성 검증
            sum_value = convert_to_numeric(row.get(self.sum_col))
            
            if total1 is not None and total2 is not None:
                calculated_sum = total1 + total2
                
                # sum 컬럼이 있고 값이 다른 경우
                if sum_value is not None and abs(calculated_sum - sum_value) > 0.1:
                    print(f"    ❌ 합계 불일치: {total1} + {total2} = {calculated_sum}, sum={sum_value}")
                    row_exceptions.append({
                        'type': 'sum_mismatch',
                        'message': f"합계 불일치: total1({total1}) + total2({total2}) = {calculated_sum}, sum({sum_value})",
                        'severity': 'high',
                        'row': idx
                    })
                
                # 홀별 점수 합과 total1+total2 비교 (par + 플레이어점수)
                hole_sum, valid_holes = self._calculate_hole_sum(row, par_row)
                if valid_holes > 0 and abs(hole_sum - calculated_sum) > 0.1:
                    print(f"    ❌ 홀별 합계 불일치: 홀별합(par+플레이어)={hole_sum}, total1+total2={calculated_sum}")
                    row_exceptions.append({
                        'type': 'hole_sum_mismatch',
                        'message': f"홀별 합계 불일치: 홀별합(par+플레이어)({hole_sum}) ≠ total1+total2({calculated_sum})",
                        'severity': 'medium',
                        'row': idx
                    })
            
            # 2-3. total1, total2가 너무 작은 경우 (30 이하)
            if total1 is not None and total1 <= self.min_total_threshold:
                print(f"    ⚠️ total1이 작음: {total1} ≤ {self.min_total_threshold}")
                row_exceptions.append({
                    'type': 'low_total1',
                    'message': f"total1이 너무 작음: {total1} (임계값: {self.min_total_threshold})",
                    'severity': 'low',
                    'row': idx
                })
            
            if total2 is not None and total2 <= self.min_total_threshold:
                print(f"    ⚠️ total2가 작음: {total2} ≤ {self.min_total_threshold}")
                row_exceptions.append({
                    'type': 'low_total2',
                    'message': f"total2가 너무 작음: {total2} (임계값: {self.min_total_threshold})",
                    'severity': 'low',
                    'row': idx
                })
            
            # 행별 예외를 전체 예외에 추가
            for exc in row_exceptions:
                exc['folder'] = folder_name
                exceptions.append(exc)
            
            # 행별 예외 요약
            if row_exceptions:
                print(f"  행{idx} 예외: {len(row_exceptions)}개")
        
        print(f"  총 예외: {len(exceptions)}개")
        return exceptions
    
    def _remove_empty_rows(self, df):
        """빈 행 제거 (total1, total2가 30 이하인 행)"""
        rows_to_remove = []
        
        for idx, row in df.iterrows():
            total1 = convert_to_numeric(row.get(self.total1_col))
            total2 = convert_to_numeric(row.get(self.total2_col))
            
            if (total1 is not None and total1 <= self.min_total_threshold) or \
               (total2 is not None and total2 <= self.min_total_threshold):
                rows_to_remove.append(idx)
        
        if rows_to_remove:
            return df.drop(rows_to_remove)
        return df
    
    def _calculate_sum_column(self, df):
        """total1 + total2를 sum 컬럼에 계산"""
        for idx, row in df.iterrows():
            total1 = convert_to_numeric(row.get(self.total1_col))
            total2 = convert_to_numeric(row.get(self.total2_col))
            
            if total1 is not None and total2 is not None:
                df.at[idx, self.sum_col] = int(total1 + total2)
        
        return df
    
    def _convert_to_int(self, df):
        """total1, total2, sum 컬럼을 int로 변환"""
        for col in [self.total1_col, self.total2_col, self.sum_col]:
            if col in df.columns:
                # NaN 값을 None으로 변환하고, 숫자는 int로 변환
                df[col] = df[col].apply(lambda x: int(float(x)) if pd.notna(x) and x != '' and str(x).replace('-', '').replace('.', '').isdigit() else None)
        return df
    
    def _process_single_file(self, file_path):
        """단일 CSV 파일 후처리
        
        의도: 개별 CSV 파일을 읽어서 데이터 정제 및 검증 수행
        
        Args:
            file_path: 처리할 CSV 파일 경로
        
        Returns:
            처리 결과 딕셔너리
        """
        folder_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            df = pd.read_csv(file_path, encoding=self.csv_encoding)
            logger.debug(f"처리 중: {folder_name} (shape: {df.shape})")
            
            # 먼저 int로 변환
            df = self._convert_to_int(df)
            df_cleaned = self._remove_empty_rows(df)
            df_final = self._calculate_sum_column(df_cleaned)
            exceptions = self._detect_exceptions(df_final, folder_name)
            
            df_final.to_csv(file_path, index=False, encoding=self.csv_encoding)
            
            return {
                'folder': folder_name,
                'original_rows': len(df),
                'processed_rows': len(df_final),
                'exceptions': exceptions,
                'output_file': file_path
            }
        except Exception as e:
            print(f"  ❌ 처리 실패: {folder_name} - {e}")
            return {
                'folder': folder_name,
                'error': str(e),
                'exceptions': []
            }
    
    def process_all_files(self, target_files=None):
        """모든 CSV 파일 후처리
        
        의도: 입력 폴더의 모든 CSV 파일을 후처리하여 정제된 데이터 생성
        
        Args:
            target_files: 처리할 파일명 리스트 (None이면 전체)
        
        Returns:
            후처리 결과 리스트
        """
        if not os.path.exists(self.input_folder):
            logger.error(f"입력 폴더 없음: {self.input_folder}")
            return False
        
        csv_files = [f for f in os.listdir(self.input_folder) if f.endswith('.csv')]
        
        if not csv_files:
            logger.warning(f"CSV 파일 없음: {self.input_folder}")
            return False
        
        try:
            logger.info(f"{self.case} 후처리 시작: {self.input_folder}")
            results = []
            
            for csv_file in csv_files:
                folder_name = os.path.splitext(csv_file)[0]
                # target_files가 지정되면 해당 파일만 처리
                if target_files and folder_name not in target_files:
                    continue
                    
                file_path = os.path.join(self.input_folder, csv_file)
                result = self._process_single_file(file_path)
                results.append(result)
            
            total_exceptions = sum(len(r.get('exceptions', [])) for r in results)
            logger.info(f"{self.case} 후처리 완료: {len(results)}개 파일, {total_exceptions}개 예외")
            
            return results
        except Exception as e:
            logger.error(f"{self.case} 후처리 중 오류 발생: {e}")
            return []
