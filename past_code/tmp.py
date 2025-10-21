"""
case99 폴더의 CSV 데이터 후처리 스크립트

의도: case99 폴더의 모든 CSV 파일에 대해 절대점수 → 상대점수 변환 후처리 적용
- claude_api_test_05.py의 후처리 로직을 재사용
- diff_sum < 36인 플레이어는 절대점수로 판단하여 상대점수로 변환
- 모든 플레이어의 total1, total2, sum을 절대점수 기준으로 재계산
"""

import pandas as pd
import os
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# case99 폴더 경로
CASE99_FOLDER = os.path.join(PROJECT_ROOT, "data", "result_convert", "case99")

def load_csv_data(file_path):
    """CSV 파일을 로드하고 전처리
    
    case99 폴더의 CSV 구조:
    - 첫 번째 행: PAR 데이터 (1~18홀 + total1, total2, sum)
    - 나머지 행들: 플레이어 데이터 (1~18홀 + total1, total2, sum)
    """
    try:
        df = pd.read_csv(file_path)
        # 빈 행 제거
        df = df.dropna(how='all')
        
        # 첫 번째 행이 PAR, 나머지가 플레이어인 구조로 처리
        if len(df) < 2:
            logger.warning(f"데이터가 부족합니다: {file_path}")
            return None
        
        # 첫 번째 행을 PAR로, 나머지를 플레이어로 처리
        par_row = df.iloc[0].values
        player_rows = df.iloc[1:].values
        
        # 데이터 구조: 1~18홀 + total1, total2, sum (총 21개)
        if len(par_row) != 21:
            logger.warning(f"예상된 컬럼 수(21)와 다릅니다: {len(par_row)}")
            return None
        
        # 1~18홀만 추출 (마지막 3개는 total1, total2, sum이므로 제외)
        par_scores = par_row[:18]
        player_scores_list = [row[:18] for row in player_rows]
        
        # DataFrame 재구성
        data = {
            '홀': list(range(1, 19)),  # 1~18홀
            'PAR': par_scores
        }
        
        # 플레이어 데이터 추가
        for i, player_scores in enumerate(player_scores_list):
            player_name = f"Player{i+1}"
            data[player_name] = player_scores
        
        result_df = pd.DataFrame(data)
        logger.info(f"로드 완료: PAR 1행, 플레이어 {len(player_scores_list)}행, 총 18홀")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def convert_to_csv_format(df, image_name):
    """DataFrame을 CSV 저장용 형태로 변환
    
    의도: case1_02.csv 형태로 변환 (Transpose + total1, total2, sum 추가)
    추가로 절대점수로 표시된 플레이어 데이터를 상대점수로 변환하는 후처리 적용
    
    후처리 로직:
    1. 각 플레이어의 diff_sum 계산 (|스코어 - PAR|의 합)
    2. diff_sum < 36인 플레이어는 절대점수로 판단하여 상대점수로 변환
    3. 모든 플레이어의 total1, total2, sum을 절대점수 기준으로 재계산
    
    Args:
        df: 원본 DataFrame (홀, PAR, 플레이어들)
        image_name: 이미지 파일명 (확장자 제외)
    
    Returns:
        변환된 DataFrame 또는 None (실패 시)
    """
    try:
        # PAR 행과 플레이어 행들만 추출
        par_row = df['PAR'].values
        player_rows = []
        player_names = []
        
        # 플레이어 컬럼들 추출 (PAR 제외)
        for col in df.columns:
            if col not in ['홀', 'PAR']:
                player_rows.append(df[col].values)
                player_names.append(col)
        
        if not player_rows:
            logger.warning(f"'{image_name}'에서 플레이어 데이터가 없습니다.")
            return None
        
        # 후처리: 절대점수 → 상대점수 변환
        processed_player_rows = []
        logger.info(f"총 {len(player_names)}명의 플레이어 처리 시작")
        
        for i, (player_name, player_scores) in enumerate(zip(player_names, player_rows)):
            logger.info(f"플레이어 {i+1}/{len(player_names)}: '{player_name}' 처리 중")
            
            # 1단계: 0 이하 숫자 확인
            has_zero_or_negative = any(score <= 0 for score in player_scores)
            
            # 2단계: 평균값 기반 판단
            par_average = sum(par_row) / len(par_row)  # PAR 평균
            player_average = sum(player_scores) / len(player_scores)  # 플레이어 평균
            threshold = par_average - 0.5  # 임계값
            
            logger.info(f"플레이어 '{player_name}' PAR 평균: {par_average:.2f}, 플레이어 평균: {player_average:.2f}, 임계값: {threshold:.2f}")
            
            if has_zero_or_negative:
                # 0 이하 숫자가 있으면 상대점수로 판단 (변환 안함)
                logger.info(f"플레이어 '{player_name}' 상대점수로 판단 (0 이하 숫자 포함)")
                processed_player_rows.append(player_scores)
                logger.info(f"유지된 스코어: {player_scores[:5]}... (처음 5개)")
            elif player_average > threshold:
                # 플레이어 평균 > (PAR 평균 - 0.5)이면 절대점수로 판단하여 상대점수로 변환
                logger.info(f"플레이어 '{player_name}' 절대점수 → 상대점수 변환 (플레이어 평균: {player_average:.2f} > 임계값: {threshold:.2f})")
                relative_scores = [score - par for score, par in zip(player_scores, par_row)]
                processed_player_rows.append(relative_scores)
                logger.info(f"변환된 스코어: {relative_scores[:5]}... (처음 5개)")
            else:
                # 그 외는 상대점수로 판단하여 변환 안함
                logger.info(f"플레이어 '{player_name}' 상대점수 유지 (플레이어 평균: {player_average:.2f} <= 임계값: {threshold:.2f})")
                processed_player_rows.append(player_scores)
                logger.info(f"유지된 스코어: {player_scores[:5]}... (처음 5개)")
        
        logger.info(f"후처리 완료: {len(processed_player_rows)}명의 플레이어 처리됨")
        
        # 새로운 DataFrame 생성 (Transpose)
        data = {}
        
        # 홀별 컬럼 추가 (1홀, 2홀, ..., 18홀)
        for i in range(len(par_row)):
            data[f"{i+1}홀"] = [par_row[i]] + [player[i] for player in processed_player_rows]
        
        # total1, total2, sum 초기화
        data['total1'] = []
        data['total2'] = []
        data['sum'] = []
        
        # PAR 행의 total1, total2, sum 계산
        par_total1 = sum(par_row[:9])
        par_total2 = sum(par_row[9:18])
        par_sum = par_total1 + par_total2
        
        data['total1'].append(par_total1)
        data['total2'].append(par_total2)
        data['sum'].append(par_sum)
        
        # 3단계: 각 플레이어의 total1, total2, sum 재계산 (절대점수 기준)
        for player_scores in processed_player_rows:
            # 플레이어의 1~9홀, 10~18홀 합계 계산
            player_total1 = sum(player_scores[:9])   # 전반 9홀 합계
            player_total2 = sum(player_scores[9:18]) # 후반 9홀 합계
            
            # 절대점수 기준으로 total 계산
            total1 = par_total1 + player_total1  # PAR의 total1 + 플레이어의 1~9홀 합계
            total2 = par_total2 + player_total2  # PAR의 total2 + 플레이어의 10~18홀 합계
            total_sum = total1 + total2
            
            data['total1'].append(total1)
            data['total2'].append(total2)
            data['sum'].append(total_sum)
            
            logger.info(f"플레이어 total 재계산: total1={total1}, total2={total2}, sum={total_sum}")
        
        result_df = pd.DataFrame(data)
        
        # CSV 파일로 저장
        csv_path = os.path.join(CASE99_FOLDER, f"{image_name}.csv")
        result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"CSV 저장 완료: {csv_path}")
        return result_df
        
    except Exception as e:
        logger.error(f"CSV 변환 오류 ({image_name}): {e}")
        return None

def process_case99_files():
    """case99 폴더의 모든 CSV 파일을 후처리"""
    print("=" * 60)
    print("case99 폴더 CSV 파일 후처리 시작")
    print("=" * 60)
    print(f"처리 폴더: {CASE99_FOLDER}")
    print("-" * 60)
    
    # case99 폴더 존재 확인
    if not os.path.exists(CASE99_FOLDER):
        logger.error(f"case99 폴더가 존재하지 않습니다: {CASE99_FOLDER}")
        return
    
    # CSV 파일 목록 가져오기
    csv_files = list(Path(CASE99_FOLDER).glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"case99 폴더에 CSV 파일이 없습니다: {CASE99_FOLDER}")
        return
    
    logger.info(f"처리할 CSV 파일 수: {len(csv_files)}")
    
    # 처리 결과 통계
    processed_count = 0
    failed_count = 0
    
    # 각 CSV 파일 처리
    for i, csv_file in enumerate(csv_files, 1):
        file_name = csv_file.name
        file_name_no_ext = csv_file.stem
        
        print(f"\n[{i}/{len(csv_files)}] 처리 중: {file_name}")
        
        try:
            # CSV 파일 로드
            df = load_csv_data(csv_file)
            
            if df is None:
                failed_count += 1
                print(f"  ❌ 실패: CSV 파일 로드 실패")
                continue
            
            print(f"  📊 로드 완료: {df.shape[0]}행, {df.shape[1]}열")
            
            # 후처리 적용
            result_df = convert_to_csv_format(df, file_name_no_ext)
            
            if result_df is not None:
                processed_count += 1
                print(f"  ✅ 성공: {result_df.shape[0]}행, {result_df.shape[1]}열")
                print(f"  📊 결과 미리보기:")
                print(result_df.to_string(index=False))
            else:
                failed_count += 1
                print(f"  ❌ 실패: 후처리 변환 실패")
                
        except Exception as e:
            failed_count += 1
            logger.error(f"오류 발생 ({file_name}): {e}")
            print(f"  ❌ 실패: {e}")
            continue
    
    # 최종 결과 출력
    print("\n" + "=" * 60)
    print("후처리 완료!")
    print(f"성공: {processed_count}개")
    print(f"실패: {failed_count}개")
    print("=" * 60)

if __name__ == "__main__":
    process_case99_files()
