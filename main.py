"""
골프 스코어카드 인식 메인 실행 파일
"""

import sys
from datetime import datetime
from modules.preprocessing import ImagePreprocessor
from modules.ocr_converter import OCRConverter
from modules.postprocessing import PostProcessor
from modules.claude_converter import ClaudeConverter
from config import CASES

def main():
    """메인 실행 함수"""
    start_time = datetime.now()
    print("🏌️ 골프 스코어카드 인식 시작...\n")
    
    try:
        all_exception_files = []
        
        # 각 케이스별로 처리
        for case in CASES:
            print(f"\n{'='*60}")
            print(f"🏌️ {case.upper()} 처리 시작...")
            print(f"{'='*60}")
            
            # 1단계: 전처리
            print(f"🔄 [1/3] {case} 전처리 중...")
            preprocessor = ImagePreprocessor(case=case)
            if not preprocessor.process_all_images():
                print(f"❌ {case} 전처리 실패")
                continue
            
            # 2단계: OCR 변환
            print(f"\n🔄 [2/3] {case} OCR 변환 중...")
            converter = OCRConverter(case=case)
            if not converter.convert_all_folders():
                print(f"❌ {case} OCR 변환 실패")
                continue
            
            # 3단계: 후처리
            print(f"\n🔄 [3/3] {case} 후처리 중...")
            processor = PostProcessor(case=case)
            results = processor.process_all_files()
            if not results:
                print(f"❌ {case} 후처리 실패")
                continue
            
            # 예외 파일 수집
            case_exception_files = []
            for result in results:
                if 'exceptions' in result and len(result['exceptions']) > 0:
                    case_exception_files.append(result)
                    all_exception_files.append(result)
            
            if case_exception_files:
                print(f"⚠️  {case} 예외 발견: {len(case_exception_files)}개 파일")
            else:
                print(f"✅ {case} 예외 없음")
        
        # 4단계: 예외 처리
        if all_exception_files:
            print(f"\n🔄 [4/4] 예외 처리 중...")
            print(f"⚠️  총 예외 파일: {len(all_exception_files)}개")
            
            # 예외 파일들만 재처리
            retry_success = 0
            for exception_file in all_exception_files:
                folder_name = exception_file['folder']
                original_case = exception_file.get('case', 'case1')
                
                print(f"\n🔄 {folder_name} 재처리 시도...")
                
                # 다른 케이스들로 재처리
                for retry_case in CASES:
                    if retry_case == original_case:
                        continue
                    
                    print(f"  📝 {retry_case}로 재처리 중...")
                    
                    try:
                        # 실제 재처리 로직
                        retry_preprocessor = ImagePreprocessor(case=retry_case)
                        retry_converter = OCRConverter(case=retry_case)
                        retry_processor = PostProcessor(case=retry_case)
                        
                        # 특정 폴더만 재처리하는 로직
                        # 1. 전처리 (특정 폴더만)
                        if retry_preprocessor.process_specific_folder(folder_name):
                            # 2. OCR 변환 (특정 폴더만)
                            if retry_converter.convert_specific_folder(folder_name):
                                # 3. 후처리 (특정 폴더만)
                                retry_results = retry_processor.process_specific_file(folder_name)
                                
                                # 재처리 결과 확인
                                if retry_results and 'exceptions' in retry_results and len(retry_results['exceptions']) == 0:
                                    print(f"  ✅ {retry_case} 재처리 성공!")
                                    retry_success += 1
                                    break
                                else:
                                    print(f"  ⚠️ {retry_case} 재처리 후에도 예외 존재")
                        else:
                            print(f"  ❌ {retry_case} 전처리 실패")
                    except Exception as e:
                        print(f"  ❌ {retry_case} 재처리 중 오류: {e}")
                        continue
                
                if retry_success == 0:
                    print(f"  ❌ {folder_name} 모든 케이스 재처리 실패")
            
            # 여전히 실패한 파일들에 대해 Claude 변환
            remaining_exceptions = len(all_exception_files) - retry_success
            if remaining_exceptions > 0:
                print(f"\n🔄 [5/5] Claude API 변환 ({remaining_exceptions}개 파일)")
                print(f"⚠️  예상 비용: {remaining_exceptions * 76:.2f}원")
                user_input = input("계속 진행하시겠습니까? (y/n): ").lower().strip()
                
                if user_input == 'y':
                    for case in CASES:
                        claude_converter = ClaudeConverter(case=case)
                        case_exception_names = [r['folder'] for r in all_exception_files 
                                             if r.get('case', 'case1') == case]
                        if case_exception_names:
                            claude_converter.convert_specific_images(case_exception_names)
                else:
                    print("⚠️  Claude 변환 건너뜀")
        else:
            print("\n✅ 모든 케이스에서 예외 없음")
        
        # 최종 결과
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"\n{'='*50}")
        print(f"⏱️  총 작업 시간: {elapsed_time:.2f}초")
        
        if all_exception_files:
            print(f"⚠️  처리되지 못한 케이스: {len(all_exception_files)}개")
        else:
            print("✅ 모든 데이터가 성공적으로 처리되었습니다!")
        
        print(f"{'='*50}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  사용자에 의해 중단됨")
        return False
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
