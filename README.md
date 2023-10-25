# imbalanced-ddhs

# Financial--Imbalanced-Data-With-Deep-Density-Hybrid-Sampling-

imblanced 상황에서 Hybrid sampling 방법론 중 하나인 
DDHS 논문의 아키텍쳐를 기반으로 금융 데이터에 맞게 contribution을 추가하여 구현 

## <아키텍쳐>
(실험을 통해 변경가능)

1) 연속형, 범주형 판단
2) small, big class에 따라 분리
3) 각 class에 따로 연속형만 GMM(k=2) fitting 후 상위 N % 만 Filtering
4) ctgan으로 연속형 변수만 생성 후 Density 기반의 적합성 판단
5) 생성된 연속형 변수와 cosine similarity 기반의 가장 가까운 기존 변수의 범주형값 카피

## <버전 규칙>

v[1].[2].[3]

[1] : 내부에 들여오는 버전
[2] : major한 변화 (architecture, def 추가 등)
[3] : minor한 변화 

##  <needs list>

1. 각 단계별 실험
2. 코드리뷰 - 코드 효율성, 가독성
3. 성과 측정 measure -( NIA, 생성전, 생성후 모델의 성능변화)
4. 학습시킬 데이터셋 

##  <to do list>

0. 개발스타일? -> 개발환경 통일(ex) 파이참 
1. 순차적으로 구조 점검 및 구현 -> 인덱스로 연속형, 범주형 합치는 부분, 각 함수에서 다음으로 넘어가는 부분  
2. 범주형 변수 찾아내는 부분 추가 
3. ctgan, gmm 버전 오류해결 : 기존 패키지들 회사버전 확인-> 맞추기
