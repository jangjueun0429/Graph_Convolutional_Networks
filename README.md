📘 Graph Convolutional Networks – MUTAG (R-GCN) & WordNet18 (TransE)

이 저장소는 그래프 신경망(GNN) 과 지식그래프 임베딩(KGE) 기본 모델을 실습하며 정리한 학습 기록이다.
두 개의 주요 과제를 포함하고 있다:

MUTAG 그래프 분자 데이터셋을 학습하는 R-GCN(Relational GCN)

WordNet18 지식그래프에 대해 TransE 임베딩 모델 구현

📁 Repository Structure
Graph_Convolutional_Networks/
│
├── mutag_rgcn.py            # MUTAG 데이터셋 + R-GCN 모델
├── wordnet18_transe.py      # WordNet18 + TransE 모델
├── .gitignore
└── README.md                # (바로 이 파일)

📌 Assignment 1 — MUTAG + Relational GCN
■ 1. MUTAG Dataset

188개의 화학 분자 그래프로 구성

그래프 분류 문제
목표: 분자가 돌연변이를 유발(mutagenic)하는지(0/1) 예측

각 그래프 구성 요소:

노드: 원자(atom), 7차원 원자 특성 특징

엣지(edge): 결합 종류(bond type), one-hot encoding (길이 4)

라벨: 0 또는 1

■ 2. R-GCN (Relational Graph Convolutional Network)

MUTAG는 엣지마다 relation type(결합 종류)이 있음
→ 일반 GCN 보다 R-GCN이 더 적합.

R-GCN 메시지 패싱 핵심 공식:

h_i^(l+1) = σ( Σ_r Σ_{j ∈ N_r(i)} 1 / c_{i,r}  W_r^(l) h_j^(l) )


여기서:

r = relation type

W_r = relation별 파라미터

N_r(i) = relation r로 연결된 이웃 노드들

즉, 결합 종류에 따라 다른 weight를 사용하는 GCN.

■ 3. 구현 내용 요약

PyG의 RGCNConv 사용

모델 구조:

RGCNConv → ReLU → RGCNConv → ReLU → GlobalMeanPool → Linear


edge_attr.argmax(dim=1) 로 edge type(index) 추출

Train/Test (80/20) split

CrossEntropyLoss + Adam optimizer

■ 4. 실행 방법
python mutag_rgcn.py

■ 5. 출력 예시 (요약)

데이터셋 분석

관계 타입 수 출력

Epoch별 Loss / Accuracy

최종 Test Accuracy 약 75~85% 범위

📌 Assignment 2 — WordNet18 + TransE
■ 1. WordNet18 Dataset

WordNet18은 지식그래프(Knowledge Graph) 데이터셋이다.

각 triple은 (head, relation, tail) 형태:

(h, r, t)


예:

(dog, is_a, animal)


데이터 구성:

약 40,000개 이상의 triple

엔티티 수 약 40k

relation 수 18개

■ 2. TransE 모델

TransE의 핵심 개념:

좋은 triple은
e_h + e_r ≈ e_t

즉, head + relation ≈ tail 벡터가 되도록 임베딩을 학습한다.

스코어 함수:

score(h, r, t) = - || e_h + e_r − e_t ||_1

Negative Sampling

tail을 랜덤 엔티티로 바꿔서 (h, r, t') 만들기

margin ranking loss 사용:

max(0, margin + score_neg − score_pos)

■ 3. 구현 내용 요약

Train/Valid/Test triples 모두 PyG에서 추출

nn.Embedding으로 entity / relation embedding 생성

negative sampling 직접 구현

margin-ranking loss 사용

Adam optimizer로 학습

■ 4. 실행 방법
python wordnet18_transe.py

🔧 Development Environment

macOS (Intel)

Python 3.8 (virtual env)

PyTorch 2.2.2 (CPU)

PyTorch Geometric 2.6.1
(pyg-lib / torch-sparse 경고는 정상)

🎯 What I Learned
✔ 1. 그래프 신경망의 기본 동작 원리

메시지 패싱(Message Passing)

relation-aware aggregation

graph-level readout(global pooling)

✔ 2. Knowledge Graph Embedding 방법론

triple 구조

negative sampling

margin ranking loss

✔ 3. PyTorch Geometric 실습 능력

DataLoader / Dataset 다루기

edge_attr → edge_type 변환

RGCNConv 사용법

✔ 4. GitHub 프로젝트 정리

.venv / data 제외

깔끔한 프로젝트 구조 관리
