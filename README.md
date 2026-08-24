# smu-ai-rag — ISMS-P 인증 질의응답 에이전트

상명대 계절학기 AI 서비스 실무 개발 과정 팀 프로젝트.

ISMS-P(정보보호 및 개인정보보호 관리체계) 인증 관련 질문에 답하는 LangGraph 기반 에이전트입니다.
질문 성격을 먼저 판별해 **세 갈래**로 나눠 처리합니다.

| 질문 유형 | 경로 | 데이터 소스 |
|---|---|---|
| "어떻게 이행하나요?" 같은 설명 | 벡터 검색 (RAG) | ISMS-P 인증기준 안내서 PDF |
| "결함이 몇 건인가요?" 같은 수치 | Text2SQL | `datasets/*.csv` (Supabase 또는 로컬) |
| 인사·사용법·고수준 소개 | 바로 답변 | 없음 |

세 갈래 모두 답변 생성 후 **근거 대조 검증**(`final_query`)을 거치고, 통과한 답변만 사용자에게 노출됩니다.

### 사용 데이터

- **`datasets/ISMS_P.pdf`** — 「정보보호 및 개인정보보호 관리체계(ISMS-P) 인증기준 안내서」
  개인정보보호위원회, 2023. 11. · 264쪽
- **`datasets/isms_items.csv`** — 인증기준 항목별 상세내용·주요 확인사항 (65행)
- **`datasets/isms_defects.csv`** — 통제항목별 결함 통계 (190행, 총 174건)

---

## 빠른 시작

```bash
# 1. 의존성 설치 (.venv 자동 생성)
uv sync

# 2. 환경 변수 파일 준비
cp .env.example .env      # Windows PowerShell: Copy-Item .env.example .env
#   → .env를 열어 API 키와 접속 정보를 채웁니다

# 3. Streamlit 데모 실행
uv run streamlit run src/demo/streamlit_example.py
```

브라우저가 자동으로 열리지 않으면 터미널에 표시되는 `http://localhost:8501` 로 접속하세요.

> **요구 사항** — Python 3.11 이상 3.14 미만, [uv](https://docs.astral.sh/uv/) 설치.
> `uv`가 없다면: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows)

---

## 환경 변수

`.env.example`을 복사해 `.env`를 만들고 아래 값을 채웁니다. **`.env`는 절대 커밋하지 마세요** (`.gitignore`에 등록돼 있습니다).

| 변수 | 필수 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | 필수 | LLM 및 임베딩 호출용 |
| `QDRANT_URL` | 벡터 검색 시 | Qdrant 클러스터 주소 |
| `QDRANT_API_KEY` | 벡터 검색 시 | Qdrant 인증 키 |
| `QDRANT_COLLECTION_NAME` | 선택 | 기본값 `isms_p` |
| `ISMS_DB_SOURCE` | 선택 | `supabase`(기본값) 또는 `local` |
| `SUPABASE_DB_URL` | `supabase` 사용 시 | PostgreSQL 연결 문자열 |
| `PYTHONUTF8` | 권장 | Windows 콘솔 한글 깨짐 방지 (`1`) |

---

## 실행 방법

### 1. Streamlit 데모 앱 (권장)

채팅 UI와 워크플로 정보 패널, 결함 통계 시각화를 함께 제공합니다.
사이드바에 예시 질문 버튼이 준비돼 있어 클릭만으로 실행됩니다.

```bash
uv run streamlit run src/demo/streamlit_example.py
```

포트를 바꾸려면:

```bash
uv run streamlit run src/demo/streamlit_example.py --server.port 8502
```

### 2. LangGraph Studio (그래프 시각화·디버깅)

노드 단위 실행 흐름과 상태 변화를 눈으로 확인할 때 사용합니다.

```bash
uv run langgraph dev
```

`langgraph.json`이 `src/ai/graph.py:graph`를 진입점으로 사용합니다.
서버를 띄우기 전에 설정 파일만 검사하려면:

```bash
uv run langgraph validate
```

> **Windows 참고** — `langgraph --help`는 도움말에 포함된 이모지 때문에 기본 콘솔(cp949)에서
> 인코딩 오류로 중단됩니다. `PYTHONUTF8=1 uv run langgraph --help`로 실행하세요.
> `langgraph dev` 자체는 영향을 받지 않습니다.

### 3. 파이썬에서 직접 호출

```bash
uv run python -c "
import sys; sys.path.insert(0, 'src')
from ai import create_graph
graph = create_graph()
result = graph.invoke({'messages': [{'role': 'user', 'content': '결함이 가장 많은 통제영역은?'}]})
print('의도:', result.get('intent'))
print('답변:', result['messages'][-1].content)
"
```

### 4. 라우팅 규칙 점검 (네트워크·API 키 불필요)

LLM을 호출하기 전 규칙 단계가 질문을 어디로 보내는지만 빠르게 확인합니다.

```bash
uv run python -c "
import sys; sys.path.insert(0, 'src')
from ai.nodes import _is_database_question, _is_vector_question
for q in [
    '10.4 접근통제 영역의 결함 건수는 몇 개야?',
    '가장 결함이 많이 발생한 통제영역 Top 3를 알려줘.',
    '비밀번호 작성 규칙에 대한 가이드라인을 알려줘.',
    '안녕하세요.',
]:
    route = 'database' if _is_database_question(q) else ('vector' if _is_vector_question(q) else 'LLM 판단')
    print(f'{route:10s} | {q}')
"
```

기대 출력:

```
database   | 10.4 접근통제 영역의 결함 건수는 몇 개야?
database   | 가장 결함이 많이 발생한 통제영역 Top 3를 알려줘.
vector     | 비밀번호 작성 규칙에 대한 가이드라인을 알려줘.
LLM 판단     | 안녕하세요.
```

---

## 데이터 소스 전환

`ISMS_DB_SOURCE`로 정형 데이터의 출처를 바꿉니다. **네트워크가 불안한 환경(발표·데모)에서는 `local`을 쓰세요.**

```bash
# Supabase(PostgreSQL) 조회 — 기본값
ISMS_DB_SOURCE=supabase

# datasets의 CSV를 인메모리 SQLite로 적재해 조회 (오프라인 가능)
ISMS_DB_SOURCE=local
```

`local`은 시작할 때 `isms_items.csv`, `isms_defects.csv`를 같은 이름의 테이블로 적재합니다.
두 모드 모두 동일한 SQL로 동일한 결과를 반환하는 것을 확인했습니다.

> 벡터 검색은 이 설정과 무관하게 항상 Qdrant를 사용합니다. 오프라인에서는 문서 검색 경로가 동작하지 않습니다.

---

## 프로젝트 구조

```
src/
  ai/
    graph.py       LangGraph 그래프 정의 (노드 7개, 조건부 분기)
    nodes.py       라우팅 · 검색 · 답변 생성 · 근거 검증 노드
    text2sql.py    자연어 → SQL 변환 및 실행 (Supabase / 로컬 SQLite)
    retriever.py   Qdrant 벡터 검색 (Parent-Child 청킹)
    state.py       AgentState 정의
  demo/
    streamlit_example.py   채팅 UI 데모 앱
datasets/          PDF 원문과 CSV 2종
examples/          수업 진행용 노트북 템플릿
docs/              발표자료 (presentation.html)
```

---

## 문제 해결

**문서 검색이 항상 빈 결과입니다**
Qdrant 컬렉션명을 먼저 확인하세요. 실제 컬렉션은 `isms_p`입니다.
`QDRANT_COLLECTION_NAME`이 다른 값으로 설정돼 있으면 존재하지 않는 컬렉션을 조회하게 됩니다.

**설정을 바꿨는데 반영되지 않습니다**
retriever와 Text2SQL 엔진은 모듈 전역에 캐시됩니다. **서버를 완전히 종료 후 재시작**해야 합니다.

**콘솔에 한글이 `????`로 보입니다**
Windows 콘솔 인코딩 문제일 뿐 데이터는 정상입니다. `.env`에 `PYTHONUTF8=1`을 두거나
명령 앞에 `PYTHONIOENCODING=utf-8`을 붙여 실행하세요.

**같은 질문에 답이 매번 다릅니다**
최소·최대·순위 질문은 고정 SQL로 처리하도록 되어 있습니다(`text2sql.py`의 `_get_deterministic_sql`).
그 외 질문은 LLM이 SQL을 생성하므로 표현이 달라질 수 있습니다.

**표에 있는 내용인데 "근거가 부족하다"고 답합니다**
질문이 문서 검색으로 잘못 라우팅됐을 가능성이 큽니다. 위 **라우팅 규칙 점검**으로 확인하고,
필요하면 `nodes.py`의 `DATABASE_FIELD_TERMS` 또는 `DATABASE_AGGREGATION_TERMS`에 신호어를 추가하세요.
