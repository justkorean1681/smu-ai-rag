# smu-ai-rag — ISMS-P RAG 에이전트

상명대 계절학기 팀 프로젝트. ISMS-P 인증기준 안내서(PDF)와 결함 통계 CSV를 근거로
답변하는 LangGraph 기반 RAG 에이전트입니다.

## 사용자 작업 규칙 (표준 지침)

- **실행은 사용자가 직접 한다.** 에이전트는 코드 수정까지만 하고, LangGraph 서버 구동이나
  질의 테스트는 사용자가 수행합니다. 읽기 전용 확인은 해도 됩니다.
- **API 키·연결 문자열·비밀번호를 절대 출력하지 않는다.** 오류를 진단할 때도 상태코드 수준만
  주고받습니다.
- **보안에 영향을 주는 조작은 하지 않는다.**
- **원본 과제 템플릿 구조에서 크게 벗어나지 않는다.** `examples/`의 노트북이 기준 가이드입니다.
- **커밋·푸시는 사용자가 요청할 때만 한다.**
- 팀 프로젝트라 `main`과 `member1` 두 브랜치를 같은 커밋으로 맞춰 푸시하는 흐름을 씁니다.

## 구조

| 파일 | 역할 |
|---|---|
| `src/ai/graph.py` | LangGraph 그래프 정의 (`langgraph.json`의 진입점) |
| `src/ai/nodes.py` | 라우팅·검색·답변 생성·검증 노드 |
| `src/ai/text2sql.py` | 자연어 → SQL 변환 및 실행 |
| `src/ai/retriever.py` | Qdrant 벡터 검색 |
| `src/ai/state.py` | `AgentState` 정의 |

질문은 `classify_intent`에서 `general` / `database` / `vector` 셋 중 하나로 분류되어
각 경로를 탄 뒤, `generate_answer` → `final_query`(근거 충실성 검증)를 거칩니다.
검증 실패 시 1회만 재생성하고, **검증을 통과한 최종 답변만** `messages`에 추가됩니다
(초안이 사용자에게 노출되면 같은 질문에 여러 답이 보이는 문제가 생깁니다).

## 데이터 소스

- **벡터**: Qdrant 컬렉션 `isms_p` — 1,855개 벡터, `text-embedding-3-large`(3072차원),
  메타데이터 `page` / `parent_id` / `source`. `QDRANT_COLLECTION_NAME`으로 덮어쓸 수 있습니다.
- **정형**: `ISMS_DB_SOURCE` 환경변수로 전환
  - `supabase`(기본값) — `SUPABASE_DB_URL` 사용. `isms_items` 65행, `isms_defects` 190행
  - `local` — `datasets/*.csv`를 인메모리 SQLite로 적재 (`StaticPool` 사용)

## 이 프로젝트에서 실제로 겪은 함정

1. **Qdrant 컬렉션명 불일치.** 코드가 존재하지 않는 `team1_isms_p`를 보고 있어 벡터 검색이
   전부 빈 결과였습니다. 실제 컬렉션은 `isms_p`입니다. 벡터 검색이 "근거 없음"으로 끝나면
   컬렉션명부터 의심하세요.
2. **Supabase 한글 컬럼명은 정상입니다.** 콘솔에 `����`로 보이는 건 Windows Python 출력
   인코딩(cp949) 문제일 뿐입니다. 확인할 때 `PYTHONIOENCODING=utf-8`을 붙이세요.
   테이블을 다시 업로드할 필요 없습니다.
3. **두 테이블을 JOIN하지 마세요.** `isms_items`와 `isms_defects`는 서로 다른 인증체계
   (ISMS / ISMS-P)의 번호가 섞여 있어 신뢰할 수 있는 공통 키가 없습니다. 항목번호 접두사로
   JOIN하면 `1.1.1`이 `정책의 수립`과 `정책의 승인`에 동시에 잘못 연결됩니다.
4. **최소·최대 결함 질문은 고정 SQL로 처리합니다** (`text2sql.py`의 `_get_deterministic_sql`).
   LLM에 맡기면 매번 다른 SQL이 나오고 `LIMIT 1`로 동률을 잘라버립니다. 실제 값:
   0건 포함 최소는 **0건·동률 138개**, 0건 제외 최소는 **1건·동률 21개**.
5. **`SQLDatabase`의 `max_string_length`가 기본값이면 긴 결과가 잘립니다.** 2000으로 지정해
   두었습니다. 동률 항목은 전체를 나열하지 않고 개수 + 대표 10개만 반환합니다.
6. **라우팅은 LLM 이전에 규칙으로 먼저 거릅니다** (`nodes.py`의 `_is_database_question`,
   `_is_vector_question`). LLM만 쓰면 "상세내용", "주요 확인사항" 같은 명백한 CSV 조회를
   벡터 검색으로 넘겨버렸습니다.
7. **요청 단위 상태 초기화가 필요합니다.** `classify_intent`가 이전 턴의 검색 결과와 검증
   피드백을 지웁니다. 안 지우면 새 질문에 이전 근거가 섞입니다.

## 확인 방법

```bash
# 라우팅 규칙 (LLM 호출 없음, 네트워크 불필요)
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import sys; sys.path.insert(0,'src')
from ai.nodes import _is_database_question, _is_vector_question
print(_is_database_question('1.1.1 항목의 주요 확인사항은?'))
print(_is_vector_question('비밀번호 변경 주기 가이드라인은?'))"

# 전체 동작은 사용자가 직접 실행
langgraph dev
```

> retriever와 text2sql 엔진은 모듈 전역에 캐시됩니다. 설정을 바꿨으면 **LangGraph 서버를
> 완전히 재시작**해야 반영됩니다.
