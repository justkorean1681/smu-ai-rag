import csv
import re
from functools import lru_cache
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ai.state import AgentState
from ai.retriever import get_retriever
from ai.text2sql import get_text2sql_engine
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List

load_dotenv()

llm = init_chat_model("gpt-5.4-mini")

_retriever = None
_text2sql_engine = None

DATABASE_FIELD_TERMS = (
    "상세내용",
    "상세 내용",
    "주요 확인사항",
    "주요확인사항",
    "결함수",
    "결함 수",
    "항목번호",
    "항목 번호",
    "항목명",
    "분야명",
)
DATABASE_AGGREGATION_TERMS = (
    "몇 건",
    "몇 개",
    "개수",
    "건수",
    "합계",
    "평균",
    "통계",
    "비율",
    "비중",
    "가장 많",
    "가장 적",
    "순위",
    "비교",
)
VECTOR_EXPLANATION_TERMS = (
    "어떻게",
    "방법",
    "사례",
    "가이드",
    "가이드라인",
    "필수 항목",
    "필수항목",
    "점검",
    "주의사항",
    "권고",
    "규칙",
    "주기",
    "심사 관점",
    "적용",
    "이행",
    "절차",
    "왜",
    "의미",
)


class VectorSearchQuery(BaseModel):
    """벡터 검색을 위한 쿼리 분석 결과"""
    optimized_query: str = Field(
        description="검색에 최적화된 쿼리. 핵심 키워드를 포함하고 명확하게 작성."
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="메타데이터 카테고리 필터. 현재 ISMS-P 문서에는 고정 카테고리가 없으므로 null을 반환."
    )


class AnswerValidation(BaseModel):
    """생성된 답변의 근거 일치 여부"""
    is_valid: bool = Field(
        description="답변의 핵심 내용이 제공된 검색 근거로 확인되면 true"
    )
    feedback: str = Field(
        description="검증 결과. 부적합하면 재생성에 필요한 구체적인 수정 지침"
    )


def get_cached_retriever():
    """캐시된 retriever 인스턴스 반환 (lazy initialization)"""
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


def get_cached_text2sql_engine():
    """캐시된 text2sql_engine 인스턴스 반환 (lazy initialization)"""
    global _text2sql_engine
    if _text2sql_engine is None:
        _text2sql_engine = get_text2sql_engine()
    return _text2sql_engine


def _compact_text(value: str) -> str:
    """라우팅 비교를 위해 공백과 영문 대소문자를 정규화합니다."""
    return re.sub(r"\s+", "", value).lower()


@lru_cache(maxsize=1)
def _get_dataset_terms() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """CSV에 실제 존재하는 분야명과 항목명을 라우팅 기준으로 캐시합니다."""
    csv_path = Path(__file__).resolve().parents[2] / "datasets" / "isms_items.csv"
    if not csv_path.is_file():
        return (), ()

    field_names = set()
    item_names = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            field_name = _compact_text(row.get("분야명", "").strip())
            item_name = _compact_text(row.get("항목명", "").strip())
            if field_name:
                field_names.add(field_name)
            if item_name:
                item_names.add(item_name)

    return tuple(field_names), tuple(item_names)


def _is_database_question(question: str) -> bool:
    """CSV의 명시적 필드·값 조회 질문을 LLM보다 우선하여 판별합니다."""
    compact_question = _compact_text(question)

    # CSV 컬럼을 직접 지정한 질문은 항상 정형 데이터 조회
    if any(_compact_text(term) in compact_question for term in DATABASE_FIELD_TERMS):
        return True

    # 결함에 대한 수치·통계 질문은 isms_defects 조회
    if "결함" in compact_question and any(
        _compact_text(term) in compact_question
        for term in DATABASE_AGGREGATION_TERMS
    ):
        return True

    field_names, item_names = _get_dataset_terms()
    mentions_dataset_value = any(
        term in compact_question for term in (*field_names, *item_names)
    )
    asks_for_explanation = any(
        _compact_text(term) in compact_question
        for term in VECTOR_EXPLANATION_TERMS
    )

    # CSV에 실제 존재하는 분야명/항목명의 단순 조회는 DB로 보내되,
    # 이행 방법·사례·의미를 묻는 질문은 안내서 검색에 남김
    return mentions_dataset_value and not asks_for_explanation


def _is_vector_question(question: str) -> bool:
    """가이드·이행·점검 성격의 문서 질문을 결정적으로 판별합니다."""
    compact_question = _compact_text(question)
    return any(
        _compact_text(term) in compact_question
        for term in VECTOR_EXPLANATION_TERMS
    )


def classify_intent(state: AgentState) -> AgentState:
    """
    사용자 질문의 의도를 분류하는 노드

    분류 결과:
    - 'general': 일반적인 대화나 인사
    - 'database': 데이터베이스 조회가 필요한 질문
    - 'vector': 문서 검색이 필요한 질문

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # messages에서 질문 추출
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages provided")

    # 전체 히스토리에서 마지막 사용자 메시지를 질문으로 사용
    last_human_message = next(
        (message for message in reversed(messages) if isinstance(message, HumanMessage)),
        messages[-1],
    )
    question_content = (
        last_human_message.content
        if hasattr(last_human_message, "content")
        else str(last_human_message)
    )
    question = (
        question_content
        if isinstance(question_content, str)
        else str(question_content)
    )

    system_prompt = """
당신은 ISMS-P 질의응답 에이전트의 라우팅 전문가입니다.

이전 대화 맥락을 고려하여 현재 질문을 다음 3가지 중 하나로 분류하세요:

1. 'general' - 인사, 감사, 에이전트 사용법 또는 별도 자료 조회가 필요 없는 ISMS-P의 고수준 소개
   예: "안녕하세요", "어떤 질문에 답할 수 있어?", "ISMS-P를 간단하게 소개해줘"

2. 'database' - CSV 데이터의 정확한 행 조회, 목록, 개수, 합계, 비교, 순위 또는 결함 통계가 필요한 질문
   - isms_items: 분야, 분야명, 항목번호, 항목명, 상세내용, 주요 확인사항
   - isms_defects: 통제분야, 통제영역, 통제항목, 결함수, 비율, 비중, 합계
   예: "법적요구사항 준수검토의 결함수는?", "결함이 가장 많은 통제영역은?", "1.1.1 항목의 주요 확인사항은?", "관리체계 기반 마련의 경영진 참여 상세내용은?"

3. 'vector' - ISMS-P 인증기준 안내서의 설명, 적용 방법, 요구사항, 사례 또는 심사 관점 등 문서 내용 검색이 필요한 질문
   예: "비밀번호 생성 규칙과 변경 주기 가이드라인은?", "개인정보 처리방침 공개 시 필수 항목은?", "클라우드 서비스 이용 시 보안 점검사항은?"

판단 규칙:
- 수치 계산과 통계가 핵심이면 'database'를 우선하세요.
- 기준의 의미나 이행 방법에 대한 설명이 핵심이면 'vector'를 선택하세요.
- ISMS-P의 정의·목적을 간단히 소개해 달라는 요청은 'general'로 분류하세요.
- CSV의 컬럼명이나 특정 항목의 상세내용·주요 확인사항을 묻는다면 'database'로 분류하세요.
- 특정 기준의 가이드라인, 필수 요구사항, 점검사항, 규칙·주기, 이행 방법·사례·심사 관점을 묻는다면 'vector'로 분류하세요.

반드시 'general', 'database', 'vector' 중 하나만 답변하세요.
다른 설명 없이 분류 결과만 반환하세요.
"""

    if _is_database_question(question):
        intent = "database"
    elif _is_vector_question(question):
        intent = "vector"
    else:
        # 명시적인 CSV 조회가 아닐 때만 LLM으로 의도를 분류
        conversation = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(conversation)
        intent = response.content.strip().lower()

    # 유효한 의도인지 확인
    if intent not in ['general', 'database', 'vector']:
        intent = 'general'

    return {
        "intent": intent,
        "question": question,
        # 새 질문에 이전 검색/검증 상태가 섞이지 않도록 요청 단위 상태 초기화
        "vector_results": None,
        "rewritten_query": None,
        "vector_error": None,
        "sql_query": None,
        "db_results": None,
        "retry_count": 0,
        "error": None,
        "answer_valid": None,
        "answer_feedback": None,
        "answer_retry_count": 0,
        "draft_answer": None,
    }


def general_answer(state: AgentState) -> AgentState:
    """
    일반적인 질문에 직접 답변하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])

    system_prompt = """
당신은 ISMS-P 인증기준 안내서와 관련 통계 데이터를 안내하는 AI 어시스턴트입니다.
인사나 사용법 질문에는 자연스럽고 간결하게 답변하세요.
이 에이전트가 ISMS-P 인증기준, 세부 점검항목, 주요 확인사항 및 결함 통계를 설명할 수 있음을 안내할 수 있습니다.
ISMS-P 소개 요청에는 정보보호 및 개인정보보호 관리체계를 통합적으로 수립·운영하고 인증받는 제도라는 수준에서 목적과 역할을 간단히 설명하세요.
자료 확인이 필요한 구체적인 ISMS-P 사실이나 수치를 근거 없이 만들어 답하지 마세요.
"""

    # 시스템 메시지 + 기존 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    answer = response.content

    return {
        "messages": [AIMessage(content=answer)]
    }


def vector_search(state: AgentState) -> AgentState:
    """
    Qdrant 벡터 검색을 수행하는 노드

    1. LLM으로 질문 분석 (최적화된 쿼리 + 카테고리 추출)
    2. 병렬 벡터 검색 수행

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])

    # 재작성된 쿼리가 있으면 사용, 없으면 원본 질문 사용
    original_query = state.get("rewritten_query") or state.get("question", "")

    # 이전 대화 맥락이 있으면 완전한 질문으로 재구성
    if len(messages) > 1 and not state.get("rewritten_query"):
        # rewritten_query가 없을 때만 (첫 시도) 맥락 고려
        system_prompt_complete = """
당신은 ISMS-P 질의 검색을 위한 질문 분석 전문가입니다.
이전 대화 맥락을 고려하여 현재 질문을 완전하고 명확한 질문으로 재구성하세요.

예시:
- 이전: "위험 식별 및 평가 기준을 알려줘" → 현재: "구체적으로 어떻게 해?" → 재구성: "ISMS-P 위험 식별 및 평가는 구체적으로 어떻게 수행해야 하나요?"
- 이전: "개인정보 암호화 기준은?" → 현재: "예외도 있어?" → 재구성: "ISMS-P 개인정보 암호화 기준에 예외가 있나요?"

완전한 질문만 반환하세요. 설명은 포함하지 마세요.
만약 현재 질문이 이미 완전하다면 그대로 반환하세요.
"""
        conversation_complete = [SystemMessage(content=system_prompt_complete)] + messages
        response_complete = llm.invoke(conversation_complete)
        original_query = response_complete.content.strip()

    # 1. LLM으로 쿼리 분석 및 카테고리 추출 (Structured Output)
    # 시스템 프롬프트: 역할 정의 및 카테고리 설명
    system_prompt = """당신은 ISMS-P 인증기준 안내서의 벡터 검색 쿼리 최적화 전문가입니다.
사용자의 질문을 안내서에서 찾기 쉬운 검색어로 바꾸세요.

검색 쿼리 작성 규칙:
1. 질문의 핵심인 인증기준명, 통제항목, 보호조치, 개인정보 처리 단계 및 심사 용어를 보존하세요.
2. 관련된 공식 용어 또는 동의어를 필요한 만큼만 추가하세요.
3. 질문에 없는 항목번호, 법령명, 수치 또는 사실을 추측해 넣지 마세요.
4. 문장형 질문보다 핵심 요구사항이 드러나는 간결한 검색 문구를 작성하세요.
5. 현재 문서 인덱스에는 신뢰할 수 있는 고정 카테고리가 없으므로 categories는 항상 null로 반환하세요."""

    # 유저 프롬프트: 실제 질문
    user_prompt = f"다음 질문을 분석해주세요:\n\n{original_query}"

    # 메시지 객체 생성 (Structured Output용)
    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # Structured Output으로 LLM 호출
    structured_llm = llm.with_structured_output(VectorSearchQuery)
    query_analysis = structured_llm.invoke(llm_messages)

    optimized_query = query_analysis.optimized_query
    categories = query_analysis.categories

    print(f"[벡터 검색 쿼리 분석]")
    print(f"  원본 쿼리: {original_query}")
    print(f"  최적화된 쿼리: {optimized_query}")
    print(f"  선택된 카테고리: {categories}")

    # 2. 병렬 벡터 검색 수행 (카테고리 필터 적용)
    try:
        retriever = get_cached_retriever()
        results = retriever.search(
            optimized_query,
            k=3,
            score_threshold=0.5,
            categories=categories
        )
    except Exception as exc:
        # URL, 토큰, 응답 본문 등 민감할 수 있는 상세정보는 상태나 로그에 남기지 않음
        error_type = type(exc).__name__
        print(f"[벡터 검색 오류] {error_type}")
        return {
            "vector_results": [],
            "vector_error": error_type
        }

    return {
        "vector_results": results,
        "vector_error": None
    }


def rewrite_query(state: AgentState) -> AgentState:
    """
    검색 결과가 부족할 때 쿼리를 재작성하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])

    system_prompt = """
당신은 ISMS-P 인증기준 안내서 검색 쿼리 개선 전문가입니다.

사용자의 질문이 검색 결과를 얻지 못했습니다.
이전 대화 맥락을 고려하여 질문을 다시 작성하여 더 나은 검색 결과를 얻을 수 있도록 하세요.

최적화 방법:
- 이전 대화에서 언급된 인증기준 또는 보호조치 맥락을 포함
- ISMS-P 안내서에서 사용하는 공식 용어와 관련 동의어를 추가
- 지나치게 구체적인 표현은 상위 통제영역 수준으로 넓혀 검색
- 원래 질문의 의도를 바꾸거나 항목번호·수치·법령을 추측하지 않기

재작성된 쿼리만 반환하세요. 설명은 포함하지 마세요.
"""

    # 시스템 메시지 + 전체 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    rewritten = response.content.strip()

    return {
        "rewritten_query": rewritten,
        "retry_count": state.get("retry_count", 0) + 1
    }


def database_query(state: AgentState) -> AgentState:
    """
    Text2SQL을 수행하여 데이터베이스를 조회하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])
    question = state.get("question", "")
    previous_error = state.get("error")

    # 이전 대화 맥락이 있으면 완전한 질문으로 재구성
    if len(messages) > 1:
        system_prompt = """
당신은 ISMS-P 데이터 조회를 위한 질문 분석 전문가입니다.
이전 대화 맥락을 고려하여 현재 질문을 완전하고 명확한 질문으로 재구성하세요.

예시:
- 이전: "접근권한 검토 결함수는?" → 현재: "다른 항목보다 많아?" → 재구성: "접근권한 검토의 결함수가 다른 통제항목보다 많은가?"
- 이전: "1.1.1 경영진의 참여" → 현재: "확인사항은?" → 재구성: "ISMS-P 1.1.1 경영진의 참여 항목의 주요 확인사항은 무엇인가?"

완전한 질문만 반환하세요. 설명은 포함하지 마세요.
만약 현재 질문이 이미 완전하다면 그대로 반환하세요.
"""
        conversation = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(conversation)
        complete_question = response.content.strip()
    else:
        complete_question = question

    # Text2SQL 실행
    text2sql_engine = get_cached_text2sql_engine()
    result = text2sql_engine.query(complete_question, previous_error=previous_error)

    return {
        "sql_query": result["sql_query"],
        "db_results": result["result"],
        "error": result["error"],
        "retry_count": state.get("retry_count", 0) + 1
    }


def _build_evidence_context(state: AgentState) -> str:
    """벡터 검색 및 DB 조회 결과를 답변 근거 형식으로 구성합니다."""
    context_parts = []

    if state.get("vector_results"):
        context_parts.append("[ISMS-P 안내서 검색 결과]")
        for i, doc in enumerate(state["vector_results"], 1):
            source = doc.metadata.get("source", "알 수 없음")
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"\n[문서 {i}] 출처: {source}, 페이지: {page}\n{doc.page_content}"
            )

    if state.get("db_results"):
        context_parts.append(
            f"\n[ISMS-P CSV 데이터베이스 조회 결과]\n{state['db_results']}"
        )
        if state.get("sql_query"):
            context_parts.append(f"\n[조회에 사용된 SQL]\n{state['sql_query']}")

    return "\n".join(context_parts)


def generate_answer(state: AgentState) -> AgentState:
    """
    검색 결과를 바탕으로 최종 답변을 생성하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])

    context = _build_evidence_context(state)
    answer_feedback = state.get("answer_feedback")
    retry_instruction = ""
    if answer_feedback:
        retry_instruction = f"""

이전 답변은 검증을 통과하지 못했습니다. 다음 피드백을 반드시 반영해 답변을 다시 작성하세요.
<validation_feedback>
{answer_feedback}
</validation_feedback>
"""

    system_prompt = f"""
당신은 ISMS-P 인증기준 및 결함 통계 질의응답 전문가입니다.

아래 검색 근거만 사용하여 사용자의 질문에 정확하게 답변하세요.

<evidence>
{context}
</evidence>
{retry_instruction}

답변 시 다음 규칙을 따르세요:
- 검색 근거에 명시된 기준, 항목명, 항목번호, 수치 및 주요 확인사항을 왜곡하지 마세요.
- 근거에 없는 법령, 의무, 예외, 제재, 수치 또는 사례를 추측하지 마세요.
- 문서 검색 결과를 사용했다면 답변 말미에 확인 가능한 출처와 페이지를 표시하세요.
- DB 조회 결과를 사용했다면 결과의 숫자와 분류를 정확히 전달하되 SQL 자체는 답변에 포함하지 마세요.
- 최소·최대 결함수에 동률 항목이 여러 개면 하나를 임의로 고르지 말고 동률 개수와 항목들을 설명하세요.
- 대표항목이 최대 10개로 제공되면 전체 목록이라고 표현하지 말고 예시임을 밝히세요.
- 0건 포함 여부에 따라 결과가 달라지는 근거라면 두 범위를 명확히 구분하세요.
- 질문에 필요한 근거가 부족하면 부족한 부분을 분명히 밝히고 확인되지 않은 내용을 만들지 마세요.
- 이전 대화 맥락을 고려하되 현재 제공된 근거보다 우선하지 마세요.
- 한국어로 명확하고 간결하게 답변하세요.
"""

    # 시스템 메시지 + 기존 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages
    if answer_feedback:
        conversation.append(HumanMessage(
            content="검증 피드백을 반영한 수정 답변만 다시 작성하세요."
        ))

    response = llm.invoke(conversation)
    answer = response.content

    answer_retry_count = state.get("answer_retry_count", 0)
    if answer_feedback:
        answer_retry_count += 1

    return {
        "draft_answer": answer,
        "answer_retry_count": answer_retry_count
    }


def final_query(state: AgentState) -> AgentState:
    """생성된 답변이 검색 근거에 충실한지 독립적으로 검증합니다."""
    generated_answer = state.get("draft_answer", "")

    if not generated_answer:
        return {
            "answer_valid": False,
            "answer_feedback": "생성된 답변이 없습니다. 질문과 검색 근거를 바탕으로 답변을 작성하세요."
        }

    context = _build_evidence_context(state)
    question = state.get("question", "")
    system_prompt = """
당신은 ISMS-P RAG 답변의 최종 검증자입니다.
사용자 질문, 검색 근거, 생성된 답변을 비교하여 답변의 근거 충실성을 판정하세요.

검증 기준:
1. 질문의 핵심 요구에 직접 답했는가
2. 기준명, 항목번호, 수치, 관계 및 주요 확인사항이 검색 근거와 일치하는가
3. 검색 근거에 없는 법적 의무, 예외, 제재, 수치 또는 단정적인 조언을 추가하지 않았는가
4. 문서 기반 답변의 출처와 페이지가 실제 제공된 근거와 일치하는가
5. 근거가 부족한 경우 그 한계를 솔직하게 밝혔는가
6. 최소·최대·순위 질문에서 전체 결과와 동률을 확인하고 임의의 한 항목만 선택하지 않았는가

표현 방식이나 문장 길이의 사소한 차이만으로 부적합 판정하지 마세요.
부적합한 경우 feedback에 틀린 부분과 수정 방법을 구체적으로 작성하세요.
"""
    user_prompt = f"""
<question>
{question}
</question>

<evidence>
{context}
</evidence>

<generated_answer>
{generated_answer}
</generated_answer>
"""

    structured_llm = llm.with_structured_output(AnswerValidation)
    validation = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    result = {
        "answer_valid": validation.is_valid,
        "answer_feedback": validation.feedback
    }

    # 첫 검증 통과 또는 1회 재생성 완료 후에만 사용자 메시지에 최종 답변 추가
    if validation.is_valid or state.get("answer_retry_count", 0) >= 1:
        result["messages"] = [AIMessage(content=generated_answer)]

    return result


def route_by_intent(state: AgentState) -> str:
    """
    의도에 따라 다음 노드를 결정하는 라우팅 함수

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름
    """
    intent = state.get("intent", "general")

    if intent == "general":
        return "general_answer"
    elif intent == "database":
        return "database_query"
    elif intent == "vector":
        return "vector_search"
    else:
        return "general_answer"


def check_vector_results(state: AgentState) -> str:
    """
    벡터 검색 결과를 확인하고 다음 노드를 결정하는 함수

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름
    """
    results = state.get("vector_results", [])
    vector_error = state.get("vector_error")
    retry_count = state.get("retry_count", 0)

    # 연결/컬렉션 오류는 쿼리를 바꿔도 해결되지 않으므로 빈 근거로 답변 생성
    if vector_error:
        return "generate_answer"

    # 결과가 있거나 재시도 횟수가 2회 이상이면 답변 생성
    if results or retry_count >= 2:
        return "generate_answer"
    else:
        return "rewrite_query"


def check_db_results(state: AgentState) -> str:
    """
    데이터베이스 검색 결과를 확인하고 다음 노드를 결정하는 함수

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름
    """
    error = state.get("error")
    result = state.get("db_results")
    retry_count = state.get("retry_count", 0)

    # 오류가 없고 결과가 있으면 답변 생성
    text2sql_engine = get_cached_text2sql_engine()
    if not error and result and not text2sql_engine.is_empty_result(result):
        return "generate_answer"

    # 재시도 횟수가 2회 이상이면 답변 생성 (오류 메시지 포함)
    if retry_count >= 2:
        return "generate_answer"

    # 재시도
    return "database_query"


def check_final_results(state: AgentState) -> str:
    """검증 실패 시 답변을 한 번만 재생성한 후 종료합니다."""
    if state.get("answer_valid", False):
        return "end"

    if state.get("answer_retry_count", 0) >= 1:
        return "end"

    return "generate_answer"
