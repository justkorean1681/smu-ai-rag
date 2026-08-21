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


class VectorSearchQuery(BaseModel):
    """벡터 검색을 위한 쿼리 분석 결과"""
    optimized_query: str = Field(
        description="검색에 최적화된 쿼리. 핵심 키워드를 포함하고 명확하게 작성."
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="선택된 카테고리 리스트 (1-2개). 명확하게 관련 있는 카테고리만 선택. 애매하거나 불확실한 경우 null 반환. 가능한 값: 관리체계_수립_및_운영, 보호대책_요구사항, 개인정보_처리_단계별_요구사항"
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

    # 마지막 사용자 메시지를 질문으로 사용
    question = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

    system_prompt = """
당신은 ISMS-P 인증 체계 질의 의도 분류 전문가입니다.

    이전 대화 맥락을 고려하여 현재 질문을 다음 3가지 중 하나로 분류하세요:

    1. 'general': 일상 대화나 인사.
       예: "안녕하세요", "고마워", "날씨 어때?"

    2. 'database': ISMS-P 결함 현황, 통계, 수치 데이터를 조회해야 하는 경우. (키워드: 결함, 통계, 순위, 개수, 가장 많은, 연도별, top)
       예: "가장 결함이 많은 통제영역은?", "가장 결함이 많은 통제영역 알려줘", "결함이 몇 개야?"

    3. 'vector': ISMS-P 인증 기준 안내서의 상세 내용, 가이드, 지침, 해설 등이 필요한 경우. (키워드: 인증 기준, 항목, 요건, 규칙)
       예: "경영진 참여 요건이 뭐야?", "비밀번호 작성 규칙 알려줘.", "인증 기준 1.1.1 항목이 뭐야?"

    반드시 'general', 'database', 'vector' 중 하나만 답변하세요.
    다른 설명 없이 분류 결과만 반환하세요.
"""

    # 시스템 메시지 + 전체 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    intent = response.content.strip().lower()

    # 유효한 의도인지 확인
    if intent not in ['general', 'database', 'vector']:
        intent = 'general'

    return {
        "intent": intent,
        "question": question
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
당신은 친절한 AI 어시스턴트입니다.
사용자의 질문에 자연스럽고 도움이 되는 답변을 제공하세요.
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
당신은 ISMS-P 인증 질의 분석 전문가입니다.
사용자의 이전 질문 맥락과 현재 질문을 결합하여, 인증 기준 안내서를 검색하기에 가장 완벽하고 명확한 질문으로 재구성하세요.

예시:
- 이전: "1.1.2 항목의 지정 요건이 뭐야?" → 현재: "예외는 없어?" → 재구성: "1.1.2 최고책임자 지정 요건의 예외사항은 무엇인가?"
- 이전: "접근통제 관련 결함이 많아" → 현재: "어떻게 해결해?" → 재구성: "접근통제 영역의 결함을 해결하기 위한 보완 통제 대책은 무엇인가?"
- 이전: "ISMS-P 인증 기준" → 현재: "비밀번호 규칙은?" → 재구성: "ISMS-P 인증 기준 내 비밀번호 작성 규칙은 무엇인가?"

완전한 질문만 반환하세요. 답변 설명은 포함하지 마세요.
질문이 이미 완전하다면 그대로 반환하세요.
"""
        conversation_complete = [SystemMessage(content=system_prompt_complete)] + messages
        response_complete = llm.invoke(conversation_complete)
        original_query = response_complete.content.strip()

    # 1. LLM으로 쿼리 분석 및 카테고리 추출 (Structured Output)
    # 시스템 프롬프트: 역할 정의 및 카테고리 설명
    system_prompt = """당신은 ISMS-P 인증 기준 검색 쿼리 전문가입니다.
사용자의 질문을 분석하여 ISMS-P 안내서 검색에 최적화된 쿼리를 생성하고, 관련 인증 영역(카테고리)을 선택하세요.

사용 가능한 카테고리 (인증 영역):
- 관리체계_수립_및_운영: 1.1~1.4 관리체계 기반, 위험관리, 운영, 점검 및 개선 관련
- 보호대책_요구사항: 2.1~2.12 정책/조직/자산, 인적보안, 물리보안, 접근통제, 시스템 보안 등
- 개인정보_처리_단계별_요구사항: 3.1~3.5 수집, 보유/이용, 제공, 파기, 권한보호 관련

카테고리 선택 규칙:
1. 질문이 ISMS-P 인증 항목의 어느 영역에 해당하는지 파악하여 최대 2개 선택.
2. 애매하면 null 반환.
3. 인증 기준 항목 번호(예: 1.1.2, 2.6.1)가 포함된 질문이면 해당 카테고리를 반드시 포함.

출력 지침:
1. optimized_query: ISMS-P 인증 심사 관점의 핵심 키워드를 포함한 쿼리 (예: '1.1.2 최고책임자 지정 요건')
2. categories: 위 사용 가능한 카테고리 중 선택."""

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
    retriever = get_cached_retriever()
    results = retriever.search(optimized_query, k=3, score_threshold=0.5, categories=categories)

    return {
        "vector_results": results
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
당신은 검색 쿼리 최적화 전문가입니다.

사용자의 질문이 검색 결과를 얻지 못했습니다.
이전 대화 맥락을 고려하여 질문을 다시 작성하여 더 나은 검색 결과를 얻을 수 있도록 하세요.

최적화 방법:
- 이전 대화에서 언급된 맥락을 포함
- 동의어나 관련 용어 추가
- 질문을 더 구체적이거나 더 일반적으로 변경
- 핵심 키워드 강조

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
당신은 ISMS-P 인증 정보 질의 분석 전문가입니다.
이전 대화 맥락과 현재 질문을 결합하여, 데이터베이스 조회에 최적화된 완전한 질문으로 재구성하세요.

예시:
- 이전: "접근통제 영역" → 현재: "결함이 몇 개야?" → 재구성: "접근통제 영역의 총 결함 수는 몇 개인가?"
- 이전: "1.1.2 항목" → 현재: "이름이랑 상세내용 뭐야?" → 재구성: "1.1.2 항목의 항목명과 상세내용을 알려줘."
- 이전: "가장 결함 많은 곳" → 현재: "거기 통제항목들 다 보여줘" → 재구성: "가장 결함이 많은 통제영역의 통제항목들을 모두 보여줘."

완전한 질문만 반환하세요. 설명은 포함하지 마세요.
질문이 이미 완전하다면 그대로 반환하세요.
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

    # 컨텍스트 구성
    context_parts = []

    # 벡터 검색 결과가 있으면 추가
    if state.get("vector_results"):
        docs = state["vector_results"]
        context_parts.append("관련 문서:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "알 수 없음")
            page = doc.metadata.get("page", "?")
            category = doc.metadata.get("category", "")

            # 출처 정보 구성
            source_info = f"출처: {source}, 페이지: {page}"
            if category:
                source_info += f", 카테고리: {category}"

            context_parts.append(f"\n[문서 {i}] {source_info}\n{doc.page_content}")

    # DB 검색 결과가 있으면 추가
    if state.get("db_results"):
        context_parts.append(f"\n\n데이터베이스 조회 결과:\n{state['db_results']}")
        if state.get("sql_query"):
            context_parts.append(f"\n실행된 SQL:\n{state['sql_query']}")

    context = "\n".join(context_parts)

    system_prompt = f"""
당신은 ISMS-P 인증 심사 및 관리체계 전문가입니다.

다음 정보를 바탕으로 ISMS-P 인증을 준비하는 사용자에게 정확하고 전문적인 답변을 제공하세요:

<context>
{context}
</context>

답변 규칙:
- 벡터 검색 결과(문서 내용)는 안내서의 상세 기준을 근거로 제시하세요.
- 데이터베이스 결과(통계)는 구체적인 숫자와 통계치를 제시하세요.
- 인증 기준을 설명할 때는 반드시 '항목 번호'를 언급하세요.
- ISMS-P 인증 심사 관점의 친절하고 전문적인 어조를 유지하세요.
"""

    # 시스템 메시지 + 기존 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    answer = response.content

    return {
        "messages": [AIMessage(content=answer)]
    }


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
    retry_count = state.get("retry_count", 0)

    # 결과가 있거나 재시도 횟수가 2회 이상이면 답변 생성
    retriever = get_cached_retriever()
    if retriever.is_relevant(results) or retry_count >= 2:
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
