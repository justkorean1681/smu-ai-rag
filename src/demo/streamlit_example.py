import os
import sys
from pathlib import Path
import ast
import re
import pandas as pd
import altair as alt


# src 디렉토리를 Python 경로에 추가
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

import streamlit as st
from dotenv import load_dotenv
from ai import create_graph

# 환경 변수 로드
load_dotenv()

graph = create_graph()

def display_visualization(db_results):
    if not db_results:
        return

    try:
        data_to_plot = db_results
        
        if isinstance(db_results, str):
            cleaned_str = re.sub(r"Decimal\(['\"](.*?)['\"]\)", r"\1", db_results)
            try:
                data_to_plot = ast.literal_eval(cleaned_str)
            except Exception:
                pass 

        if not isinstance(data_to_plot, list) or len(data_to_plot) == 0:
            return

        df = pd.DataFrame(data_to_plot)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        string_cols = df.select_dtypes(exclude=['number']).columns.tolist()

        if numeric_cols and string_cols:
            chart_df = pd.DataFrame({
                'Category': df[string_cols[0]].astype(str),
                'Value': df[numeric_cols[0]]
            })


            with st.expander("📊 데이터 시각화 리포트 보기", expanded=False):
                
                base = alt.Chart(chart_df).encode(
                    x=alt.X('Value:Q', 
                            axis=alt.Axis(labels=False, grid=False, domain=False, ticks=False, title='')),
                    y=alt.Y('Category:N', 
                            sort='-x', 
                            axis=alt.Axis(grid=False, domain=False, ticks=False, labelFontSize=14, labelFontWeight='bold', labelPadding=10, title='', labelLimit=300)),
                    tooltip=['Category', 'Value']
                ).properties(
                    height=max(250, len(chart_df) * 45) 
                )

                bars = base.mark_bar(cornerRadiusEnd=6).encode(
                    color=alt.Color('Value:Q', 
                                    scale=alt.Scale(scheme='greys'), 
                                    legend=None) 
                )

                text = base.mark_text(
                    align='left',
                    baseline='middle',
                    dx=10, 
                    fontSize=14,
                    fontWeight='bold',
                    color='#333333'
                ).encode(
                    text='Value:Q'
                )

                final_chart = (bars + text).configure_view(strokeWidth=0)

                st.altair_chart(final_chart, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ 차트 그리기 실패: {e}")

def init_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "안녕하세요! 🛡️ **ISMS-P 인사이트 AI 에이전트**입니다.\n\nISMS-P 인증 기준에 대한 **가이드라인 해석(문서 검색)**부터 **결함 통계 현황(DB 검색)**까지 무엇이든 물어보세요!"
            }
        ]
    
    if "selected_prompt" not in st.session_state:
        st.session_state.selected_prompt = None

def set_prompt(prompt_text):
    st.session_state.selected_prompt = prompt_text

def display_message(role: str, content: str, workflow_info: dict = None):
    """메시지 표시"""
    with st.chat_message(role):
        st.markdown(content)

        # 워크플로 정보가 있으면 표시 (assistant 메시지에만)
        if role == "assistant" and workflow_info:
            
            if workflow_info.get("db_results"):
                display_visualization(workflow_info["db_results"])
                
            display_workflow_info(workflow_info)


def display_workflow_info(result: dict):
    """워크플로 정보 표시"""
    with st.expander("🔍 워크플로 정보"):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("의도", result.get("intent", "N/A"))

            if result.get("retry_count"):
                st.metric("재시도 횟수", result["retry_count"])

        with col2:
            if result.get("vector_results"):
                st.metric("검색된 문서", len(result["vector_results"]))

            if result.get("db_results"):
                st.info("DB 검색 수행됨")

        # 벡터 검색 결과 상세 표시
        if result.get("vector_results"):
            st.markdown("#### 📄 검색된 문서")
            for i, doc in enumerate(result["vector_results"], 1):
                with st.expander(f"문서 {i}: {doc.metadata.get('source', '알 수 없음')}"):
                    # 메타데이터 표시
                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        st.caption(f"📖 페이지: {doc.metadata.get('page', 'N/A')}")
                    with meta_cols[1]:
                        if doc.metadata.get('category'):
                            st.caption(f"🏷️ 카테고리: {doc.metadata.get('category')}")
                    with meta_cols[2]:
                        if doc.metadata.get('score'):
                            st.caption(f"⭐ 점수: {doc.metadata.get('score', 0):.3f}")

                    # 문서 내용 표시
                    st.markdown("**내용:**")
                    st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

        # SQL 쿼리 표시
        if result.get("sql_query"):
            st.code(result["sql_query"], language="sql")

        # 재작성된 쿼리 표시
        if result.get("rewritten_query"):
            st.info(f"재작성된 쿼리: {result['rewritten_query']}")

        # 오류 표시
        if result.get("error"):
            st.error(f"오류: {result['error']}")


def main():
    """메인 함수"""
    st.set_page_config(
        page_title="ISMS-P AI 에이전트",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🛡️ ISMS-P AI 에이전트 워크플로")
    st.markdown("---")

    # 사이드바 - 환경 변수 확인
    with st.sidebar:
        st.header("⚙️ 설정 확인")

        required_vars = {
            "OPENAI_API_KEY": "OpenAI API",
            "QDRANT_URL": "Qdrant URL",
            "QDRANT_API_KEY": "Qdrant API Key",
            "SUPABASE_DB_URL": "Supabase DB"
        }

        for var, name in required_vars.items():
            if os.getenv(var):
                st.success(f"✓ {name}")
            else:
                st.error(f"✗ {name}")

        st.markdown("---")
        st.header("📖 사용 방법")
        with st.expander("🗣️ 일반 대화", expanded=True): # 첫 번째는 열어두기
            st.button("안녕하세요.", on_click=set_prompt, args=("안녕하세요.",), use_container_width=True)
            st.button("오늘 날씨 어때?", on_click=set_prompt, args=("오늘 날씨 어때?",), use_container_width=True)

        with st.expander("🔍 가이드라인 문서 검색", expanded=False):
            st.button("1.1.1 항목의 확인 사항은?", on_click=set_prompt, args=("ISMS-P 인증 기준 1.1.1 항목의 주요 확인 사항은?",), use_container_width=True)
            st.button("비밀번호 작성 가이드라인", on_click=set_prompt, args=("비밀번호 작성 규칙에 대한 가이드라인을 알려줘.",), use_container_width=True)
            st.button("경영진 참여 요건", on_click=set_prompt, args=("경영진 참여 요건이 뭐야?",), use_container_width=True)

        with st.expander("📊 결함 통계 DB 검색", expanded=False):
            st.button("결함이 가장 많은 통제영역 Top3", on_click=set_prompt, args=("가장 결함이 많이 발생한 통제영역 Top 3를 알려줘.",), use_container_width=True)
            st.button("결함 20건 이상인 통제영역은?", on_click=set_prompt, args=("결함 발생 건수가 20건 이상인 통제영역은 어디야?",), use_container_width=True)
            st.button("10.4 접근통제 영역의 결함 수는?", on_click=set_prompt, args=("10.4 접근통제 영역의 결함 건수는 몇 개야?",), use_container_width=True)

        if st.button("🔄 대화 내용 초기화", type="primary", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "안녕하세요! 🛡️ **ISMS-P 인사이트 AI 에이전트**입니다.\n\nISMS-P 인증 기준에 대한 **가이드라인 해석(문서 검색)**부터 **결함 통계 현황(DB 검색)**까지 무엇이든 물어보세요!"
                }
            ]
            st.rerun()

    # 세션 상태 초기화
    init_session_state()

    # 이전 메시지 표시
    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("workflow_info")  # 워크플로 정보가 있으면 전달
        )

    # 사용자 입력
    user_input = st.chat_input("결함 통계나 ISMS-P 가이드라인에 대해 질문해보세요...")
    
    prompt = None
    if user_input:
        prompt = user_input
    elif st.session_state.selected_prompt:
        prompt = st.session_state.selected_prompt
        st.session_state.selected_prompt = None # 한 번 실행 후 변수 비우기 (중복 실행 방지)

    # prompt가 존재하면 챗봇 답변 생성 시작
    if prompt:
        # 사용자 메시지 표시 및 저장
        display_message("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        # 워크플로 실행
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                try:
                    # 그래프 실행
                    result = graph.invoke({
                        "messages": [{"role": "user", "content": prompt}]
                    })

                    # 답변 표시 (messages의 마지막 AIMessage에서 추출)
                    messages = result.get("messages", [])
                    if messages:
                        # 마지막 메시지에서 content 추출
                        last_message = messages[-1]
                        answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
                    else:
                        answer = "죄송합니다. 답변을 생성할 수 없습니다."

                    st.markdown(answer)

                    # 워크플로 정보 표시
                    display_workflow_info(result)

                    if result.get("db_results"):
                        display_visualization(result["db_results"])

                    # 어시스턴트 메시지와 워크플로 정보 함께 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "workflow_info": result  # 워크플로 정보 저장
                    })

                except Exception as e:
                    error_msg = f"오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()
