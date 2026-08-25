import os
import sys
import html
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

def _generate_brand_gradient(n: int):
    """브랜드 블루 톤(진한 → 연한)으로 n개의 그라데이션 컬러 생성"""
    start = (30, 78, 140)     # 진한 네이비 (#1E4E8C)
    end = (191, 219, 254)     # 연한 스카이블루 (#BFDBFE)
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        r = int(start[0] + (end[0] - start[0]) * t)
        g = int(start[1] + (end[1] - start[1]) * t)
        b = int(start[2] + (end[2] - start[2]) * t)
        colors.append(f"#{r:02X}{g:02X}{b:02X}")
    return colors


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
            }).sort_values('Value', ascending=False).reset_index(drop=True)

            # 값이 큰 항목일수록 진한 브랜드 블루가 되도록 그라데이션 매핑
            categories_sorted = chart_df['Category'].tolist()
            gradient_colors = _generate_brand_gradient(len(categories_sorted))
            color_scale = alt.Scale(domain=categories_sorted, range=gradient_colors)

            max_val = float(chart_df['Value'].max())
            track_max = max_val * 1.4 if max_val > 0 else 1
            chart_df['Track'] = track_max

            with st.expander("📊 데이터 시각화 리포트", expanded=False):

                bar_height = 24
                row_height = max(260, len(chart_df) * 68)

                base = alt.Chart(chart_df).encode(
                    y=alt.Y('Category:N',
                            sort='-x',
                            scale=alt.Scale(paddingInner=0.5, paddingOuter=0.3),
                            axis=alt.Axis(grid=False, domain=False, ticks=False,
                                           labelFontSize=14, labelFontWeight=600,
                                           labelPadding=14, title='', labelLimit=320,
                                           labelColor='#334155')),
                ).properties(height=row_height)

                # 배경 트랙 바 (연한 회색빛 파랑) — 진행바 느낌을 위한 바탕
                track = base.mark_bar(
                    cornerRadius=10, height=bar_height, color='#EEF2FA'
                ).encode(
                    x=alt.X('Track:Q',
                            scale=alt.Scale(domain=[0, track_max]),
                            axis=alt.Axis(labels=False, grid=False, domain=False, ticks=False, title=''))
                )

                # 실제 데이터 바 — 브랜드 그라데이션 + 은은한 그림자
                bars = base.mark_bar(
                    cornerRadius=10, height=bar_height,
                    stroke='#FFFFFF', strokeWidth=0
                ).encode(
                    x=alt.X('Value:Q',
                            scale=alt.Scale(domain=[0, track_max]),
                            axis=alt.Axis(labels=False, grid=False, domain=False, ticks=False, title='')),
                    color=alt.Color('Category:N', scale=color_scale, legend=None),
                    tooltip=[alt.Tooltip('Category:N', title='항목'),
                             alt.Tooltip('Value:Q', title='건수')]
                )

                # 값 라벨 (막대 끝 바깥쪽, 굵은 네이비)
                text = base.mark_text(
                    align='left',
                    baseline='middle',
                    dx=10,
                    fontSize=13.5,
                    fontWeight=700,
                    color='#1E3A8A'
                ).encode(
                    x=alt.X('Value:Q', scale=alt.Scale(domain=[0, track_max])),
                    text=alt.Text('Value:Q', format=',.0f')
                )

                final_chart = (
                    (track + bars + text)
                    .properties(padding={'left': 8, 'right': 24, 'top': 16, 'bottom': 16})
                    .configure_view(strokeWidth=0)
                    .configure_axis(grid=False)
                )

                st.altair_chart(final_chart, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ 차트 시각화 실패: {e}")



def init_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "안녕하세요. **ISMS-P 인사이트 AI** 입니다.\n\n인증 기준 가이드라인(문서 검색) 및 결함 통계 현황(DB 검색)에 대해 질문을 남겨주시면 분석하여 답변해 드립니다."
            }
        ]
    
    if "selected_prompt" not in st.session_state:
        st.session_state.selected_prompt = None

def set_prompt(prompt_text):
    st.session_state.selected_prompt = prompt_text

def format_bubble_content(content: str) -> str:
    """말풍선 내부에 표시할 텍스트를 안전하게 HTML로 변환 (간단한 마크다운 지원)"""
    escaped = html.escape(content)
    # **bold**
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    # `code`
    escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
    # 줄바꿈
    escaped = escaped.replace("\n", "<br>")
    return escaped


def display_message(role: str, content: str, workflow_info: dict = None):
    """카카오톡 스타일 말풍선으로 메시지 표시"""
    bubble_html = format_bubble_content(content)

    if role == "user":
        st.markdown(
            f"""
            <div class="chat-row user">
                <div class="bubble user">{bubble_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="chat-row assistant">
                <div class="avatar">💠</div>
                <div class="bubble-col">
                    <div class="sender-name">ISMS-P 인사이트 AI</div>
                    <div class="bubble assistant">{bubble_html}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if workflow_info:
            _spacer, _content, _empty = st.columns([0.06, 0.66, 0.28])
            with _content:
                if workflow_info.get("db_results"):
                    display_visualization(workflow_info["db_results"])
                display_workflow_info(workflow_info)


def display_typing_bubble():
    """AI 응답 대기 중, 아바타가 포함된 '입력 중' 말풍선 표시"""
    st.markdown(
        """
        <div class="chat-row assistant">
            <div class="avatar">💠</div>
            <div class="bubble-col">
                <div class="sender-name">ISMS-P 인사이트 AI</div>
                <div class="bubble assistant typing">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_workflow_info(result: dict):
    """워크플로 정보 표시"""
    with st.expander("🔍 AI 워크플로 분석 과정", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("분석 의도", result.get("intent", "N/A"))
            if result.get("retry_count"):
                st.metric("재시도 횟수", result["retry_count"])

        with col2:
            if result.get("vector_results"):
                st.metric("참고 문서 건수", len(result["vector_results"]))
            if result.get("db_results"):
                st.success("통계 DB 검색 완료")

        if result.get("vector_results"):
            st.markdown("#### 참고 가이드라인 문서")
            for i, doc in enumerate(result["vector_results"], 1):
                with st.expander(f"📄 문서 {i}: {doc.metadata.get('source', '알 수 없음')} (Page: {doc.metadata.get('page', 'N/A')})"):
                    st.caption(f"카테고리: {doc.metadata.get('category', 'N/A')} | 연관도: {doc.metadata.get('score', 0):.3f}")
                    st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

        if result.get("sql_query"):
            st.markdown("#### 생성된 SQL 쿼리")
            st.code(result["sql_query"], language="sql")

        if result.get("rewritten_query"):
            st.markdown("#### AI 질문 재구성")
            st.info(f"{result['rewritten_query']}")

        if result.get("error"):
            st.error(f"오류 발생: {result['error']}")


def main():
    """메인 함수"""
    st.set_page_config(
        page_title="ISMS-P 인사이트 AI",
        page_icon="💠", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 🎨 커스텀 CSS: 
    st.markdown("""
        <style>
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

        /* 안전한 폰트 적용 */
        h1, h2, h3, h4, h5, h6, p, label, div[data-testid="stMarkdownContainer"] {
            font-family: 'Pretendard', -apple-system, sans-serif !important;
        }
        
        /* 앱 전체 배경 (카카오톡풍 연한 블루톤) */
        .stApp {
            background-color: #E9EEF6 !important;
        }
        
        /* 사이드바 배경 */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        /* 헤더 타이틀 */
        h1 {
            color: #2C6FBB !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            font-size: 2.2rem !important;
            margin-bottom: 2rem !important;
        }
        h2, h3 { 
            color: #3A7BC8 !important; 
            font-weight: 700 !important; 
            letter-spacing: -0.02em !important;
        }

        /* 기본 버튼 */
        .stButton > button {
            background-color: #FFFFFF !important;
            color: #1D4ED8 !important;
            border: 1px solid #BFDBFE !important;
            border-radius: 12px !important;
            font-weight: 600;
            padding: 0.5rem 1rem !important;
            transition: all 0.2s ease-in-out !important;
        }
        .stButton > button:hover {
            border-color: #93C5FD !important;
            background-color: #EFF6FF !important;
            transform: translateY(-1px);
        }

        /* Primary 버튼 */
        div[data-testid="stButton"] > button[kind="primary"] {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border: none !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background-color: #1D4ED8 !important;
        }

        /* =========================================
           ✅ 채팅 입력창 테두리 포커스 시에만 파란색
           ========================================= */
        
        /* 1. 기본 상태 (회색 테두리) */
        .stChatInputContainer,
        div[data-testid="stChatInput"] > div {
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            background-color: #FFFFFF !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02) !important;
        }

        /* 2. 포커스 상태 (파란색 테두리) */
        .stChatInputContainer:focus-within,
        div[data-testid="stChatInput"] > div:focus-within {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 1px #3B82F6 !important;
        }

        /* 3. 내부 기본 테두리 무력화 */
        .stChatInputContainer div[data-baseweb="textarea"],
        .stChatInputContainer div[data-baseweb="textarea"]:focus-within {
            border: none !important;
            box-shadow: none !important;
            background-color: transparent !important;
        }
        .stChatInputContainer textarea {
            outline: none !important;
            box-shadow: none !important;
        }

        /* 보내기 버튼 색상 */
        button[data-testid="stChatInputSubmitButton"]:not(:disabled) {
            background-color: #3B82F6 !important;
            color: white !important;
        }
        button[data-testid="stChatInputSubmitButton"]:not(:disabled):hover {
            background-color: #2563EB !important;
        }
        button[data-testid="stChatInputSubmitButton"]:not(:disabled) svg {
            fill: white !important;
            color: white !important;
        }

        /* =========================================
           ✅ 카카오톡 스타일 채팅 말풍선
           ========================================= */
        .chat-row {
            display: flex;
            align-items: flex-end;
            margin-bottom: 14px;
            width: 100%;
        }

        /* 사용자 메시지: 오른쪽 정렬 */
        .chat-row.user {
            justify-content: flex-end;
        }

        /* AI 메시지: 왼쪽 정렬 */
        .chat-row.assistant {
            justify-content: flex-start;
            align-items: flex-start;
        }

        .avatar {
            width: 38px;
            height: 38px;
            min-width: 38px;
            border-radius: 50%;
            background-color: #FFFFFF;
            border: 1px solid #DCE3ED;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            margin-right: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }

        .bubble-col {
            display: flex;
            flex-direction: column;
            max-width: 72%;
        }

        .sender-name {
            font-size: 12px;
            color: #64748B;
            margin: 0 0 4px 4px;
            font-weight: 600;
        }

        .bubble {
            padding: 10px 14px;
            font-size: 15px;
            line-height: 1.55;
            word-break: break-word;
            box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        }

        /* 사용자 말풍선: 파란색, 오른쪽 꼬리 */
        .bubble.user {
            max-width: 72%;
            background-color: #4A90E2;
            color: #FFFFFF;
            border-radius: 18px 18px 4px 18px;
        }

        /* AI 말풍선: 흰색, 왼쪽 꼬리 */
        .bubble.assistant {
            background-color: #FFFFFF;
            color: #1E293B;
            border: 1px solid #E5E9F0;
            border-radius: 18px 18px 18px 4px;
        }

        .bubble code {
            background-color: rgba(0,0,0,0.06);
            padding: 1px 5px;
            border-radius: 4px;
            font-size: 13px;
        }

        /* AI 응답 대기 중 "입력 중" 말풍선 */
        .bubble.assistant.typing {
            display: flex;
            align-items: center;
            gap: 5px;
            padding: 14px 18px;
            width: fit-content;
            align-self: flex-start;
        }
        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #94A3B8;
            animation: typing-bounce 1.4s infinite ease-in-out both;
        }
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes typing-bounce {
            0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
            40% { transform: scale(1); opacity: 1; }
        }
        
        /* Expander (접기/펴기) 메뉴 */
        [data-testid="stExpander"] {
            border: 1px solid #E0E7FF;
            border-radius: 12px;
            background-color: #F8FAFC !important;
        }
        
        /* 통계 메트릭 라벨 색상 */
        [data-testid="stMetricValue"] {
            color: #2563EB !important;
            font-weight: 800 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("ISMS-P 인사이트 AI")

    # 사이드바
    with st.sidebar:
        st.header("시스템 상태")

        required_vars = {
            "OPENAI_API_KEY": "OpenAI API",
            "QDRANT_URL": "Qdrant URL",
            "QDRANT_API_KEY": "Qdrant API Key",
            "SUPABASE_DB_URL": "Supabase DB"
        }
        
        # 사이드바 상태 체크 색상을 seagreen으로
        for var, name in required_vars.items():
            if os.getenv(var):
                st.markdown(f"<span style='color: seagreen; font-weight: bold;'>✓</span> {name}", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: #EF4444; font-weight: bold;'>-</span> {name} (오류)", unsafe_allow_html=True)

        st.markdown("<hr style='border:1px solid #E2E8F0; margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.header("추천 프롬프트")
        
        with st.expander("일반 대화", expanded=True): 
            st.button("안녕하세요.", on_click=set_prompt, args=("안녕하세요.",), use_container_width=True)
            st.button("오늘 날씨 어때?", on_click=set_prompt, args=("오늘 날씨 어때?",), use_container_width=True)

        with st.expander("가이드라인 검색", expanded=False):
            st.button("1.1.1 항목 확인 사항", on_click=set_prompt, args=("ISMS-P 인증 기준 1.1.1 항목의 주요 확인 사항은?",), use_container_width=True)
            st.button("비밀번호 작성 가이드", on_click=set_prompt, args=("비밀번호 작성 규칙에 대한 가이드라인을 알려줘.",), use_container_width=True)
            st.button("경영진 참여 요건", on_click=set_prompt, args=("경영진 참여 요건이 뭐야?",), use_container_width=True)

        with st.expander("결함 통계 검색", expanded=False):
            st.button("결함 Top 3 통제영역", on_click=set_prompt, args=("가장 결함이 많이 발생한 통제영역 Top 3를 알려줘.",), use_container_width=True)
            st.button("결함 20건 이상 영역", on_click=set_prompt, args=("결함 발생 건수가 20건 이상인 통제영역은 어디야?",), use_container_width=True)
            st.button("10.4 접근통제 결함 수", on_click=set_prompt, args=("10.4 접근통제 영역의 결함 건수는 몇 개야?",), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("새로운 대화 시작", type="primary", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "안녕하세요. **ISMS-P 인사이트 AI** 입니다.\n\n인증 기준 가이드라인(문서 검색) 및 결함 통계 현황(DB 검색)에 대해 질문을 남겨주시면 분석하여 답변해 드립니다."
                }
            ]
            st.rerun()

    # 세션 상태 초기화
    init_session_state()

    # 사용자 입력
    user_input = st.chat_input("질문을 입력해주세요...")

    prompt = None
    if user_input:
        prompt = user_input
    elif st.session_state.selected_prompt:
        prompt = st.session_state.selected_prompt
        st.session_state.selected_prompt = None

    # 새 질문이 들어오면 대화 목록에 먼저 추가
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

    # 전체 대화 내역을 카카오톡 스타일 말풍선으로 표시
    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("workflow_info")
        )

    # 마지막 메시지가 사용자 질문이면 AI 응답 생성
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_prompt = st.session_state.messages[-1]["content"]

        typing_placeholder = st.empty()
        with typing_placeholder.container():
            display_typing_bubble()

        try:
            result = graph.invoke({
                "messages": [{"role": "user", "content": last_prompt}]
            })

            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
            else:
                answer = "죄송합니다. 답변을 생성할 수 없습니다."

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "workflow_info": result
            })

        except Exception as e:
            error_msg = f"오류가 발생했습니다: {str(e)}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

        typing_placeholder.empty()
        st.rerun()

if __name__ == "__main__":
    main()