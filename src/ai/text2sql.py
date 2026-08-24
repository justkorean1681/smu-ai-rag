import os
from pathlib import Path

import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


class Text2SQLEngine:
    def __init__(self):
        """Text2SQL 엔진 초기화"""
        # 기본값은 Supabase이며, local 설정 시 프로젝트 CSV를 사용
        self.db = self._create_database()

        # LLM 초기화
        self.llm = init_chat_model("gpt-5.4-mini")

        # 데이터베이스 스키마 정보 캐싱
        self.schema_info = self.db.get_table_info()
        self.sql_dialect = "SQLite" if self.db.dialect == "sqlite" else "PostgreSQL"

    def _create_database(self) -> SQLDatabase:
        """설정에 따라 로컬 CSV 또는 Supabase 데이터베이스를 구성합니다."""
        database_source = os.getenv("ISMS_DB_SOURCE", "supabase").strip().lower()

        if database_source == "supabase":
            database_url = os.getenv("SUPABASE_DB_URL")
            if not database_url:
                raise ValueError(
                    "ISMS_DB_SOURCE가 supabase이지만 SUPABASE_DB_URL이 설정되지 않았습니다."
                )
            return SQLDatabase.from_uri(
                database_url,
                max_string_length=2000,
            )

        if database_source != "local":
            raise ValueError("ISMS_DB_SOURCE는 local 또는 supabase만 사용할 수 있습니다.")

        project_root = Path(__file__).resolve().parents[2]
        datasets_dir = project_root / "datasets"
        dataset_paths = {
            "isms_items": datasets_dir / "isms_items.csv",
            "isms_defects": datasets_dir / "isms_defects.csv",
        }

        missing_files = [
            path.name for path in dataset_paths.values() if not path.is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"datasets 폴더에서 다음 파일을 찾을 수 없습니다: {', '.join(missing_files)}"
            )

        # StaticPool을 사용해 모든 조회가 동일한 인메모리 DB를 보도록 유지
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        for table_name, csv_path in dataset_paths.items():
            dataframe = pd.read_csv(
                csv_path,
                encoding="utf-8-sig",
                keep_default_na=False,
            )
            dataframe.to_sql(
                table_name,
                engine,
                if_exists="replace",
                index=False,
            )

        return SQLDatabase(
            engine,
            include_tables=list(dataset_paths),
            sample_rows_in_table_info=3,
            max_string_length=2000,
        )

    def _get_deterministic_sql(self, question: str) -> str | None:
        """최소·최대 결함 질문은 동률을 보존하는 고정 SQL로 처리합니다."""
        normalized_question = "".join(question.lower().split())
        if "결함" not in normalized_question:
            return None

        asks_for_minimum = any(
            term in normalized_question
            for term in ("가장적", "최소", "제일적")
        )
        asks_for_maximum = any(
            term in normalized_question
            for term in ("가장많", "최대", "제일많")
        )

        aggregate_function = "STRING_AGG" if self.db.dialect != "sqlite" else "GROUP_CONCAT"

        if asks_for_minimum:
            if self.db.dialect == "sqlite":
                return f"""
WITH scopes AS (
    SELECT '전체 항목(0건 포함)' AS "비교범위", MIN("결함수") AS "기준결함수"
    FROM isms_defects
    UNION ALL
    SELECT '결함 발생 항목(0건 제외)' AS "비교범위", MIN("결함수") AS "기준결함수"
    FROM isms_defects
    WHERE "결함수" > 0
),
matches AS (
    SELECT
        s."비교범위",
        s."기준결함수",
        d."통제분야",
        d."통제항목",
        ROW_NUMBER() OVER (
            PARTITION BY s."비교범위"
            ORDER BY d."통제분야", d."통제항목"
        ) AS "순번",
        COUNT(*) OVER (PARTITION BY s."비교범위") AS "동률항목수"
    FROM scopes s
    JOIN isms_defects d ON d."결함수" = s."기준결함수"
)
SELECT
    "비교범위",
    "기준결함수" AS "결함수",
    MAX("동률항목수") AS "동률항목수",
    {aggregate_function}(
        CASE WHEN "순번" <= 10 THEN "통제분야" || ' > ' || "통제항목" END,
        ' | '
    ) AS "대표항목(최대10개)"
FROM matches
GROUP BY "비교범위", "기준결함수"
ORDER BY "기준결함수";
""".strip()

            return f"""
WITH scopes AS (
    SELECT '전체 항목(0건 포함)' AS "비교범위", MIN("결함수") AS "기준결함수"
    FROM isms_defects
    UNION ALL
    SELECT '결함 발생 항목(0건 제외)' AS "비교범위", MIN("결함수") AS "기준결함수"
    FROM isms_defects
    WHERE "결함수" > 0
),
matches AS (
    SELECT
        s."비교범위",
        s."기준결함수",
        d."통제분야",
        d."통제항목",
        ROW_NUMBER() OVER (
            PARTITION BY s."비교범위"
            ORDER BY d."통제분야", d."통제항목"
        ) AS "순번",
        COUNT(*) OVER (PARTITION BY s."비교범위") AS "동률항목수"
    FROM scopes s
    JOIN isms_defects d ON d."결함수" = s."기준결함수"
)
SELECT
    "비교범위",
    "기준결함수" AS "결함수",
    MAX("동률항목수") AS "동률항목수",
    {aggregate_function}(
        CASE WHEN "순번" <= 10 THEN "통제분야" || ' > ' || "통제항목" END,
        ' | ' ORDER BY "순번"
    ) AS "대표항목(최대10개)"
FROM matches
GROUP BY "비교범위", "기준결함수"
ORDER BY "기준결함수";
""".strip()

        if asks_for_maximum:
            if self.db.dialect == "sqlite":
                return f"""
WITH target AS (
    SELECT MAX("결함수") AS "기준결함수" FROM isms_defects
),
matches AS (
    SELECT
        t."기준결함수",
        d."통제분야",
        d."통제항목",
        ROW_NUMBER() OVER (ORDER BY d."통제분야", d."통제항목") AS "순번",
        COUNT(*) OVER () AS "동률항목수"
    FROM target t
    JOIN isms_defects d ON d."결함수" = t."기준결함수"
)
SELECT
    "기준결함수" AS "결함수",
    MAX("동률항목수") AS "동률항목수",
    {aggregate_function}(
        CASE WHEN "순번" <= 10 THEN "통제분야" || ' > ' || "통제항목" END,
        ' | '
    ) AS "대표항목(최대10개)"
FROM matches
GROUP BY "기준결함수";
""".strip()

            return f"""
WITH target AS (
    SELECT MAX("결함수") AS "기준결함수" FROM isms_defects
),
matches AS (
    SELECT
        t."기준결함수",
        d."통제분야",
        d."통제항목",
        ROW_NUMBER() OVER (ORDER BY d."통제분야", d."통제항목") AS "순번",
        COUNT(*) OVER () AS "동률항목수"
    FROM target t
    JOIN isms_defects d ON d."결함수" = t."기준결함수"
)
SELECT
    "기준결함수" AS "결함수",
    MAX("동률항목수") AS "동률항목수",
    {aggregate_function}(
        CASE WHEN "순번" <= 10 THEN "통제분야" || ' > ' || "통제항목" END,
        ' | ' ORDER BY "순번"
    ) AS "대표항목(최대10개)"
FROM matches
GROUP BY "기준결함수";
""".strip()

        return None

    def generate_sql(self, question: str, feedback: str = None) -> str:
        """
        자연어 질문을 SQL 쿼리로 변환

        Args:
            question: 사용자의 자연어 질문
            feedback: 이전 시도의 오류 피드백 (재시도 시)

        Returns:
            생성된 SQL 쿼리
        """
        deterministic_sql = self._get_deterministic_sql(question)
        if deterministic_sql:
            return deterministic_sql

        system_prompt = f"""
당신은 {self.sql_dialect} 전문가입니다.
사용자의 질문을 정확한 SQL 쿼리로 변환하세요.

<database_schema>
{self.schema_info}
</database_schema>

<table_descriptions>
- isms_items: ISMS-P 인증기준 세부 항목 (분야, 분야명, 항목번호, 항목명, 상세내용, 주요 확인사항)
- isms_defects: ISMS 및 ISMS-P 통제항목별 결함 통계 (통제분야, 통제영역, 통제항목, 결함수)
</table_descriptions>

<rules>
- {self.sql_dialect} 문법을 사용하세요
- SELECT 쿼리만 생성하세요 (INSERT, UPDATE, DELETE 금지)
- 일반 목록 결과는 최대 10개로 제한하세요 (LIMIT 10)
- SQL 쿼리만 반환하고, 설명은 포함하지 마세요
- 코드 블록(```)이나 'sql' 키워드 없이 순수 쿼리만 반환하세요
- NULL 값을 주의해서 처리하세요
- 존재하지 않는 컬럼을 사용하지 마세요
- 한글 컬럼명은 반드시 큰따옴표로 감싸세요
- 문자열 검색은 표기와 띄어쓰기 차이를 고려하여 필요한 경우 LIKE와 %를 사용하세요
- 합계는 결과가 없을 때도 0이 되도록 COALESCE(SUM(...), 0)를 사용하세요
- 결함 통계 질문은 isms_defects만 사용하고, 인증기준의 상세내용·주요 확인사항 질문은 isms_items만 사용하세요
- 두 테이블은 서로 다른 인증체계의 번호가 섞여 있고 신뢰할 수 있는 공통 키가 없으므로 JOIN하지 마세요
- 최소·최대 질문은 LIMIT 1로 임의의 한 행만 고르지 말고 같은 결함수의 동률 항목을 모두 반영하세요
- "가장 적게 발생"처럼 0건 포함 여부가 모호하면 0건 포함 결과와 0건 제외 결과를 구분하세요
- ISMS와 ISMS-P의 통제분야와 항목명을 임의로 혼합하지 마세요
- 세미콜론(;)으로 쿼리를 종료하세요
</rules>
"""

        if feedback:
            system_prompt += f"\n\n이전 시도의 오류:\n{feedback}\n\n위 오류를 고려하여 쿼리를 수정하세요."

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]

        response = self.llm.invoke(messages)
        sql_query = response.content.strip()

        # 코드 블록 제거
        if sql_query.startswith("```"):
            lines = sql_query.split("\n")
            sql_query = "\n".join(lines[1:-1]) if len(lines) > 2 else sql_query
            sql_query = sql_query.replace("sql", "").strip()

        return sql_query

    def execute_sql(self, sql_query: str) -> tuple[str, str]:
        """
        SQL 쿼리 실행

        Args:
            sql_query: 실행할 SQL 쿼리

        Returns:
            (결과 문자열, 오류 메시지) 튜플
        """
        try:
            result = self.db.run(sql_query)
            return result, None
        except Exception as e:
            error_msg = str(e)
            return None, error_msg

    def query(self, question: str, previous_error: str = None) -> dict:
        """
        질문에 대한 SQL 생성 및 실행

        Args:
            question: 사용자 질문
            previous_error: 이전 시도의 오류 (재시도 시)

        Returns:
            결과 딕셔너리 (sql_query, result, error)
        """
        # SQL 생성
        sql_query = self.generate_sql(question, feedback=previous_error)

        # SQL 실행
        result, error = self.execute_sql(sql_query)

        return {
            "sql_query": sql_query,
            "result": result,
            "error": error
        }

    def is_empty_result(self, result: str) -> bool:
        """
        결과가 비어있는지 확인

        Args:
            result: SQL 실행 결과

        Returns:
            결과가 비어있으면 True
        """
        if not result:
            return True

        # 빈 결과 패턴 확인
        empty_patterns = ["[]", "()", "no rows", "0 rows"]
        result_lower = result.lower().strip()

        return any(pattern in result_lower for pattern in empty_patterns)


def get_text2sql_engine() -> Text2SQLEngine:
    """Text2SQL 엔진 인스턴스 반환"""
    return Text2SQLEngine()
