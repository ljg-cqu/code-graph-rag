from __future__ import annotations

from codebase_rag.orchestrator.investigation_tracker import InvestigationState
from codebase_rag.orchestrator.sufficiency_analyzer import (
    QuestionType,
    analyze_requirements,
)
from codebase_rag.orchestrator.sufficiency_gatekeeper import (
    MAX_REJECTION_LIMIT,
    evaluate_parallel_worker_sufficiency,
    evaluate_sufficiency,
)
from codebase_rag.tools.tool_descriptions import AgenticToolName


class TestSufficiencyGatekeeperLogic:
    def test_gatekeeper_insufficient_file_read(self) -> None:
        state = InvestigationState(
            rounds_completed=3,
            tools_used={AgenticToolName.QUERY_GRAPH, AgenticToolName.SEMANTIC_SEARCH},
            files_read=[],
        )
        reqs = analyze_requirements("how does the authentication logic work")
        allowed, feedback = evaluate_sufficiency(state, reqs)
        assert not allowed
        assert feedback is not None
        assert "MUST read the actual source files" in feedback

    def test_gatekeeper_insufficient_rounds(self) -> None:
        state = InvestigationState(
            rounds_completed=1,
            tools_used={
                AgenticToolName.QUERY_GRAPH,
                AgenticToolName.SEMANTIC_SEARCH,
                AgenticToolName.READ_FILE,
            },
            files_read=["main.py"],
        )
        reqs = analyze_requirements("how does the authentication logic work")
        allowed, feedback = evaluate_sufficiency(state, reqs)
        assert not allowed
        assert feedback is not None
        assert "at least 3 rounds" in feedback

    def test_gatekeeper_sufficient_functional(self) -> None:
        state = InvestigationState(
            rounds_completed=3,
            tools_used={
                AgenticToolName.SEMANTIC_SEARCH,
                AgenticToolName.QUERY_GRAPH,
                AgenticToolName.READ_FILE,
            },
            files_read=["main.py"],
        )
        reqs = analyze_requirements("how does the authentication logic work")
        allowed, feedback = evaluate_sufficiency(state, reqs)
        assert allowed
        assert feedback is None

    def test_gatekeeper_max_rejection_override(self) -> None:
        state = InvestigationState(
            rounds_completed=1,
            tools_used={AgenticToolName.QUERY_GRAPH, AgenticToolName.SEMANTIC_SEARCH},
            files_read=[],
        )
        reqs = analyze_requirements("how does the authentication logic work")
        allowed, feedback = evaluate_sufficiency(
            state, reqs, rejection_count=MAX_REJECTION_LIMIT
        )
        assert allowed
        assert feedback is None

    def test_gatekeeper_graceful_degradation(self) -> None:
        state = InvestigationState(
            rounds_completed=3,
            tools_used={AgenticToolName.READ_FILE},
            files_read=["main.py"],
            tool_failures={AgenticToolName.SEMANTIC_SEARCH, AgenticToolName.QUERY_GRAPH},
        )
        reqs = analyze_requirements("how does the authentication logic work")
        allowed, feedback = evaluate_sufficiency(state, reqs)
        assert allowed
        assert feedback is None

    def test_gatekeeper_all_file_tools_failed(self) -> None:
        state = InvestigationState(
            rounds_completed=3,
            tools_used={AgenticToolName.SEMANTIC_SEARCH, AgenticToolName.QUERY_GRAPH},
            files_read=[],
            tool_failures={
                AgenticToolName.READ_FILE,
                AgenticToolName.GET_CODE_SNIPPET,
                AgenticToolName.GET_FUNCTION_SOURCE,
            },
        )
        reqs = analyze_requirements("how does the authentication logic work")
        allowed, feedback = evaluate_sufficiency(state, reqs)
        assert allowed
        assert feedback is None


class TestSufficiencyAnalyzerClassification:
    def test_analyze_diagnostic(self) -> None:
        reqs = analyze_requirements(
            "Why is the authentication failing with a 401 error?"
        )
        assert reqs.question_type == QuestionType.DIAGNOSTIC
        assert reqs.requires_cross_validation is True
        assert reqs.requires_file_read is True
        assert reqs.min_rounds == 3

    def test_analyze_functional(self) -> None:
        reqs = analyze_requirements(
            "How does the context compression mechanism work?"
        )
        assert reqs.question_type == QuestionType.FUNCTIONAL
        assert reqs.requires_file_read is True
        assert reqs.min_rounds == 3

    def test_analyze_structural(self) -> None:
        reqs = analyze_requirements("List all classes in the services module")
        assert reqs.question_type == QuestionType.STRUCTURAL
        assert reqs.requires_file_read is False
        assert reqs.min_rounds == 2


class TestInvestigationTracker:
    def test_tracker_records_all_file_tools(self) -> None:
        state = InvestigationState()
        state.record_tool(AgenticToolName.READ_FILE, "main.py")
        state.record_tool(AgenticToolName.GET_CODE_SNIPPET, "MyClass.method")
        state.record_tool(AgenticToolName.GET_FUNCTION_SOURCE, "node-123")
        assert "main.py" in state.files_read
        assert "MyClass.method" in state.files_read
        assert "node-123" in state.files_read
        assert AgenticToolName.READ_FILE in state.tools_used
        assert AgenticToolName.GET_CODE_SNIPPET in state.tools_used
        assert AgenticToolName.GET_FUNCTION_SOURCE in state.tools_used

    def test_tracker_detects_tool_failure_via_record_tool(self) -> None:
        state = InvestigationState()
        state.record_tool(AgenticToolName.SEMANTIC_SEARCH, "auth logic", results_count=0)
        assert AgenticToolName.SEMANTIC_SEARCH in state.tool_failures

    def test_tracker_parallel_worker_state(self) -> None:
        state = InvestigationState.from_parallel_worker(worker_id=5)
        assert isinstance(state, InvestigationState)
        assert state.rounds_completed == 0
        assert len(state.tools_used) == 0
        assert len(state.files_read) == 0

    def test_tracker_graph_queries_incremented(self) -> None:
        state = InvestigationState()
        state.record_tool(AgenticToolName.QUERY_GRAPH, "find all classes")
        state.record_tool(AgenticToolName.QUERY_GRAPH, "find all functions")
        assert state.graph_queries_run == 2

    def test_tracker_no_failure_for_positive_results(self) -> None:
        state = InvestigationState()
        state.record_tool(AgenticToolName.SEMANTIC_SEARCH, "auth logic", results_count=5)
        assert AgenticToolName.SEMANTIC_SEARCH not in state.tool_failures


class TestParallelWorkerSufficiency:
    def test_parallel_worker_sufficiency_pass(self) -> None:
        state = InvestigationState(
            tools_used={AgenticToolName.QUERY_GRAPH, AgenticToolName.READ_FILE},
            files_read=["auth.py"],
        )
        reqs = analyze_requirements("how does authentication work")
        allowed, warning = evaluate_parallel_worker_sufficiency(
            state, reqs, worker_id=0
        )
        assert allowed
        assert warning is None

    def test_parallel_worker_sufficiency_warning(self) -> None:
        state = InvestigationState(
            tools_used={AgenticToolName.SEMANTIC_SEARCH},
            files_read=[],
        )
        reqs = analyze_requirements("how does authentication work")
        allowed, warning = evaluate_parallel_worker_sufficiency(
            state, reqs, worker_id=3
        )
        assert not allowed
        assert warning is not None
        assert "[Worker 3]" in warning
        assert "file-level evidence" in warning

    def test_parallel_worker_graph_not_queried_warning(self) -> None:
        state = InvestigationState(
            tools_used={AgenticToolName.READ_FILE},
            files_read=["main.py"],
        )
        reqs = analyze_requirements("list all classes in the module")
        allowed, warning = evaluate_parallel_worker_sufficiency(
            state, reqs, worker_id=7
        )
        assert not allowed
        assert warning is not None
        assert "[Worker 7]" in warning
        assert "code graph" in warning

    def test_parallel_worker_graph_failure_skipped(self) -> None:
        state = InvestigationState(
            tools_used={AgenticToolName.READ_FILE},
            files_read=["main.py"],
            tool_failures={AgenticToolName.QUERY_GRAPH},
        )
        reqs = analyze_requirements("list all classes in the module")
        allowed, warning = evaluate_parallel_worker_sufficiency(
            state, reqs, worker_id=2
        )
        assert allowed
        assert warning is None

