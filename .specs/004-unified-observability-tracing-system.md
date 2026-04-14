# Unified Observability & Tracing System Design Specification
**Version**: 1.0
**Date**: 2024-04-14
**Status**: Final
**Author**: Code Graph RAG Team
**Depends On**:
- [001-context-window-management-system.md](./001-context-window-management-system.md)
- [002-parallel-task-orchestration-system-v2.md](./002-parallel-task-orchestration-system-v2.md)
- [003-unified-hybrid-retrieval-system.md](./003-unified-hybrid-retrieval-system.md)

## Revision History
| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2024-04-14 | Initial complete design | Team |

## 1. Purpose
This document specifies the design for the Unified Observability & Tracing System, which addresses critical visibility gaps in Code Graph RAG:
1. No visibility into RAG pipeline execution steps (hard to debug why a query returned specific results)
2. No historical record of query execution for performance optimization
3. No built-in debugging tools to understand search result relevance
4. No performance metrics to identify bottlenecks
5. No audit logging for compliance use cases

The system integrates with all existing components from the preceding specifications to provide end-to-end visibility into every part of the RAG pipeline, from query input to final response generation.

## 2. Scope
### 2.1 In Scope
- Structured, end-to-end tracing for all pipeline steps with OpenTelemetry-compatible span format
- Local trace storage with configurable retention policy
- Built-in debug REST API for accessing trace data
- Query replay functionality to reproduce past execution runs
- Performance metrics aggregation and reporting
- Audit logging for all user interactions and tool calls
- Simple terminal-based debug dashboard for local use
- Dedicated `--debug` CLI flag to enable verbose tracing output
- Integration with all existing components: context compression, parallel orchestration, hybrid retrieval, LLM calls

### 2.2 Out of Scope
- Distributed tracing across multiple nodes/services
- Third-party APM (Application Performance Monitoring) integration (Datadog, New Relic, etc.)
- Alerting and notification capabilities
- User-facing production dashboard UI (only terminal and API provided)
- Long-term cloud storage for traces (local storage only)

## 3. Requirements
### 3.1 Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | System shall trace all pipeline steps with full context: query input, search methods used, results retrieved, LLM prompts, final response, errors | Critical |
| FR2 | System shall store trace history with configurable retention period (default: 7 days) | High |
| FR3 | System shall provide a debug API to retrieve traces, list past queries, and access execution metrics | High |
| FR4 | Users shall be able to replay past queries exactly as they were executed to reproduce issues | High |
| FR5 | System shall collect and aggregate performance metrics: latency per pipeline step, search relevance scores, parallel execution efficiency, token usage | High |
| FR6 | System shall provide audit logging of all user queries, tool calls, and file modifications for compliance purposes | Medium |
| FR7 | `--debug` CLI flag shall enable real-time, colored trace output in the terminal during execution | High |
| FR8 | Tracing shall be fully optional and can be disabled completely for minimal overhead | High |
| FR9 | All existing components shall be instrumented without breaking existing functionality | Critical |

### 3.2 Non-Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| NFR1 | Tracing overhead shall be <5% of total execution time when enabled | Critical |
| NFR2 | Trace storage shall not exceed 1GB per default 7-day retention period | High |
| NFR3 | System shall be fully backwards compatible with no changes required for existing users | Critical |
| NFR4 | Trace data shall be stored in an open format (JSON/SQLite) for easy export and analysis | Medium |
| NFR5 | Debug API shall have authentication disabled by default for local use, with optional API key auth for shared instances | Medium |

## 4. System Architecture
### 4.1 Component Overview
The Unified Observability & Tracing System consists of 6 core components:

```
┌─────────────────────────┐
│   Trace Instrumentation │
│  (added to all existing  │
│   components to generate│
│   span data)             │
└─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│    Trace Collector      │
│  (aggregates spans from │
│   all components into   │
│   complete traces)       │
└─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│     Trace Storage       │
│  (local SQLite/JSON     │
│   storage with retention│
│   policy management)     │
└─────────────────────────┘
┌─────────────────────────┐
│  Metrics Aggregator     │
│  (calculates performance│
│   metrics from trace    │
│   data)                  │
└─────────────────────────┘
┌─────────────────────────┐
│    Audit Logger         │
│  (logs compliance-relevant│
│   events to separate    │
│   storage)               │
└─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│    Debug API & CLI      │
│  (exposes trace data,   │
│   metrics, and replay   │
│   functionality)         │
└─────────────────────────┘
```

### 4.2 End-to-End Workflow
1. **Instrumentation**: All pipeline components generate span data for each operation they perform
2. **Collection**: Trace Collector aggregates spans into complete end-to-end traces with unique trace IDs
3. **Storage**: Traces are stored in local SQLite database with automatic pruning based on retention policy
4. **Metrics/Audit**: Metrics Aggregator processes trace data to generate performance insights, while Audit Logger extracts compliance-relevant events
5. **Access**: Users access trace data via terminal debug output, debug REST API, or query replay functionality

## 5. Implementation Details
### 5.1 Trace Instrumentation
**Files**: Instrumentation added to all existing components
1. OpenTelemetry-compatible span format with standard fields:
   ```json
   {
     "trace_id": "uuid",
     "span_id": "uuid",
     "parent_span_id": "uuid|null",
     "name": "operation_name",
     "start_time": "iso_timestamp",
     "end_time": "iso_timestamp",
     "duration_ms": 123,
     "status": "ok|error",
     "attributes": { /* key-value operation-specific data */ },
     "events": [ /* timestamped events during execution */ ],
     "context": { /* execution context: model used, config values */ }
   }
   ```
2. Instrumentation added to these core components:
   - Query preprocessing, routing, and hybrid retrieval
   - Parallel task orchestration and subtask execution
   - Context compression operations
   - LLM calls (prompt, response, token usage, model used)
   - Tool execution (file reads/writes, shell commands, graph queries)
   - Result aggregation and response generation

### 5.2 Trace Collector
**File**: `codebase_rag/observability/trace_collector.py`
1. Singleton collector that receives spans from all components
2. Correlates spans into complete traces using trace ID propagation
3. Adds standard context attributes to all spans: user ID, session ID, repository path, model config
4. Handles span sampling:
   - Default: 100% sampling for local use
   - Configurable sampling rate for high-throughput deployments
   - Always sample error traces regardless of sampling rate

### 5.3 Trace Storage
**File**: `codebase_rag/observability/trace_storage.py`
1. Default storage backend: SQLite database for structured, queryable trace data
2. Optional lightweight JSON file storage for embedded deployments
3. Automatic retention policy enforcement:
   - Default retention: 7 days
   - Automatic pruning of oldest traces when storage limit is reached (default: 1GB)
4. Simple export functionality to export traces to JSON lines format for external analysis

### 5.4 Metrics Aggregator
**File**: `codebase_rag/observability/metrics_aggregator.py`
1. Calculates core performance metrics from trace data:
   - End-to-end query latency (p50, p95, p99)
   - Latency per pipeline step (retrieval, compression, LLM call, etc.)
   - Retrieval relevance scores per search method
   - Parallel execution efficiency (speedup vs sequential, worker utilization)
   - Token usage per query type, model usage breakdown
   - Error rate per component
2. Provides both real-time and historical metrics
3. Supports filtering by date range, query type, model used, and repository

### 5.5 Audit Logger
**File**: `codebase_rag/observability/audit_logger.py`
1. Logs compliance-relevant events to separate immutable audit log:
   - All user queries and responses
   - All file modification operations (create, replace, delete)
   - All shell command executions
   - Model configuration changes
   - User authentication events (if enabled)
2. Immutable log format: appended only, no modification possible
3. Configurable audit log retention (default: 30 days)
4. Supports export to standard SIEM formats

### 5.6 Debug API & CLI
**File**: `codebase_rag/observability/debug_api.py`
1. REST API endpoints (available on `http://localhost:8080/debug` when enabled):
   - `GET /debug/traces`: List recent traces with filtering options
   - `GET /debug/traces/{trace_id}`: Get full trace details
   - `POST /debug/traces/{trace_id}/replay`: Replay a past query exactly as it was executed
   - `GET /debug/metrics`: Get performance metrics
   - `GET /debug/audit`: Get audit log entries (requires auth)
2. CLI commands:
   - `cgr debug traces`: List recent traces in terminal
   - `cgr debug trace <trace_id>`: Show full trace details
   - `cgr debug replay <trace_id>`: Replay a past query
   - `cgr debug metrics`: Show performance metrics dashboard in terminal
3. `--debug` CLI flag: Enables real-time colored trace output during query execution

### 5.7 Configuration Parameters
Add new environment variables to `codebase_rag/config.py` and `.env.example`:
| Variable Name | Default | Description |
|---------------|---------|-------------|
| `OBSERVABILITY_ENABLED` | true | Enable/disable all observability features |
| `OBSERVABILITY_TRACING_ENABLED` | true | Enable/disable tracing |
| `OBSERVABILITY_TRACING_RETENTION_DAYS` | 7 | Number of days to retain trace data |
| `OBSERVABILITY_TRACING_STORAGE_LIMIT_GB` | 1.0 | Maximum storage size for trace data |
| `OBSERVABILITY_TRACING_SAMPLING_RATE` | 1.0 | Sampling rate for traces (1.0 = 100% sampling) |
| `OBSERVABILITY_METRICS_ENABLED` | true | Enable/disable metrics collection |
| `OBSERVABILITY_AUDIT_LOG_ENABLED` | false | Enable/disable audit logging (disabled by default for privacy) |
| `OBSERVABILITY_AUDIT_LOG_RETENTION_DAYS` | 30 | Number of days to retain audit log entries |
| `OBSERVABILITY_DEBUG_API_ENABLED` | false | Enable/disable debug REST API (disabled by default) |
| `OBSERVABILITY_DEBUG_API_PORT` | 8080 | Port for debug API |
| `OBSERVABILITY_DEBUG_API_AUTH_KEY` | (none) | Optional API key for debug API authentication |

## 6. API Specification
### 6.1 Trace Data Model
```python
@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time: datetime
    end_time: datetime
    duration_ms: int
    status: str  # "ok" | "error"
    attributes: dict[str, Any]
    events: list[dict[str, Any]]

@dataclass
class Trace:
    trace_id: str
    start_time: datetime
    end_time: datetime
    duration_ms: int
    status: str
    spans: list[Span]
    user_query: str
    final_response: str
    repository_path: str
    model_config: dict[str, Any]
```

### 6.2 Debug API Endpoints
```http
GET /debug/traces
Query params: limit=20, offset=0, start_date="2024-01-01", end_date="2024-01-02", status="ok|error"
Response: List of trace summaries
```

```http
GET /debug/traces/{trace_id}
Response: Full Trace object with all spans and context
```

```http
POST /debug/traces/{trace_id}/replay
Request body: { "override_query": "optional new query to replace original" }
Response: Replay execution results with side-by-side comparison to original trace
```

```http
GET /debug/metrics
Query params: range="24h", group_by="step"
Response: Performance metrics data
```

### 6.3 CLI Commands
```bash
# List recent traces
cgr debug traces --limit 20 --last 24h

# Show full trace details
cgr debug trace <trace-id>

# Replay a past query
cgr debug replay <trace-id> [--query "optional override query"]

# Show performance metrics dashboard
cgr debug metrics --range 7d

# Enable real-time debug output for a query
cgr query "your question" --debug
```

## 7. Integration Points
### 7.1 Integration with Existing Components
- **Hybrid Retrieval**: Instrument search methods to capture query, result count, latency, relevance scores
- **Parallel Orchestration**: Instrument subtask execution, worker utilization, speedup metrics
- **Context Compression**: Capture original token count, compressed token count, reduction percentage, retention score
- **LLM Calls**: Capture prompt, response, model used, token count (prompt/completion/total), latency, cost
- **Tool Execution**: Capture tool name, parameters, execution time, result size, errors

### 7.2 Backwards Compatibility
- All observability features are optional and do not change existing functionality
- Tracing is enabled by default with minimal overhead and automatic storage management
- Users who want to disable all observability features can set `OBSERVABILITY_ENABLED=false`

## 8. Testing Plan
### 8.1 Unit Tests
1. **Instrumentation Tests**: Verify all components generate correct span data for both success and error cases
2. **Trace Collector Tests**: Verify span correlation into complete traces works correctly
3. **Storage Tests**: Verify trace storage, retrieval, and retention policy enforcement works correctly
4. **Metrics Tests**: Verify metrics aggregation calculates correct values from trace data
5. **Replay Tests**: Verify query replay functionality exactly reproduces past execution runs

### 8.2 Integration Tests
1. End-to-end tracing test for complete query execution pipeline
2. Performance test to verify <5% overhead when tracing is enabled
3. API test to verify all debug endpoints work correctly
4. CLI test to verify all debug commands work as expected

### 8.3 Edge Case Tests
1. Test with large trace data to verify storage limit enforcement works correctly
2. Test with failed queries to verify error traces are properly captured and stored
3. Test trace export functionality to verify data is exported correctly in open format
4. Test replay functionality with modified query parameters to verify override works correctly

## 9. Migration Guide
This release is fully backwards compatible with no breaking changes:
- All observability features are enabled by default with safe default configuration
- Tracing uses minimal overhead and automatically manages storage to avoid bloat
- Audit logging is disabled by default for privacy, users can enable it if needed
- Debug API is disabled by default for security, users can enable it for development use

## 10. Documentation Updates
1. Update `.env.example` with all new observability configuration options
2. Add observability section to main README including debug CLI usage
3. Add API documentation for all debug endpoints
4. Add troubleshooting guide using tracing to debug common issues
5. Add performance optimization guide using metrics to identify bottlenecks

## 11. Future Enhancements
1. Add third-party APM integration (OpenTelemetry exporter for Datadog, New Relic, Jaeger, etc.)
2. Add alerting capabilities for performance anomalies and errors
3. Add distributed tracing support for multi-service deployments
4. Add web-based debug dashboard UI for easier trace analysis
5. Add custom metrics support for user-defined pipeline steps
6. Add trace comparison functionality to compare execution between different query runs or versions
