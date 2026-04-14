# Configurable Worker LLM for Parallel Sub-Agents Design Specification
## Version: 1.0 | Last Updated: 2024-XX-XX | Status: Draft

## 1. Overview & Problem Statement
Current Code-Graph-RAG concurrent sub-agent implementation uses the same orchestrator LLM for all sub-agent workers by default, with no support for configuring separate LLM models specifically for sub-agent tasks. This prevents users from optimizing cost and performance for parallel workloads (e.g. using smaller/cheaper models for sub-agent tasks while keeping a more powerful model for the main orchestrator and Cypher generation).

### User Requirement
Enable users to configure a dedicated list of LLMs for sub-agent workers, with the following requirements:
1. **Round-robin assignment**: Orchestrator assigns configured worker LLMs to sub-agents in round-robin fashion
2. **Dynamic prompt configuration**: Users can specify worker LLMs directly in their chat prompts (e.g. "Use gpt-4o-mini and claude-3-haiku for the 8 parallel workers")
3. **Fallback behavior**: If no worker LLMs are configured, sub-agents continue to use the orchestrator LLM as default
4. **Cypher LLM isolation**: The dedicated Cypher LLM remains unchanged and is used exclusively for Cypher query generation by both orchestrator and sub-agents, no user configurable list for Cypher LLM
5. **Full feature parity**: Sub-agents using worker LLMs have access to all existing tools and resources (graph queries, Cypher LLM, file access, etc.) just like the main orchestrator

## 2. Target Use Cases
| Use Case | Description |
|----------|-------------|
| Cost optimization | Use cheaper smaller models (e.g. gpt-4o-mini, claude-3-haiku, local Ollama models) for parallel sub-agent tasks, while retaining a powerful model for the main orchestrator |
| Performance optimization | Distribute parallel workload across multiple LLM providers/models to avoid rate limits and reduce total execution time |
| Heterogeneous task assignment | Use specialized models for specific subtask types (e.g. code-specific models for code review, general reasoning models for documentation analysis) |
| Dynamic per-task configuration | Specify different worker models for different parallel tasks directly in chat prompts, no permanent config changes required |

## 3. Design Goals
### Goals
1. Zero breaking changes to existing functionality: existing sub-agent workflows without worker LLM configuration continue to work unchanged
2. Minimal code modification: build on existing sub-agent orchestrator and config system with minimal changes
3. Support both static configuration (environment variables) and dynamic configuration (chat prompts)
4. Round-robin LLM assignment for even workload distribution
5. Full validation of worker LLM configurations before task execution
6. Automatic fallback to orchestrator LLM if no worker LLMs are configured

### Non-Goals
1. Per-subtask LLM routing (v1 uses uniform round-robin assignment only)
2. LLM performance-based routing (v1 uses static round-robin only)
3. Dynamic worker LLM adjustment mid-execution (v1 configures worker LLMs once per task execution)

## 4. Architecture Design
### Integration with Existing System
The new configurable worker LLM feature extends the existing concurrent sub-agent system with minimal changes:

```
┌──────────────────────────────────────────────────────────┐
│                      USER INTERFACE                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │ New intent detection for worker LLM configuration  │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────┬────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────┐
│                     AppConfig Layer                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │ New worker_llms config field (list of ModelConfig) │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────┬────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────┐
│                RAG Orchestrator Layer                    │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Updated SubAgentOrchestrator with LLM round-robin  │  │
│  │ assignment logic, agent factory accepts LLM config │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────┬─────────────┬────────────────────┘
                      │             │
          ┌───────────▼─────────┐   │
          │   Sub-Agent Pool    │   │
          │ (each agent uses    │   │
          │  assigned worker    │   │
          │  LLM config)        │   │
          └───────────┬─────────┘   │
                      │             │
                      └─────────────▼────────────────────┐
                                                    ┌────▼────┐
                                                    │ Existing│
                                                    │ Services│
                                                    │ Layer   │
                                                    │ (Cypher │
                                                    │ LLM, Graph│
                                                    │  etc.)  │
                                                    └─────────┘
```

## 5. Core Changes
### 5.1 Configuration Updates (AppConfig)
New config options added to `AppConfig` in `codebase_rag/config.py`:
| Config Option | Default | Description |
|---------------|---------|-------------|
| `CGR_WORKER_LLMS` | Empty list | Comma-separated list of LLM model strings in `provider:model` format, or JSON array of full ModelConfig objects for worker LLMs |
| `CGR_WORKER_LLM_ASSIGNMENT_STRATEGY` | `round-robin` | Strategy for assigning worker LLMs to sub-agents (only `round-robin` supported in v1) |

#### Example static configuration via environment variable:
```bash
# Use gpt-4o-mini and claude-3-haiku as worker LLMs
CGR_WORKER_LLMS="openai:gpt-4o-mini,anthropic:claude-3-haiku-20240307"
```

#### New AppConfig properties and methods:
1. `active_worker_llms: list[ModelConfig]`: Parsed list of validated worker LLM configurations
2. `set_worker_llms(llms: list[str | ModelConfig])`: Method to dynamically set worker LLMs (used for prompt-based configuration)
3. Updated `_get_default_config()` method to support parsing worker LLM entries

### 5.2 SubAgentOrchestrator Updates
Changes to `SubAgentOrchestrator` in `codebase_rag/orchestrator/subagent_orchestrator.py`:
1. New `llm_index: int` counter for round-robin LLM assignment
2. Updated `initialize_agents()` method to assign LLMs to agents in round-robin fashion:
   - If `active_worker_llms` is not empty: cycle through the list assigning each agent the next LLM config in sequence
   - If `active_worker_llms` is empty: assign orchestrator LLM config to all agents (existing default behavior)
3. Updated `_default_agent_factory()` method to accept an optional `llm_config: ModelConfig` parameter, which overrides the default orchestrator LLM for the sub-agent
4. Updated `adjust_worker_count()` method to assign LLMs to newly added agents using the same round-robin logic

### 5.3 Intent Detection Updates
Changes to the Task Splitter intent detection logic in `codebase_rag/orchestrator/task_splitter.py` to recognize worker LLM configuration in user prompts:
#### Supported natural language triggers (examples):
```
"Review the codebase with 10 parallel workers using gpt-4o-mini"
"Use claude-3-haiku and llama3 as worker LLMs for this parallel task"
"Spawn 8 sub-agents using openai:gpt-4o-mini and anthropic:claude-3-sonnet"
```

When worker LLM triggers are detected:
1. Extract all LLM model strings from the prompt
2. Validate each model string using the existing `parse_model_string()` method
3. Call `settings.set_worker_llms()` to dynamically configure the worker LLM list for this task
4. Confirm the configuration to the user before execution (unless YOLO mode is enabled)

### 5.4 Sub-Agent Initialization
Each sub-agent instance will:
1. Use its assigned worker LLM config for all general reasoning tasks
2. Continue to use the global shared Cypher LLM config for all Cypher query generation tasks (unchanged from existing behavior)
3. Inherit all other configuration (tool permissions, rate limits, timeout settings) from the main orchestrator config (unchanged)

## 6. User Interface
### 6.1 Static Configuration
Users can configure worker LLMs permanently via environment variables or `.env` file:
```bash
# Example .env configuration
CGR_WORKER_LLMS="ollama:llama3,openai:gpt-4o-mini,anthropic:claude-3-haiku-20240307"
```

### 6.2 Dynamic Prompt Configuration
Users can specify worker LLMs directly in their chat prompts for individual tasks, no permanent config changes required:
```
"Generate unit tests for all functions using 6 parallel workers with gpt-4o-mini and claude-3-haiku"
```

The system will confirm the configuration before execution:
```
I will spawn 6 parallel workers using the following LLMs (round-robin assignment):
1. OpenAI gpt-4o-mini
2. Anthropic claude-3-haiku-20240307

Estimated time: 2 minutes. Proceed? [Y/n]
```

### 6.3 CLI Support
New optional CLI flag for explicit worker LLM configuration:
```bash
cgr ask-agent "Review all Python files for security vulnerabilities" --workers 8 --worker-llms "openai:gpt-4o-mini,anthropic:claude-3-haiku"
```

## 7. Security & Validation
1. **LLM Configuration Validation**: All worker LLM configurations are validated using the existing `ModelConfig.validate_api_key()` method before task execution, with clear error messages if any API keys are missing
2. **No Privilege Escalation**: Sub-agents using worker LLMs have the same permissions as sub-agents using the orchestrator LLM (read-only by default, write operations require approval)
3. **Rate Limiting**: Per-worker rate limiting is applied to all worker LLMs to prevent hitting API limits
4. **Cypher LLM Isolation**: The Cypher LLM configuration is completely separate and cannot be modified via worker LLM configuration, ensuring consistent Cypher generation performance

## 8. Implementation Roadmap
### Phase 1 (MVP)
- [ ] Add worker LLM config fields to AppConfig with parsing and validation
- [ ] Update SubAgentOrchestrator to support round-robin LLM assignment
- [ ] Update agent factory to accept custom LLM config
- [ ] Add CLI flag for worker LLM configuration
- [ ] Add basic validation of worker LLM configurations

### Phase 2
- [ ] Add intent detection for worker LLM configuration in user prompts
- [ ] Add confirmation UI for dynamic worker LLM configuration
- [ ] Add support for full ModelConfig objects in worker LLM configuration (custom endpoints, API keys, etc.)
- [ ] Add execution metrics showing LLM usage per worker

## 9. Testing Strategy
1. **Unit Tests**: Test LLM parsing, validation, and round-robin assignment logic
2. **Integration Tests**: Test end-to-end parallel execution with configured worker LLMs
3. **Fallback Tests**: Verify fallback to orchestrator LLM when no worker LLMs are configured
4. **Validation Tests**: Verify proper error handling for invalid worker LLM configurations and missing API keys
5. **Security Tests**: Verify Cypher LLM remains isolated and cannot be modified via worker LLM configuration

## 10. Compatibility
- Fully backward compatible: existing workflows without worker LLM configuration continue to work unchanged
- Works with all existing LLM providers (OpenAI, Google, Anthropic, Ollama, Azure, etc.)
- Compatible with all existing sub-agent features (auto-scaling, retries, progress UI, etc.)
- No new external dependencies required, all changes build on existing codebase modules

## 11. Implementation Complexity Assessment
MVP is very low-risk and low-effort to implement, estimated 3-5 days of development:
- Config updates + validation: ~1 day
- SubAgentOrchestrator LLM assignment logic: ~1 day
- Intent detection + UI updates: ~1 day
- Integration testing + documentation: ~1 day
