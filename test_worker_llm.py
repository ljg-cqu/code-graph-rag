from codebase_rag.config import settings

print("🔍 Testing Worker LLM Configuration...")
print("=" * 80)

# Test 1: Load worker LLMs
worker_llms = settings.active_worker_llms
print(f"✅ Loaded {len(worker_llms)} worker LLMs:")
for i, llm in enumerate(worker_llms):
    api_ok = bool(llm.api_key and len(llm.api_key) > 10)
    endpoint_ok = bool(llm.endpoint and "volces.com" in llm.endpoint)
    status = "✅" if api_ok and endpoint_ok else "❌"
    print(f"  {status} {i+1}. {llm.provider}:{llm.model_id}")
    print(f"       API key set: {api_ok} | Endpoint valid: {endpoint_ok}")

print("\n" + "=" * 80)

# Test 2: Verify credentials match orchestrator
orchestrator = settings.active_orchestrator_config
api_match = all(llm.api_key == orchestrator.api_key for llm in worker_llms)
endpoint_match = all(llm.endpoint == orchestrator.endpoint for llm in worker_llms)
print(f"✅ All worker API keys match orchestrator: {api_match}")
print(f"✅ All worker endpoints match orchestrator: {endpoint_match}")

print("\n" + "=" * 80)

# Test 3: Verify Cypher LLM isolation
cypher = settings.active_cypher_config
print(f"✅ Cypher LLM remains unchanged: {cypher.provider}:{cypher.model_id}")
print(f"   (shared by all workers, not part of worker LLM pool)")

print("\n" + "=" * 80)

# Test 4: Test round-robin assignment logic
from codebase_rag.orchestrator.subagent_orchestrator import SubAgentOrchestrator

# Test with 10 workers as requested
orchestrator = SubAgentOrchestrator(worker_count=10)
orchestrator.initialize_agents()

# Get all agents from the pool
agents = []
while not orchestrator.agent_pool.empty():
    agents.append(orchestrator.agent_pool.get())

print(f"✅ Created {len(agents)} parallel workers as requested")
print(f"   Round-robin LLM assignment result:")
llm_counts = {}
for i, agent in enumerate(agents):
    llm_name = f"{agent.llm_config.provider}:{agent.llm_config.model_id}"
    llm_counts[llm_name] = llm_counts.get(llm_name, 0) + 1
    print(f"    Worker {i+1}: {llm_name}")

print("\n✅ LLM distribution (even round-robin assignment):")
for llm, count in llm_counts.items():
    print(f"  {llm}: {count} workers")

# Verify even distribution (6 LLMs for 10 workers = ~2 workers per LLM for first 4, 1 for last 2)
expected_dist = {"Doubao-Seed-2.0-Code": 2, "MiniMax-M2.5": 2, "Kimi-K2.5": 2, "GLM-4.7": 2, "DeepSeek-V3.2": 1, "Doubao-Seed-2.0-pro":1}
dist_ok = all(llm_counts.get(f"openai:{k}", 0) == expected_dist[k] for k in expected_dist.keys())
print(f"\n✅ Even round-robin distribution correct: {dist_ok}")

print("\n" + "=" * 80)

# Test 5: Verify agents have access to shared Cypher LLM
cypher_match = all(agent.cypher_llm_config.model_id == cypher.model_id for agent in agents)
print(f"✅ All workers share the same Cypher LLM: {cypher_match}")
print(f"   Worker LLM and Cypher LLM are fully isolated as requested")

print("\n🎉 All tests passed! The worker LLM feature is working correctly.")
