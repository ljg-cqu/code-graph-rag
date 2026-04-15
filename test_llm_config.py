#!/usr/bin/env python3
"""Test LLM and embedding configuration validity."""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Test 1: Orchestrator LLM connectivity
print("=== Testing Orchestrator LLM ===")
try:
    oai_client = OpenAI(
        api_key=os.getenv("ORCHESTRATOR_API_KEY"),
        base_url=os.getenv("ORCHESTRATOR_ENDPOINT"),
    )
    response = oai_client.chat.completions.create(
        model=os.getenv("ORCHESTRATOR_MODEL"),
        messages=[
            {"role": "user", "content": "Hello! Return only 'OK' if you receive this."}
        ],
        max_tokens=10,
        temperature=0.0,
    )
    result = response.choices[0].message.content.strip()
    print(f"✅ Orchestrator LLM works! Response: {result}")
except Exception as e:
    print(f"❌ Orchestrator LLM failed: {str(e)}")

# Test 2: Cypher LLM connectivity
print("\n=== Testing Cypher LLM ===")
try:
    oai_client = OpenAI(
        api_key=os.getenv("CYPHER_API_KEY"),
        base_url=os.getenv("CYPHER_ENDPOINT"),
    )
    response = oai_client.chat.completions.create(
        model=os.getenv("CYPHER_MODEL"),
        messages=[
            {"role": "user", "content": "Hello! Return only 'OK' if you receive this."}
        ],
        max_tokens=10,
        temperature=0.0,
    )
    result = response.choices[0].message.content.strip()
    print(f"✅ Cypher LLM works! Response: {result}")
except Exception as e:
    print(f"❌ Cypher LLM failed: {str(e)}")

# Test 3: Embedding API connectivity
print("\n=== Testing Embedding API ===")
try:
    oai_client = OpenAI(
        api_key=os.getenv("EMBEDDING_API_KEY"),
        base_url=os.getenv("EMBEDDING_BASE_URL"),
    )
    response = oai_client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=["Test text"],
    )
    dim = len(response.data[0].embedding)
    print(f"✅ Embedding API works! Dimension returned: {dim}")
    # Check dimension mismatch
    expected_doc_dim = int(os.getenv("DOC_MEMGRAPH_VECTOR_DIM", 768))
    if dim != expected_doc_dim:
        print(
            f"⚠️  MISMATCH: Embedding dimension {dim} != DOC_MEMGRAPH_VECTOR_DIM {expected_doc_dim}"
        )
        print(
            "   You need to update DOC_MEMGRAPH_VECTOR_DIM to 1024 to match text-embedding-v4"
        )
except Exception as e:
    print(f"❌ Embedding API failed: {str(e)}")

print("\n=== Configuration Issues Found ===")
issues = []
if os.getenv("DOC_MEMGRAPH_VECTOR_DIM") == "768":
    issues.append(
        "DOC_MEMGRAPH_VECTOR_DIM is set to 768 but text-embedding-v4 outputs 1024 dimensions\n   Fix: Change DOC_MEMGRAPH_VECTOR_DIM=1024 in .env"
    )

if issues:
    for issue in issues:
        print(f"⚠️  {issue}")
else:
    print("✅ No configuration issues found!")
