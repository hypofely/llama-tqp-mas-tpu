# LatentMAS Integration Requirements

## Overview

The current project implements a C++ Multi-Agent System (MAS) with KV cache cloning and adapter support. Several LatentMAS projects extend the original research with different approaches. This document outlines what would be required to integrate these projects.

## Current MAS Implementation

The current C++ implementation in `llama-tqp-mas-cpp` includes:

- **Multi-agent generation** with RecursiveLink and CrossLink adapters
- **KV cache cloning** via `llama_mas_kv_clone()`
- **CUDA adapter acceleration** for mean pooling and forward passes
- **HTTP API** for agent management and generation
- **Hidden states persistence** (JSON-based, with on/off toggle)

## LatentMAS Projects Analysis

### 1. Gen-Verse/LatentMAS (Original)

**Repository:** https://github.com/Gen-Verse/LatentMAS

**Purpose:** Original LatentMAS research implementation (Python-based)

**Key Features:**
- Latent space collaboration between agents
- Shared KV cache mechanism
- Recursive cross-attention between agents

**Integration Requirements:**
- **Not needed:** This is the original research that the C++ implementation is based on
- The current C++ implementation already implements the core concepts from this project

### 2. Bookmaster9/kNN-latentMAS

**Repository:** https://github.com/Bookmaster9/kNN-latentMAS

**Purpose:** Extends LatentMAS with k-nearest neighbors for agent selection

**Key Features:**
- Uses kNN to select relevant agents based on query similarity
- Improves efficiency by not using all agents for every query
- Reduces computational overhead

**Integration Requirements:**

**Architecture Changes:**
- Implement kNN index for agent embeddings
- Store agent prompt embeddings in a searchable structure
- Add agent selection logic before generation

**Implementation Steps:**
1. **Embedding Storage:**
   - Store prompt embeddings for each agent during initialization
   - Use FAISS or similar library for efficient kNN search
   - Add embedding dimension configuration

2. **kNN Selection:**
   - Add HTTP endpoint for kNN-based agent selection
   - Implement `llama_mas_select_agents_knn()` function
   - Support configurable k (number of neighbors)

3. **API Extensions:**
   ```
   POST /agents/select-knn
   {
       "query": "user query text",
       "k": 5,
       "agent_ids": ["agent1", "agent2", ...]
   }
   ```

4. **Dependencies:**
   - FAISS or Annoy for kNN search
   - Embedding model (e.g., sentence-transformers)
   - Python subprocess or C++ embedding library

**Estimated Effort:** 1-2 weeks

### 3. nhminle/LatentMAS-Hybrid

**Repository:** https://github.com/nhminle/LatentMAS-Hybrid

**Purpose:** Hybrid approach combining LatentMAS with other collaboration mechanisms

**Key Features:**
- Combines explicit and implicit collaboration
- Mixes latent and token-level interactions
- Adaptive collaboration strategies

**Integration Requirements:**

**Architecture Changes:**
- Add hybrid collaboration mode to MAS context
- Implement token-level interaction mechanisms
- Add adaptive strategy selection

**Implementation Steps:**
1. **Hybrid Modes:**
   - Add `llama_mas_mode_t` enum: LATENT, TOKEN, HYBRID
   - Implement token-level cross-attention
   - Add mode switching logic

2. **Adaptive Strategy:**
   - Implement performance monitoring
   - Add automatic mode selection based on context
   - Add manual mode override via API

3. **API Extensions:**
   ```
   POST /mas/set-mode
   {
       "mode": "hybrid",
       "adaptive": true
   }
   ```

**Estimated Effort:** 2-3 weeks

### 4. Arifuzzamanjoy/latent_mas_slora

**Repository:** https://github.com/Arifuzzamanjoy/latent_mas_slora

**Purpose:** Integrates Low-Rank Adaptation (LoRA) with LatentMAS

**Key Features:**
- Uses LoRA adapters for agent specialization
- Reduces parameter count compared to full adapters
- Enables dynamic agent specialization

**Integration Requirements:**

**Architecture Changes:**
- Add LoRA adapter support to llama backend
- Implement per-agent LoRA loading
- Add LoRA adapter management API

**Implementation Steps:**
1. **LoRA Integration:**
   - Extend llama.cpp to support LoRA (if not already available)
   - Add LoRA adapter loading to `llama_mas_context`
   - Implement per-agent LoRA paths

2. **Adapter Management:**
   - Add HTTP endpoints for LoRA adapter CRUD operations
   - Implement adapter swapping without full model reload
   - Add adapter merging strategies

3. **API Extensions:**
   ```
   POST /agents/agent1/lora/load
   {
       "lora_path": "/path/to/adapter.safetensors"
   }
   ```

4. **Dependencies:**
   - LoRA adapter format support (safetensors)
   - Adapter merging utilities

**Estimated Effort:** 2-4 weeks

### 5. VectorArc/avp-python

**Repository:** https://github.com/VectorArc/avp-python

**Purpose:** Agent Vector Protocol (AVP) for sharing KV-cache and hidden states between agents

**Key Features:**
- Standardized protocol for vector sharing
- Enables cross-process agent communication
- Supports hidden state sharing (what you requested!)

**Integration Requirements:**

**Architecture Changes:**
- Implement AVP protocol in C++ mas-server
- Add hidden state extraction from llama internals
- Add cross-process communication

**Implementation Steps:**
1. **Hidden State Extraction:**
   - Hook into llama internal layers to capture hidden states
   - Implement `llama_get_hidden_states()` function
   - Add per-layer state capture

2. **AVP Protocol Implementation:**
   - Implement AVP message format (likely JSON/protobuf)
   - Add state serialization/deserialization
   - Implement agent discovery and handshake

3. **Cross-Process Communication:**
   - Add ZeroMQ or gRPC for inter-process communication
   - Implement agent registration and discovery
   - Add state synchronization mechanisms

4. **API Extensions:**
   ```
   POST /avp/register
   {
       "agent_id": "agent1",
       "endpoint": "tcp://localhost:9001"
   }
   
   POST /avp/share-state
   {
       "agent_id": "agent1",
       "layer": 10,
       "state": {...}
   }
   ```

5. **Current Hidden States Feature:**
   - The current JSON-based hidden states storage is a basic version
   - AVP would provide a more robust, standardized approach
   - Could extend current implementation to support AVP protocol

**Estimated Effort:** 3-5 weeks

## Integration Priority

### High Priority
1. **VectorArc/avp-python** - Directly addresses your requirement for hidden states sharing
   - Provides standardized protocol
   - Enables true cross-agent collaboration
   - Builds on current hidden states feature

### Medium Priority
2. **Bookmaster9/kNN-latentMAS** - Efficiency improvement
   - Reduces computational overhead
   - Useful for large-scale agent systems

### Low Priority
3. **nhminle/LatentMAS-Hybrid** - Experimental feature
   - Complex implementation
   - Unclear performance benefits

4. **Arifuzzamanjoy/latent_mas_slora** - Alternative to current adapters
   - Current adapter system works well
   - LoRA may not provide significant benefits for this use case

## Recommended Integration Path

### Phase 1: AVP Integration (3-5 weeks)
1. Implement hidden state extraction from llama internals
2. Add AVP protocol support to mas-server
3. Implement inter-process communication
4. Test with simple 2-agent setup
5. Extend to multi-agent scenarios

### Phase 2: kNN Integration (1-2 weeks)
1. Add embedding storage and indexing
2. Implement kNN selection logic
3. Integrate with existing MAS generation
4. Benchmark against full-agent approach

### Phase 3: Evaluation (1 week)
1. Compare performance of integrated features
2. Evaluate quality improvements
3. Decide on further integration

## Technical Challenges

### AVP Integration
1. **Hidden State Access:** Llama.cpp doesn't expose intermediate layer outputs
   - **Solution:** Modify llama.cpp to add hooks or use internal state access
2. **Protocol Complexity:** AVP protocol may be complex
   - **Solution:** Start with simplified subset, extend as needed
3. **Synchronization:** Cross-process state synchronization
   - **Solution:** Use proven messaging libraries (ZeroMQ)

### kNN Integration
1. **Embedding Computation:** Need efficient embedding generation
   - **Solution:** Use lightweight embedding model or on-the-fly computation
2. **Index Updates:** Dynamic agent addition/removal
   - **Solution:** Use dynamic index structures (FAISS)

## Dependencies Summary

| Project | New Dependencies | Complexity |
|---------|----------------|------------|
| kNN-latentMAS | FAISS, embedding model | Medium |
| LatentMAS-Hybrid | None (internal changes) | High |
| latent_mas_slora | safetensors, LoRA support | High |
| avp-python | ZeroMQ/gRPC, llama modifications | High |

## Current Implementation vs Projects

**What We Already Have:**
- Multi-agent generation with adapters
- KV cache cloning
- Basic hidden states storage (JSON)
- HTTP API for management
- CUDA acceleration

**What's Missing:**
- Standardized hidden states sharing protocol (AVP provides this)
- Efficient agent selection (kNN provides this)
- Adaptive collaboration strategies (Hybrid provides this)
- LoRA adapter support (slora provides this)

## Recommendation

Start with **VectorArc/avp-python** integration as it directly addresses your hidden states sharing requirement and provides a standardized protocol for agent collaboration. The current JSON-based hidden states feature can be extended to support the AVP protocol.
