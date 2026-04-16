# Distributed Training and SETI@home-Style Compute Sharing Research Report

## Executive Summary

This report surveys distributed training techniques, federated learning approaches, BOINC/SETI@home architecture, and emerging decentralized AI training projects. It provides architectural patterns and technical requirements for implementing SwarmResearch v0.3-style public compute sharing for AI research.

---

## 1. Distributed Training Techniques

### 1.1 Data Parallelism (DP)

**Concept**: Distributes input data across multiple devices, with each GPU maintaining a complete copy of model parameters.

**How it works**:
- Training batch is divided across data samples
- Each GPU hosts a local model replica for forward/backward propagation
- Gradients synchronized via AllReduce operations after each iteration

**Key Optimizations**:
- **ZeRO (Zero Redundancy Optimizer)**: Partitions optimizer states, gradients, and parameters across GPUs
- **FSDP (Fully Sharded Data Parallel)**: Shards model parameters across data-parallel workers
- Reduces memory redundancy while maintaining training efficiency

**Communication Pattern**: AllReduce for gradient synchronization

**Challenges**:
- Memory-intensive due to replicated model copies
- Communication overhead increases with model size
- Each node must fit the entire model (impractical for billion-parameter models in decentralized settings)

### 1.2 Model Parallelism (MP)

**Concept**: Divides model parameters across GPUs, with each device holding only a portion of the model.

**Variants**:

#### Tensor Parallelism (TP)
- Partitions weight matrices within layers across multiple devices
- Enables parallel matrix operations
- Requires frequent AllReduce operations (typically 2 per Transformer layer)
- Best suited for single-node deployments with fast interconnects (NVLink)
- Communication-intensive but effective for memory reduction

**Communication Pattern**: AllGather and Reduce-Scatter operations

#### Pipeline Parallelism (PP)
- Segments model into sequential stages assigned to different devices
- Processes micro-batches in a pipelined manner
- Lower communication bandwidth requirements (only at layer boundaries)
- Suffers from pipeline "bubbles" (idle slots) due to dependencies

**Key Algorithms**:
- **GPipe**: Micro-batching to reduce forward bubbles
- **PipeDream**: One-forward-one-backward (1F1B) scheme for memory optimization
- **AdaPipe**: Dynamic programming for model partitioning

**Communication Pattern**: Point-to-point send/receive between pipeline stages

### 1.3 3D/4D Parallelism

Modern distributed training combines multiple parallelism strategies:

| Parallelism | Target | Communication | Best For |
|------------|--------|---------------|----------|
| Data Parallelism | Input batch | AllReduce | Large datasets |
| Tensor Parallelism | Layer weights | AllGather/Reduce-Scatter | Single-node, large layers |
| Pipeline Parallelism | Model layers | P2P Send/Receive | Cross-node, deep models |
| Sequence/Context Parallelism | Sequence dimension | Ring/All-to-All | Long-context training |
| Expert Parallelism (EP) | MoE experts | All-to-All | Mixture-of-Experts models |

**Frameworks**: Megatron-LM, DeepSpeed, MegaScale

---

## 2. Federated Learning Approaches

### 2.1 Core Concepts

**Federated Learning (FL)** trains a shared global model by aggregating models from multiple clients trained on local private datasets (McMahan et al., 2017).

**Key Characteristics**:
- Data remains on client devices (privacy-preserving)
- Periodic model aggregation rather than continuous synchronization
- Handles Non-IID (non-independent and identically distributed) data

### 2.2 FedAvg Algorithm

The prototypical FL algorithm:
1. Server distributes global model to clients
2. Clients train locally on private data
3. Clients send model updates to server
4. Server aggregates updates (weighted average)
5. Process repeats

### 2.3 Extensions and Variants

| Algorithm | Key Innovation | Use Case |
|-----------|---------------|----------|
| FedProx | Proximal term for heterogeneous data | Non-IID data distributions |
| FedDistill | Knowledge distillation from soft predictions | Communication reduction |
| Local SGD | Multiple local steps before synchronization | Bandwidth-constrained settings |
| DiLoCo | Inner/outer optimizer with infrequent sync | Cross-datacenter training |

### 2.4 DiLoCo: Distributed Low-Communication Training

**Key Innovation**: Dual optimization architecture with infrequent synchronization

**Algorithm**:
- **Inner Optimizer** (AdamW): Updates local weights every step
- **Outer Optimizer** (SGD with Nesterov Momentum): Synchronizes across workers every H steps

**Benefits**:
- 500x reduction in communication vs. standard data parallelism
- Robust to data heterogeneity
- Handles dynamic node participation

**Trade-offs**:
- Performance degradation as worker count increases (~1.5x compute penalty at 8 nodes)
- Requires careful tuning of inner step count (H)

---

## 3. BOINC/SETI@home Architecture

### 3.1 Overview

**BOINC (Berkeley Open Infrastructure for Network Computing)** is the dominant middleware for volunteer computing, originally developed for SETI@home (2002).

**Key Statistics**:
- First BOINC project launched: 2004
- Supports: Windows, Mac, Linux, Android
- Folding@home peak: 280,000 GPUs + 4.8M CPUs (first to cross 10^18 FLOPS)

### 3.2 Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                      BOINC SERVER                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Database   │  │  Scheduler  │  │  Work Generator     │  │
│  │  (MySQL)    │  │  (CGI/FCGI) │  │  (Daemon)           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Feeder    │  │  Validator  │  │  File Server        │  │
│  │  (Cache)    │  │  (Redundancy)│  │  (Input/Output)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP RPC
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     BOINC CLIENT                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Core Client │  │    GUI      │  │  Screensaver (opt)  │  │
│  │ (Job Mgmt)  │  │  (Manager)  │  │  (Visualization)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Key Design Principles

1. **Job Distribution**: Server divides work into independent units
2. **Redundant Computing**: Each unit sent to 2+ volunteers for validation
3. **Result Validation**: Compares outputs from multiple volunteers
4. **Credit System**: Incentivizes participation via leaderboards/badges
5. **Fault Tolerance**: Handles sporadically available, untrusted nodes

### 3.4 Server Architecture Details

**Scheduler**:
- Handles client RPCs via CGI/FCGI
- Uses shared-memory cache for job dispatch (avoids DB queries)
- Feeder process replenishes cache from database
- Can dispatch hundreds of jobs/second

**Database**:
- Central MySQL/MariaDB instance
- Stores: volunteers, hosts, apps, jobs, job instances
- XML "blobs" for job details to reduce table count

### 3.5 Client Architecture

**Core Client**:
- Manages job execution and file transfers
- Runs at lowest process priority
- Memory footprint limits to prevent paging
- Mobile: runs only when plugged in + charged + WiFi

**Security Model**:
- Sandboxed execution environment
- Applications run with restricted privileges
- Result validation catches malicious/incorrect outputs

---

## 4. ML Distributed Computing Projects

### 4.1 Folding@home

**Purpose**: Protein folding simulation for disease research

**Peak Capacity**:
- 280,000 GPUs + 4.8M CPUs
- First to exceed 1 exaFLOPS (10^18)
- GPUs contributed ~94% of total FLOPS

**Relevance to AI**:
- Demonstrates scale of volunteer compute possible
- Different workload characteristics than AI training
- Raw FLOPS counts don't directly translate to AI context

### 4.2 Hivemind / Learning@home

**Purpose**: PyTorch library for decentralized deep learning

**Key Features**:
- Distributed training without master node (DHT-based peer discovery)
- Fault-tolerant backpropagation
- Decentralized parameter averaging
- Decentralized Mixture-of-Experts (DeMoE)

**Projects Using Hivemind**:
- **Petals**: Collaborative inference/fine-tuning of 100B+ models
- **Training Transformers Together**: NeurIPS 2021 demo
- **CALM**: Arabic language model
- **sahajBERT**: Bengali ALBERT model

### 4.3 Petals

**Purpose**: BitTorrent-style LLM inference and fine-tuning

**Architecture**:
- Servers host subsets of model layers (Transformer blocks)
- Clients form pipeline-parallel chains of servers
- Supports parameter-efficient fine-tuning (adapters, prompt tuning)
- DHT-based peer discovery and coordination

**Optimizations**:
- Dynamic quantization
- Prioritizes low-latency connections
- Load balancing between servers

**Models Supported**: Llama 3.1 (405B), Mixtral (8x22B), Falcon (40B+), BLOOM (176B)

### 4.4 Prime Intellect

**Projects**:

#### INTELLECT-1 (Oct 2024)
- First 10B-parameter model trained decentralized
- 42 days across 3 continents, 5 countries
- 83% compute utilization, 96% US-only communication
- Used PRIME framework with DiLoCo + int8 all-reduce

#### INTELLECT-2 (Apr 2025)
- 32B parameter reasoning model
- Decentralized RL training on QwQ-32B
- PRIME-RL: Async RL with 3 independent stages
- SHARDCAST: P2P model weight distribution
- Crypto-economic staking for participation

**Key Innovations**:
- OpenDiLoCo: Open-source DiLoCo implementation
- ElasticDeviceMesh: Dynamic node participation
- GENESYS: Synthetic reasoning task generation
- TOPLOC: Proof-of-inference verification

### 4.5 Nous Research

**Projects**:
- **Psyche Consilience**: 40B-parameter decentralized pre-training
- **DeMo optimizer**: Communication-efficient single-step optimizer
- Uses DeMo (Decoupled Momentum) for gradient compression

### 4.6 Pluralis Research

**Approach**: Protocol Models with constrained activation subspaces

**Achievements**:
- 8B Llama-like model trained decentralized
- 7.5B model through volunteer network
- 100x activation size reduction via subspace constraints
- Decentralized context parallelism experiments

---

## 5. Communication Patterns for Distributed Agent Swarms

### 5.1 AllReduce Patterns

#### Ring AllReduce
```
Phase 1: Scatter-Reduce
- GPUs arranged in ring topology
- Each partitions gradients into N chunks
- Send to right neighbor, receive from left
- After N-1 rounds: each GPU has one complete fused chunk

Phase 2: AllGather
- Each GPU transmits its fused chunk to right neighbor
- After N-1 rounds: all GPUs have complete fused gradients
```

**Advantages**:
- Communication volume independent of GPU count
- Bandwidth-optimal for large messages
- Overlaps communication with computation

### 5.2 Gossip-Based Communication

**Used in**: Hivemind, Moshpit SGD

**Pattern**:
- Dynamic peer sampling via gossip protocol
- Asynchronous, best-effort RPC
- Eventually consistent parameter averaging
- No global synchronization required

**Trade-offs**:
- Converges eventually rather than bit-exactly
- Handles churn well but sacrifices determinism

### 5.3 Multi-Agent Swarm Protocols

#### Emerging Standards

| Protocol | Purpose | Key Features |
|----------|---------|--------------|
| **MCP** (Model Context Protocol) | Tool integration | Dynamic tool discovery, multi-server connections |
| **A2A** (Agent-to-Agent) | Inter-agent communication | Task negotiation, conflict resolution, knowledge transfer |
| **AOP** (Agent Orchestration Protocol) | Distributed deployment | Agent discovery, service orchestration |

#### SwarmSys Patterns
- **Profile Adaptation**: Competence embeddings drift ~0.14 cosine shift/round
- **Interaction Topology**: Hub-spoke → small-world transition
- **Pheromone-Based Optimization**: Reinforcement learning for coordination
- **Contribution Balance**: Entropy-based fairness (Hc=0.72)

### 5.4 Consensus Protocols for Swarms

| Protocol | Convergence | Fault Tolerance | Use Case |
|----------|-------------|-----------------|----------|
| CBBA (Consensus-Based Bundle Algorithm) | 12 iterations | High | Task allocation |
| Distance-Based Consensus | 18 iterations | 98% success (15% packet loss) | Formation maintenance |
| Gossip Protocol | 25-30 iterations | 90% (15% dropout) | Communication failure |

---

## 6. SwarmResearch v0.3 Architecture Patterns

### 6.1 Design Goals for Public Compute Sharing

1. **Accessibility**: Enable participation from consumer hardware
2. **Scalability**: Handle 1000s of heterogeneous nodes
3. **Fault Tolerance**: Graceful handling of node churn
4. **Security**: Verify computations, prevent attacks
5. **Incentives**: Reward meaningful contributions

### 6.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SWARMRESEARCH v0.3 NETWORK                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │   Bootstrap │    │   Model     │    │    Coordination         │  │
│  │   Servers   │◄──►│   Registry  │◄──►│    Layer (DHT)          │  │
│  │   (DHT)     │    │   (IPFS)    │    │    (Hivemind-style)     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘  │
│         ▲                                            ▲               │
│         │                                            │               │
│         └────────────────────────────────────────────┘               │
│                          P2P Gossip                                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     WORKER NODES (Volunteers)                    ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         ││
│  │  │  Node A  │  │  Node B  │  │  Node C  │  │  Node D  │  ...    ││
│  │  │ (GPU)    │  │ (CPU)    │  │ (GPU)    │  │ (Mobile) │         ││
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘         ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### 6.3 Core Components

#### 1. Bootstrap Layer
- DHT-based peer discovery (Kademlia-style)
- Initial connection points for new nodes
- Reputation tracking

#### 2. Model Registry
- IPFS for model weight storage
- Content-addressed, distributed
- Version control for model checkpoints

#### 3. Coordination Layer
- Task distribution and scheduling
- Gradient aggregation (DiLoCo-style)
- Verification and consensus

#### 4. Worker Nodes
- Lightweight client software
- Automatic hardware capability detection
- Sandboxed execution environment

### 6.4 Training Modes

#### Mode 1: Data Parallel with DiLoCo
```
For each outer step:
  For H inner steps:
    Each node trains on local data shard
    Inner optimizer (AdamW/Muon) updates local weights
  Aggregate pseudo-gradients across nodes
  Outer optimizer (SGD+Nesterov) updates global weights
```

**Configuration**:
- Inner steps (H): 100-500
- Compression: FP16/INT8/INT4 gradients
- Sync frequency: Every 500 steps = 500x bandwidth reduction

#### Mode 2: Pipeline Parallel (SWARM-style)
```
Model layers → Distributed across nodes
Each node holds 1/N of layers
Forward: Activations flow through pipeline
Backward: Gradients flow backward
Adaptive routing around slow/failed nodes
```

**Benefits**:
- Nodes don't need to fit full model
- Communication scales linearly with model dimension
- Larger models = better compute/communication ratio (square-cube law)

#### Mode 3: Hybrid (3D Decentralized)
```
Combine DP + PP + Compression:
- Data parallelism within geographic regions
- Pipeline parallelism across regions
- Gradient compression for cross-region sync
```

### 6.5 Security and Verification

| Threat | Mitigation |
|--------|------------|
| Byzantine nodes | Krum/Bulyan aggregation, median-based rules |
| Gradient attacks | Momentum-based verification, historical analysis |
| Sybil attacks | Proof-of-work, staking, reputation systems |
| Data poisoning | Differential privacy, robust aggregation |

**Verification Methods**:
1. **Redundant Computation**: Send to 2+ nodes, compare results
2. **Proof-of-Inference**: TOPLOC-style cryptographic proofs
3. **Gradient Statistics**: Monitor for anomalous updates
4. **Checkpoint Validation**: Periodic full validation on trusted nodes

---

## 7. Technical Requirements and Challenges

### 7.1 Network Requirements

| Training Mode | Bandwidth/Node | Latency Tolerance |
|--------------|----------------|-------------------|
| Centralized DP | 100-400 Gbps (NVLink) | <1ms |
| DiLoCo (500 inner) | 100-500 Mbps | <100ms |
| Pipeline Parallel | 1-10 Gbps | <10ms |
| Volunteer (compressed) | 10-100 Mbps | <500ms |

### 7.2 Hardware Requirements

**Minimum Node**:
- 8GB+ RAM
- GPU optional (CPU fallback)
- 10GB+ storage
- Stable internet connection

**Recommended Node**:
- 16GB+ VRAM GPU (RTX 3090/4090, A100)
- 32GB+ RAM
- NVMe SSD
- 100Mbps+ upload

### 7.3 Software Stack

| Component | Technology Options |
|-----------|-------------------|
| ML Framework | PyTorch, JAX |
| Distributed | Hivemind, DeepSpeed, Megatron |
| Communication | gRPC, libp2p, custom (PCCL) |
| Storage | IPFS, centralized CDN |
| Orchestration | Kubernetes, custom scheduler |
| Verification | Custom (TOPLOC-style) |

### 7.4 Key Challenges

#### 1. Communication Bottleneck
- **Problem**: Internet bandwidth << datacenter interconnect
- **Solutions**: 
  - DiLoCo (500x reduction)
  - Gradient compression (quantization, sparsification)
  - Pipeline parallelism

#### 2. Node Churn
- **Problem**: Volunteers join/leave unpredictably
- **Solutions**:
  - Elastic device mesh (PRIME)
  - Checkpoint frequency tuning
  - Redundant computation

#### 3. Heterogeneity
- **Problem**: Varying compute capabilities
- **Solutions**:
  - Adaptive batch sizing
  - Load balancing (SWARM)
  - Genetic algorithm grouping (Ravnest)

#### 4. Security
- **Problem**: Untrusted nodes may return bad results
- **Solutions**:
  - Byzantine-resilient aggregation
  - Redundant validation
  - Economic incentives + slashing

#### 5. Performance Gap
- **Problem**: Decentralized 1000x smaller than frontier
- **Current State**:
  - INTELLECT-1: 10B params, 6e22 FLOP
  - Frontier (Grok-4): ~6e25 FLOP
  - Gap: 1000x
- **Trend**: Decentralized growing 20x/year vs. 5x/year centralized

### 7.5 Compression Techniques Summary

| Technique | Compression | Overhead | Best For |
|-----------|-------------|----------|----------|
| Top-K Sparsification | 10-500x | Sorting | Conservative compression |
| DGC (Deep Gradient Compression) | 100-1000x | Momentum correction | High compression |
| QSGD (Quantized SGD) | 4-32x | Rounding error | General use |
| INT8 Quantization | 4x | Minimal | Default choice |
| INT4 Quantization | 8x | Some degradation | Bandwidth-constrained |
| Error Feedback | N/A | Storage | All lossy methods |

---

## 8. Recommendations for SwarmResearch v0.3

### 8.1 Phase 1: Foundation (Months 1-3)

1. **Implement DiLoCo-based training**
   - Start with 100-500M parameter models
   - 100-500 inner steps
   - FP16/INT8 compression

2. **Build bootstrap infrastructure**
   - DHT peer discovery
   - Basic reputation system
   - Model checkpointing to IPFS

3. **Develop lightweight client**
   - PyTorch-based
   - Auto hardware detection
   - Sandboxed execution

### 8.2 Phase 2: Scale (Months 4-6)

1. **Add pipeline parallelism option**
   - For models >2B parameters
   - Adaptive routing
   - Load balancing

2. **Implement verification**
   - Redundant computation
   - Gradient statistics monitoring
   - Checkpoint validation

3. **Launch incentive system**
   - Credit leaderboard
   - Contribution tracking
   - Token rewards (optional)

### 8.3 Phase 3: Production (Months 7-12)

1. **Support 10B+ parameter models**
2. **Implement Byzantine-resilient aggregation**
3. **Add mobile/CPU support**
4. **Build model hub for sharing fine-tuned adapters**

---

## 9. References and Further Reading

### Key Papers

1. **DiLoCo**: Douillard et al., "Distributed Low-Communication Training of Language Models" (2023)
2. **SWARM**: Ryabinin et al., "SWARM Parallelism" (2023)
3. **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
4. **BOINC**: Anderson, "BOINC: A Platform for Volunteer Computing" (2019)
5. **Petals**: Borzunov et al., "Collaborative Inference and Fine-tuning of Large Models" (2023)

### Projects to Study

- Hivemind: https://github.com/learning-at-home/hivemind
- Petals: https://github.com/bigscience-workshop/petals
- Prime Intellect: https://github.com/primeintellect
- OpenDiLoCo: https://github.com PrimeIntellect-ai/OpenDiLoCo

### Frameworks

- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- DeepSpeed: https://github.com/microsoft/DeepSpeed
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html

---

## 10. Conclusion

Distributed training over public compute is technically feasible today, as demonstrated by projects like INTELLECT-1, Petals, and SWARM. Key enablers include:

1. **DiLoCo-style algorithms** for 100-500x communication reduction
2. **Gradient compression** (quantization + sparsification)
3. **Pipeline parallelism** for large models
4. **Byzantine-resilient aggregation** for security
5. **Economic incentives** for sustained participation

The main challenge is the 1000x compute gap vs. frontier models. However, with decentralized training growing at 20x/year (vs. 5x/year for centralized), this gap could close within 5-6 years if trends continue.

SwarmResearch v0.3 should focus on:
- Building a robust, easy-to-use client
- Starting with smaller models (100M-1B)
- Implementing strong verification
- Creating meaningful incentives

The SETI@home model proves that volunteer computing can achieve massive scale. The challenge is adapting it for the more demanding requirements of AI training.
