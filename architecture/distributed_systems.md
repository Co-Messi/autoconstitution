# SwarmResearch v0.2/v0.3: Distributed Systems Architecture

## Executive Summary

SwarmResearch enables SETI@home-style public compute sharing for distributed ML training. This architecture leverages **DiLoCo (Distributed Low-Communication Training)** to achieve 500x communication reduction while maintaining synchronous training performance. The system is designed to tolerate heterogeneous nodes, intermittent connectivity, and Byzantine participants.

---

## 1. Node Discovery and Communication

### 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SWARM RESEARCH NETWORK                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   Bootstrap  │◄────┤   Bootstrap  │◄────┤   Bootstrap  │  DHT Ring       │
│  │   Node A     │────►│   Node B     │────►│   Node C     │  (Chord/Kademlia)│
│  └──────┬───────┘     └──────────────┘     └──────────────┘                 │
│         │                                                                    │
│         │         ┌─────────────────────────────────────────┐               │
│         └────────►│         Coordinator Pool                │               │
│                   │  ┌────────┐ ┌────────┐ ┌────────┐      │               │
│                   │  │Coord-1 │ │Coord-2 │ │Coord-N │ ...  │               │
│                   │  └───┬────┘ └───┬────┘ └───┬────┘      │               │
│                   └──────┼──────────┼──────────┼───────────┘               │
│                          │          │          │                           │
│    ┌─────────────────────┼──────────┼──────────┼─────────────────────┐     │
│    │                     │          │          │                     │     │
│    │  ┌─────────┐   ┌────▼───┐ ┌────▼───┐ ┌───▼────┐   ┌─────────┐  │     │
│    │  │ Worker  │   │ Worker │ │ Worker │ │ Worker │   │ Worker  │  │     │
│    │  │ Island 1│   │Island 2│ │Island 3│ │Island 4│   │Island N │  │     │
│    │  │ (Home)  │   │(Cloud) │ │(Home)  │ │(Cloud) │   │ (Edge)  │  │     │
│    │  └────┬────┘   └───┬────┘ └───┬────┘ └───┬────┘   └────┬────┘  │     │
│    │       │            │          │          │             │       │     │
│    │  ┌────▼────┐  ┌────▼────┐ ┌───▼─────┐ ┌──▼─────┐  ┌────▼────┐  │     │
│    │  │GPU: RTX │  │GPU: A100│ │GPU: RTX │ │GPU: T4 │  │GPU: CPU │  │     │
│    │  │4090 x4  │  │x8       │ │3090 x2  │ │x4      │  │(ARM)    │  │     │
│    │  └─────────┘  └─────────┘ └─────────┘ └────────┘  └─────────┘  │     │
│    │                                                                  │     │
│    └──────────────────────────────────────────────────────────────────┘     │
│                         HETEROGENEOUS WORKER POOL                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Discovery Protocol

#### Bootstrap Phase

```python
class BootstrapProtocol:
    """
    Kademlia DHT-based node discovery with hierarchical bootstrapping.
    """
    
    BOOTSTRAP_NODES = [
        "bootstrap.swarmresearch.org:6881",
        "bootstrap-backup.swarmresearch.org:6881",
        "bootstrap-eu.swarmresearch.org:6881"
    ]
    
    def __init__(self, node_id: bytes, listen_port: int):
        self.node_id = node_id  # 160-bit SHA-1 hash of public key
        self.routing_table = KBucketTable(k=20)  # Kademlia k-buckets
        self.protocol_version = "0.3.0"
        self.capabilities = self._announce_capabilities()
    
    def _announce_capabilities(self) -> Dict:
        """Announce node capabilities for optimal task matching."""
        return {
            "compute": {
                "gpu": detect_gpus(),  # CUDA, ROCm, Metal
                "cpu_cores": os.cpu_count(),
                "memory_gb": psutil.virtual_memory().total // (1024**3),
                "disk_gb": shutil.disk_usage("/").free // (1024**3)
            },
            "network": {
                "bandwidth_mbps": self._estimate_bandwidth(),
                "latency_ms": self._estimate_latency(),
                "public_ip": self._detect_public_ip()
            },
            "availability": {
                "uptime_hours": self._estimate_uptime(),
                "reliability_score": self._historical_reliability(),
                "max_task_duration": self._max_task_duration()
            },
            "security": {
                "public_key": self.public_key.hex(),
                "attestation": self._generate_attestation()
            }
        }
```

#### Kademlia DHT Integration

```python
class KademliaDiscovery:
    """
    Distributed hash table for peer discovery with geographic optimization.
    """
    
    ALPHA = 3  # Parallelism parameter
    K = 20     # Bucket size
    
    async def find_nodes(self, target_id: bytes, count: int = K) -> List[Node]:
        """
        Find k closest nodes to target using iterative lookup.
        """
        queried = set()
        closest = self.routing_table.find_closest(target_id, self.K)
        
        while True:
            # Select α nodes to query that we haven't asked yet
            to_query = [n for n in closest[:self.ALPHA] 
                       if n.id not in queried]
            
            if not to_query:
                break
            
            # Parallel queries
            tasks = [self._query_node(n, target_id) for n in to_query]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for node, result in zip(to_query, results):
                queried.add(node.id)
                if isinstance(result, list):
                    closest.extend(result)
            
            closest = sorted(set(closest), 
                           key=lambda n: xor_distance(n.id, target_id))[:self.K]
        
        return closest[:count]
```

### 1.3 Communication Layer

#### Protocol Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│         (Task Assignment, Result Submission)                 │
├─────────────────────────────────────────────────────────────┤
│                    Message Layer (gRPC/QUIC)                 │
│         (Protobuf, Compressed, Encrypted)                    │
├─────────────────────────────────────────────────────────────┤
│                    Security Layer (TLS 1.3 + Noise)          │
│         (End-to-end encryption, Perfect forward secrecy)     │
├─────────────────────────────────────────────────────────────┤
│                    Transport Layer (QUIC/UDP)                │
│         (Multiplexed streams, NAT traversal)                 │
├─────────────────────────────────────────────────────────────┤
│                    Network Layer                             │
│         (IPv4/IPv6, STUN/TURN for NAT traversal)             │
└─────────────────────────────────────────────────────────────┘
```

#### QUIC-Based Communication

```python
class SwarmQUICProtocol:
    """
    QUIC-based protocol for low-latency, reliable communication.
    Supports connection migration for mobile/spotty connections.
    """
    
    def __init__(self, cert_manager: CertificateManager):
        self.cert_manager = cert_manager
        self.connections: Dict[NodeId, QuicConnection] = {}
        self.streams: Dict[str, StreamHandler] = {}
    
    async def create_connection(self, node: Node) -> QuicConnection:
        """Create encrypted QUIC connection with certificate pinning."""
        configuration = QuicConfiguration(
            alpn_protocols=["swarm-research/0.3"],
            is_client=True,
            max_datagram_frame_size=65536
        )
        
        # Load or verify node certificate
        node_cert = await self.cert_manager.get_certificate(node.id)
        configuration.load_verify_locations(cadata=node_cert)
        
        connection = await connect(
            node.address,
            configuration=configuration,
            create_protocol=SwarmQuicProtocol,
        )
        
        return connection
    
    async def send_message(self, 
                          connection: QuicConnection, 
                          message: SwarmMessage,
                          priority: MessagePriority = MessagePriority.NORMAL) -> None:
        """
        Send message with appropriate stream configuration.
        """
        # Control messages: reliable, ordered stream
        # Gradient updates: reliable but can tolerate delay
        # Heartbeats: unreliable, low priority
        
        stream_id = self._allocate_stream(connection, priority)
        data = self._serialize_message(message)
        
        # Apply compression based on message type
        if message.type == MessageType.GRADIENT_UPDATE:
            data = await self._compress_gradients(data)
        
        await connection.send_stream_data(stream_id, data, end_stream=True)
```

#### NAT Traversal

```python
class NATTraversal:
    """
    ICE-like NAT traversal for home workers behind routers.
    """
    
    STUN_SERVERS = [
        "stun.l.google.com:19302",
        "stun1.l.google.com:19302",
        "stun.swarmresearch.org:3478"
    ]
    
    TURN_SERVERS = [
        "turn.swarmresearch.org:3478"  # For symmetric NAT fallback
    ]
    
    async def establish_direct_connection(self, peer: Node) -> Connection:
        """
        Attempt direct connection using ICE-like process.
        """
        # Gather local candidates
        local_candidates = await self._gather_candidates()
        
        # Exchange candidates via signaling server
        peer_candidates = await self._signal_exchange(peer, local_candidates)
        
        # Try direct connection (host, srflx candidates)
        for local in local_candidates:
            for remote in peer_candidates:
                if self._can_pair(local, remote):
                    conn = await self._try_connect(local, remote)
                    if conn:
                        return conn
        
        # Fallback to TURN relay
        return await self._create_relay_connection(peer)
```

---

## 2. Task Distribution

### 2.1 Hierarchical Task Assignment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TASK DISTRIBUTION HIERARCHY                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     GLOBAL COORDINATOR                               │   │
│  │  - Maintains global model checkpoint                                 │   │
│  │  - Assigns outer optimization rounds                                 │   │
│  │  - Validates aggregated updates                                      │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                           │
│                    ┌─────────────┼─────────────┐                            │
│                    ▼             ▼             ▼                            │
│  ┌─────────────────────┐ ┌──────────────┐ ┌──────────────┐                 │
│  │   REGIONAL POOL     │ │ REGIONAL POOL│ │ REGIONAL POOL│                 │
│  │   (North America)   │ │   (Europe)   │ │   (Asia)     │                 │
│  │                     │ │              │ │              │                 │
│  │  ┌───────────────┐  │ │ ┌──────────┐ │ │ ┌──────────┐ │                 │
│  │  │  Coordinator  │  │ │ │Coordinator│ │ │ │Coordinator│                 │
│  │  │  (US-East)    │  │ │ │(EU-West) │ │ │ │(AP-South)│ │                 │
│  │  └───────┬───────┘  │ │ └────┬─────┘ │ │ └────┬─────┘ │                 │
│  │          │          │ │      │       │ │      │       │                 │
│  │  ┌───────┴───────┐  │ │ ┌────┴────┐  │ │ ┌────┴────┐  │                 │
│  │  │  Worker Pool  │  │ │ │Worker   │  │ │ │Worker   │  │                 │
│  │  │  - 8x RTX4090 │  │ │ │Pool     │  │ │ │Pool     │  │                 │
│  │  │  - 4x A100    │  │ │ │- Mixed  │  │ │ │- Mixed  │  │                 │
│  │  │  - 12x RTX3090│  │ │ │ GPUs    │  │ │ │ GPUs    │  │                 │
│  │  └───────────────┘  │ │ └─────────┘  │ │ └─────────┘  │                 │
│  └─────────────────────┘ └──────────────┘ └──────────────┘                 │
│                                                                              │
│  OUTER OPTIMIZATION ROUND (H ~ 500-2000 inner steps)                       │
│  └── Each worker performs H local steps before synchronization              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Task Scheduler

```python
class TaskScheduler:
    """
    Intelligent task distribution considering worker capabilities and load.
    """
    
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
        self.worker_pool = WorkerPool()
        self.task_queue = PriorityQueue()
        self.metrics = MetricsCollector()
    
    async def schedule_outer_round(self, 
                                   global_model: ModelState,
                                   h_steps: int) -> RoundResult:
        """
        Schedule an outer optimization round across all available workers.
        DiLoCo: Workers train independently for H steps, then synchronize.
        """
        # Get eligible workers
        workers = await self.worker_pool.get_available_workers(
            min_compute_score=0.5,
            min_reliability=0.8
        )
        
        # Assign tasks based on capabilities
        assignments = self._assign_tasks(workers, global_model, h_steps)
        
        # Distribute model checkpoints
        await self._distribute_models(assignments, global_model)
        
        # Monitor progress with timeout handling
        results = await self._collect_results(assignments, timeout=h_steps * 2)
        
        # Aggregate using DiLoCo outer optimizer
        aggregated = self._aggregate_results(results)
        
        return aggregated
    
    def _assign_tasks(self, 
                     workers: List[Worker], 
                     model: ModelState,
                     h_steps: int) -> Dict[WorkerId, TaskAssignment]:
        """
        Assign tasks to workers based on their capabilities.
        """
        assignments = {}
        
        for worker in workers:
            # Calculate optimal batch size based on GPU memory
            batch_size = self._compute_batch_size(worker.gpu_memory)
            
            # Adjust H based on worker reliability
            # Less reliable workers get smaller H to checkpoint more often
            adjusted_h = int(h_steps * worker.reliability_score)
            
            # Select data shard based on worker location and data distribution
            data_shard = self._select_data_shard(worker, model)
            
            assignments[worker.id] = TaskAssignment(
                worker_id=worker.id,
                model_checkpoint=model.checkpoint_id,
                inner_steps=adjusted_h,
                batch_size=batch_size,
                learning_rate=self._adaptive_lr(worker),
                data_shard=data_shard,
                deadline=time.time() + adjusted_h * self._estimate_step_time(worker)
            )
        
        return assignments
```

### 2.3 Adaptive Load Balancing

```python
class AdaptiveLoadBalancer:
    """
    Dynamic load balancing based on real-time worker performance.
    """
    
    def __init__(self):
        self.worker_performance: Dict[WorkerId, PerformanceHistory] = {}
        self.rebalancing_threshold = 0.2  # 20% performance variance triggers rebalance
    
    async def monitor_and_rebalance(self, active_round: TrainingRound):
        """
        Monitor worker progress and rebalance if significant variance detected.
        """
        while active_round.in_progress:
            progress = await self._collect_progress(active_round)
            
            # Calculate performance variance
            speeds = [p.steps_per_second for p in progress.values()]
            variance = statistics.variance(speeds) / statistics.mean(speeds)
            
            if variance > self.rebalancing_threshold:
                # Identify stragglers and fast workers
                median_speed = statistics.median(speeds)
                stragglers = [w for w, p in progress.items() 
                             if p.steps_per_second < median_speed * 0.7]
                fast_workers = [w for w, p in progress.items() 
                               if p.steps_per_second > median_speed * 1.3]
                
                # Rebalance by offloading from stragglers
                if stragglers and fast_workers:
                    await self._rebalance_tasks(stragglers, fast_workers)
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _rebalance_tasks(self, 
                               stragglers: List[WorkerId],
                               fast_workers: List[WorkerId]):
        """
        Migrate pending tasks from slow workers to fast workers.
        """
        for straggler_id in stragglers:
            remaining_task = await self._extract_remaining_task(straggler_id)
            
            # Find best target worker
            target = min(fast_workers, 
                        key=lambda w: self.worker_performance[w].current_load)
            
            # Checkpoint and migrate
            checkpoint = await self._create_migration_checkpoint(straggler_id)
            await self._assign_to_worker(target, remaining_task, checkpoint)
```

---

## 3. Result Aggregation with DiLoCo

### 3.1 DiLoCo Algorithm Implementation

```python
class DiLoCoAggregator:
    """
    Distributed Low-Communication training aggregation.
    Based on: https://arxiv.org/abs/2311.08105
    
    Key insight: Workers perform H inner steps independently,
    then communicate only the outer gradient (parameter delta).
    This achieves 500x communication reduction vs. synchronous SGD.
    """
    
    def __init__(self, 
                 inner_optimizer: Type[Optimizer] = AdamW,
                 outer_optimizer: Type[Optimizer] = NesterovSGD,
                 h_steps: int = 500):
        self.inner_optimizer = inner_optimizer
        self.outer_optimizer = outer_optimizer
        self.h_steps = h_steps
        self.shared_params: Optional[TensorDict] = None
        
    async def outer_optimization_step(self, 
                                      workers: List[Worker],
                                      global_step: int) -> ModelState:
        """
        Execute one DiLoCo outer optimization step.
        
        Algorithm:
        1. Each worker i performs H inner steps: θ_i → θ_i^(H)
        2. Compute outer gradient: Δ_i = θ_i^(H) - θ^(0)
        3. Aggregate outer gradients: Δ = average(Δ_i)
        4. Update shared params: θ^(new) = OuterOpt(θ^(0), Δ)
        """
        # Phase 1: Distribute current shared parameters
        await self._broadcast_shared_params(workers)
        
        # Phase 2: Workers perform H independent inner steps
        inner_tasks = [
            self._run_inner_optimization(worker, self.h_steps)
            for worker in workers
        ]
        inner_results = await asyncio.gather(*inner_tasks, return_exceptions=True)
        
        # Phase 3: Collect outer gradients (parameter deltas)
        outer_gradients = []
        valid_workers = []
        
        for worker, result in zip(workers, inner_results):
            if isinstance(result, Exception):
                logger.warning(f"Worker {worker.id} failed: {result}")
                continue
            
            # Compute outer gradient: Δ = θ_i^(H) - θ^(0)
            delta = self._compute_parameter_delta(
                result.final_params, 
                self.shared_params
            )
            
            # Apply gradient compression
            compressed_delta = await self._compress_gradient(delta, worker)
            
            outer_gradients.append(compressed_delta)
            valid_workers.append(worker)
        
        # Phase 4: Byzantine-robust aggregation
        aggregated_delta = await self._robust_aggregate(
            outer_gradients, 
            valid_workers
        )
        
        # Phase 5: Apply outer optimizer
        self.shared_params = self.outer_optimizer.step(
            params=self.shared_params,
            gradient=aggregated_delta,
            momentum=self._get_outer_momentum(global_step)
        )
        
        return ModelState(
            params=self.shared_params,
            outer_step=global_step,
            inner_steps=global_step * self.h_steps,
            participating_workers=len(valid_workers)
        )
    
    def _compute_parameter_delta(self, 
                                  final_params: TensorDict,
                                  initial_params: TensorDict) -> TensorDict:
        """
        Compute outer gradient as parameter difference.
        This is more stable than accumulating gradients over H steps.
        """
        delta = {}
        for name in final_params.keys():
            delta[name] = final_params[name] - initial_params[name]
        return delta
```

### 3.2 Gradient Compression

```python
class GradientCompressor:
    """
    Multi-tier gradient compression for bandwidth-constrained workers.
    Combines multiple techniques for maximum compression.
    """
    
    def __init__(self, 
                 quantization_bits: int = 8,
                 sparsity_ratio: float = 0.01,
                 error_feedback: bool = True):
        self.quantization_bits = quantization_bits
        self.sparsity_ratio = sparsity_ratio
        self.error_feedback = error_feedback
        self.residual_errors: Dict[WorkerId, TensorDict] = {}
    
    async def compress(self, 
                      gradient: TensorDict,
                      worker_id: WorkerId,
                      target_compression: float = 100.0) -> CompressedGradient:
        """
        Compress gradient using multiple techniques.
        
        Pipeline:
        1. Error feedback: Add residual from previous round
        2. Sparsification: Keep only top-k% by magnitude
        3. Quantization: Reduce precision to 8-bit or 4-bit
        4. Encoding: Apply entropy coding
        """
        # Step 1: Error feedback
        if self.error_feedback and worker_id in self.residual_errors:
            gradient = self._add_residual(gradient, self.residual_errors[worker_id])
        
        # Step 2: Sparsification (Top-K)
        sparse_grad, mask = self._top_k_sparsify(gradient, self.sparsity_ratio)
        
        # Step 3: Quantization
        quantized = self._quantize(sparse_grad, bits=self.quantization_bits)
        
        # Step 4: Entropy encoding
        encoded = self._entropy_encode(quantized)
        
        # Compute and store residual for next round
        if self.error_feedback:
            reconstructed = self._decompress(encoded, mask)
            self.residual_errors[worker_id] = self._compute_residual(
                gradient, reconstructed
            )
        
        return CompressedGradient(
            data=encoded,
            mask=mask,
            metadata=CompressionMetadata(
                original_size=self._compute_size(gradient),
                compressed_size=len(encoded),
                compression_ratio=self._compute_size(gradient) / len(encoded),
                quantization_bits=self.quantization_bits,
                sparsity=self.sparsity_ratio
            )
        )
    
    def _top_k_sparsify(self, 
                       gradient: TensorDict, 
                       sparsity: float) -> Tuple[TensorDict, TensorDict]:
        """
        Keep only top-k% of values by magnitude.
        """
        # Flatten all gradients
        flat_grads = torch.cat([g.flatten() for g in gradient.values()])
        
        # Find threshold for top-k%
        k = int(len(flat_grads) * sparsity)
        threshold = torch.kthvalue(torch.abs(flat_grads), len(flat_grads) - k)[0]
        
        # Create mask and sparse gradient
        mask = {}
        sparse = {}
        for name, grad in gradient.items():
            mask[name] = torch.abs(grad) >= threshold
            sparse[name] = grad * mask[name]
        
        return sparse, mask
    
    def _quantize(self, 
                  gradient: TensorDict, 
                  bits: int = 8) -> TensorDict:
        """
        Linear quantization to specified bit width.
        """
        quantized = {}
        for name, grad in gradient.items():
            # Find min/max for this tensor
            min_val = grad.min()
            max_val = grad.max()
            
            # Quantize to [0, 2^bits - 1]
            scale = (max_val - min_val) / (2**bits - 1)
            quantized[name] = torch.round((grad - min_val) / scale).to(torch.uint8)
            quantized[name].metadata = {
                'scale': scale.item(),
                'min': min_val.item(),
                'bits': bits
            }
        
        return quantized
```

### 3.3 Streaming DiLoCo for Continuous Updates

```python
class StreamingDiLoCo(DiLoCoAggregator):
    """
    Streaming variant that overlaps communication with computation.
    Based on: https://arxiv.org/abs/2502.12996
    
    Instead of blocking at outer step, gradients are streamed
    continuously in the background.
    """
    
    def __init__(self, *args, stream_buffer_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_buffer_size = stream_buffer_size
        self.gradient_streams: Dict[WorkerId, asyncio.Queue] = {}
        self.accumulated_deltas: Dict[WorkerId, List[TensorDict]] = {}
    
    async def eager_update_loop(self, workers: List[Worker]):
        """
        Continuous streaming of partial updates.
        """
        # Start background streaming tasks
        stream_tasks = [
            asyncio.create_task(self._stream_worker_updates(worker))
            for worker in workers
        ]
        
        # Continuously apply updates as they arrive
        while True:
            for worker_id, stream in self.gradient_streams.items():
                if not stream.empty():
                    partial_delta = await stream.get()
                    
                    # Apply partial update immediately (eager)
                    self.shared_params = self._apply_partial_update(
                        self.shared_params,
                        partial_delta,
                        weight=1.0 / len(workers)
                    )
            
            await asyncio.sleep(0.001)  # 1ms polling
    
    async def _stream_worker_updates(self, worker: Worker):
        """
        Stream partial deltas from a worker during its inner optimization.
        """
        buffer = []
        
        async for partial_delta in worker.stream_partial_deltas():
            buffer.append(partial_delta)
            
            # Send when buffer is full
            if len(buffer) >= self.stream_buffer_size:
                aggregated_partial = self._aggregate_buffer(buffer)
                await self.gradient_streams[worker.id].put(aggregated_partial)
                buffer = []
```

---

## 4. Fault Tolerance

### 4.1 Failure Detection

```python
class FailureDetector:
    """
    Phi-accrual failure detector for adaptive suspicion thresholds.
    """
    
    def __init__(self, 
                 threshold: float = 8.0,
                 min_std_dev: float = 0.5,
                 max_sample_size: int = 1000):
        self.threshold = threshold
        self.min_std_dev = min_std_dev
        self.max_sample_size = max_sample_size
        self.heartbeats: Dict[NodeId, List[float]] = {}
        self.last_heartbeat: Dict[NodeId, float] = {}
    
    def heartbeat(self, node_id: NodeId):
        """Record heartbeat arrival."""
        now = time.time()
        
        if node_id in self.last_heartbeat:
            interval = now - self.last_heartbeat[node_id]
            
            if node_id not in self.heartbeats:
                self.heartbeats[node_id] = []
            
            self.heartbeats[node_id].append(interval)
            
            # Keep only recent samples
            if len(self.heartbeats[node_id]) > self.max_sample_size:
                self.heartbeats[node_id] = self.heartbeats[node_id][-self.max_sample_size:]
        
        self.last_heartbeat[node_id] = now
    
    def suspicion_level(self, node_id: NodeId) -> float:
        """
        Calculate suspicion level using phi-accrual.
        Returns probability that node has failed.
        """
        if node_id not in self.heartbeats:
            return 0.0 if node_id not in self.last_heartbeat else 1.0
        
        intervals = self.heartbeats[node_id]
        if len(intervals) < 2:
            return 0.0
        
        # Calculate mean and standard deviation
        mean = statistics.mean(intervals)
        std_dev = max(statistics.stdev(intervals), self.min_std_dev)
        
        # Time since last heartbeat
        time_since = time.time() - self.last_heartbeat.get(node_id, 0)
        
        # Phi calculation
        phi = -math.log10(math.exp(-self._phi_factor(time_since, mean, std_dev)))
        
        return min(phi / self.threshold, 1.0)
    
    def _phi_factor(self, t: float, mean: float, std_dev: float) -> float:
        """Calculate the phi factor for the distribution."""
        return 0.5 * ((t - mean) / std_dev) ** 2
```

### 4.2 Checkpoint and Recovery

```python
class CheckpointManager:
    """
    Distributed checkpointing with redundancy.
    """
    
    def __init__(self, 
                 storage_backends: List[StorageBackend],
                 replication_factor: int = 3):
        self.storage_backends = storage_backends
        self.replication_factor = replication_factor
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
    
    async def save_checkpoint(self, 
                             model_state: ModelState,
                             checkpoint_id: str,
                             priority: CheckpointPriority = CheckpointPriority.NORMAL):
        """
        Save checkpoint to multiple storage backends.
        """
        # Serialize model state
        serialized = self._serialize(model_state)
        
        # Select backends based on priority
        if priority == CheckpointPriority.CRITICAL:
            backends = self.storage_backends  # All backends
        else:
            backends = random.sample(
                self.storage_backends, 
                min(self.replication_factor, len(self.storage_backends))
            )
        
        # Parallel write to selected backends
        write_tasks = [
            self._write_to_backend(backend, checkpoint_id, serialized)
            for backend in backends
        ]
        results = await asyncio.gather(*write_tasks, return_exceptions=True)
        
        # Verify replication
        successful = sum(1 for r in results if not isinstance(r, Exception))
        if successful < self.replication_factor // 2 + 1:
            raise CheckpointError(f"Insufficient replication: {successful}/{self.replication_factor}")
        
        # Record metadata
        self.checkpoints[checkpoint_id] = CheckpointMetadata(
            id=checkpoint_id,
            timestamp=time.time(),
            model_version=model_state.version,
            storage_locations=[b.name for b, r in zip(backends, results) 
                              if not isinstance(r, Exception)],
            size_bytes=len(serialized)
        )
        
        return self.checkpoints[checkpoint_id]
    
    async def load_checkpoint(self, checkpoint_id: str) -> ModelState:
        """
        Load checkpoint with automatic failover.
        """
        metadata = self.checkpoints.get(checkpoint_id)
        if not metadata:
            raise CheckpointError(f"Unknown checkpoint: {checkpoint_id}")
        
        # Try locations in order of preference
        for location in metadata.storage_locations:
            backend = self._get_backend(location)
            try:
                data = await backend.read(checkpoint_id)
                return self._deserialize(data)
            except Exception as e:
                logger.warning(f"Failed to load from {location}: {e}")
                continue
        
        raise CheckpointError(f"All locations failed for checkpoint: {checkpoint_id}")
```

### 4.3 Straggler Mitigation

```python
class StragglerMitigation:
    """
    Handle slow workers without blocking the entire round.
    """
    
    def __init__(self, 
                 timeout_factor: float = 1.5,
                 min_participation: float = 0.6):
        self.timeout_factor = timeout_factor
        self.min_participation = min_participation
    
    async def collect_with_timeout(self,
                                   workers: List[Worker],
                                   expected_duration: float) -> Dict[WorkerId, Result]:
        """
        Collect results with adaptive timeout.
        """
        timeout = expected_duration * self.timeout_factor
        results = {}
        pending = set(w.id for w in workers)
        
        # Start collection
        start_time = time.time()
        
        while pending and (time.time() - start_time) < timeout:
            for worker_id in list(pending):
                if await self._result_ready(worker_id):
                    results[worker_id] = await self._get_result(worker_id)
                    pending.remove(worker_id)
            
            # Check if we have enough results
            participation = len(results) / len(workers)
            if participation >= self.min_participation:
                logger.info(f"Sufficient participation reached: {participation:.2%}")
                break
            
            await asyncio.sleep(1)
        
        # Handle stragglers
        for worker_id in pending:
            logger.warning(f"Worker {worker_id} timed out, checkpointing partial progress")
            await self._checkpoint_partial(worker_id)
        
        return results
```

---

## 5. Security Model

### 5.1 Threat Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THREAT MODEL                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BYZANTINE ATTACKS (Arbitrary behavior from compromised nodes):             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. GRADIENT INVERSION ATTACKS                                       │   │
│  │    - Send negated gradients to drive model in wrong direction       │   │
│  │    - Detection: Statistical outlier detection in gradient space     │   │
│  │                                                                     │   │
│  │ 2. MODEL POISONING ATTACKS                                          │   │
│  │    - Send crafted gradients to implant backdoors                    │   │
│  │    - Detection: Consistency checking with virtual data samples      │   │
│  │                                                                     │   │
│  │ 3. SYBIL ATTACKS                                                    │   │
│  │    - Create many fake identities to gain voting power               │   │
│  │    - Mitigation: Proof-of-work or stake-based identity              │   │
│  │                                                                     │   │
│  │ 4. FREE-RIDER ATTACKS                                               │   │
│  │    - Claim rewards without contributing computation                 │   │
│  │    - Detection: Verifiable computation, sampling verification       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  NETWORK ATTACKS:                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ - Man-in-the-middle: TLS 1.3 + certificate pinning                  │   │
│  │ - DDoS: Rate limiting, proof-of-work for connection establishment   │   │
│  │ - Eclipse attacks: DHT hardening with random lookups                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ASSUMPTIONS:                                                               │
│  - At most f < n/3 Byzantine workers in any round                         │
│  - Coordinators are trusted (v0.2), decentralized in v0.3                 │
│  - Network is asynchronous but with bounded delay                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Byzantine-Robust Aggregation

```python
class ByzantineRobustAggregator:
    """
    Robust aggregation tolerating up to f < n/3 Byzantine workers.
    Combines multiple defense mechanisms.
    """
    
    def __init__(self, 
                 max_byzantine_ratio: float = 0.33,
                 aggregation_rule: str = "multi-krum"):
        self.max_byzantine_ratio = max_byzantine_ratio
        self.aggregation_rule = aggregation_rule
    
    async def aggregate(self, 
                       gradients: List[TensorDict],
                       worker_ids: List[WorkerId]) -> TensorDict:
        """
        Byzantine-robust aggregation using Multi-Krum.
        
        Multi-Krum selects m gradients that are closest to their neighbors,
        then averages them. This filters out outliers from Byzantine workers.
        """
        n = len(gradients)
        f = int(n * self.max_byzantine_ratio)
        m = n - f - 2  # Number of gradients to select
        
        if self.aggregation_rule == "multi-krum":
            return await self._multi_krum(gradients, m, f)
        elif self.aggregation_rule == "trimmed-mean":
            return await self._trimmed_mean(gradients, f)
        elif self.aggregation_rule == "median":
            return await self._coordinate_median(gradients)
        else:
            raise ValueError(f"Unknown aggregation rule: {self.aggregation_rule}")
    
    async def _multi_krum(self, 
                         gradients: List[TensorDict],
                         m: int,
                         f: int) -> TensorDict:
        """
        Multi-Krum: Select m gradients with smallest neighborhood distance.
        """
        # Flatten gradients for distance computation
        flat_grads = [self._flatten(g) for g in gradients]
        
        # Compute pairwise distances
        distances = torch.zeros(len(gradients), len(gradients))
        for i in range(len(gradients)):
            for j in range(i + 1, len(gradients)):
                dist = torch.norm(flat_grads[i] - flat_grads[j])
                distances[i, j] = distances[j, i] = dist
        
        # For each gradient, find sum of distances to n-f-2 closest neighbors
        scores = []
        for i in range(len(gradients)):
            sorted_dists = torch.sort(distances[i])[0]
            # Sum distances to n-f-2 closest (excluding self)
            score = sorted_dists[1:n-f-1].sum()
            scores.append((score.item(), i))
        
        # Select m gradients with lowest scores
        scores.sort()
        selected_indices = [idx for _, idx in scores[:m]]
        
        # Average selected gradients
        selected = [gradients[i] for i in selected_indices]
        return self._average_gradients(selected)
    
    async def _trimmed_mean(self, 
                           gradients: List[TensorDict],
                           f: int) -> TensorDict:
        """
        Trimmed mean: Remove f largest and f smallest values per coordinate.
        """
        result = {}
        for name in gradients[0].keys():
            stacked = torch.stack([g[name] for g in gradients])
            
            # Sort along worker dimension
            sorted_vals, _ = torch.sort(stacked, dim=0)
            
            # Trim f from each end and compute mean
            trimmed = sorted_vals[f:-f] if f > 0 else sorted_vals
            result[name] = trimmed.mean(dim=0)
        
        return result
```

### 5.3 Consistency Verification

```python
class ConsistencyVerifier:
    """
    Verify worker contributions using virtual data samples.
    Based on: https://arxiv.org/abs/2411.10212
    """
    
    def __init__(self, 
                 virtual_sample_count: int = 100,
                 consistency_threshold: float = 0.9):
        self.virtual_sample_count = virtual_sample_count
        self.consistency_threshold = consistency_threshold
        self.virtual_dataset = self._generate_virtual_dataset()
    
    def _generate_virtual_dataset(self) -> Dataset:
        """
        Generate synthetic data samples for consistency checking.
        These don't need to be realistic, just consistent across verifications.
        """
        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        
        virtual_samples = []
        for _ in range(self.virtual_sample_count):
            # Generate random but deterministic input
            sample = torch.randn(1, 3, 224, 224)  # Example: image
            label = torch.randint(0, 1000, (1,))   # Example: classification
            virtual_samples.append((sample, label))
        
        return virtual_samples
    
    async def verify_worker(self, 
                           worker: Worker,
                           gradient: TensorDict,
                           model_state: ModelState) -> float:
        """
        Compute consistency score for a worker's gradient.
        
        The worker should produce similar gradient direction on virtual data
        as it did on its real training data (for honest workers).
        """
        # Load worker's claimed model state
        worker_model = self._load_model(model_state)
        
        # Compute gradient on virtual samples
        virtual_gradient = await self._compute_virtual_gradient(
            worker_model, 
            self.virtual_dataset
        )
        
        # Compare direction similarity
        similarity = self._gradient_similarity(gradient, virtual_gradient)
        
        return similarity
    
    def _gradient_similarity(self, 
                            g1: TensorDict, 
                            g2: TensorDict) -> float:
        """
        Compute cosine similarity between two gradients.
        """
        flat1 = torch.cat([v.flatten() for v in g1.values()])
        flat2 = torch.cat([v.flatten() for v in g2.values()])
        
        return F.cosine_similarity(flat1.unsqueeze(0), 
                                   flat2.unsqueeze(0)).item()
    
    async def filter_byzantine_workers(self,
                                      workers: List[Worker],
                                      gradients: List[TensorDict],
                                      model_state: ModelState) -> Tuple[List[Worker], List[TensorDict]]:
        """
        Filter out workers with low consistency scores.
        """
        scores = []
        for worker, gradient in zip(workers, gradients):
            score = await self.verify_worker(worker, gradient, model_state)
            scores.append((score, worker, gradient))
        
        # Filter by threshold
        valid = [(w, g) for s, w, g in scores if s >= self.consistency_threshold]
        
        if len(valid) < len(workers) * 0.5:
            logger.warning("Too many workers failed consistency check!")
            # Fall back to median aggregation
            return workers, gradients
        
        return [w for w, _ in valid], [g for _, g in valid]
```

### 5.4 Identity and Reputation

```python
class ReputationSystem:
    """
    Track worker reputation for Sybil resistance and quality assurance.
    """
    
    def __init__(self):
        self.reputations: Dict[WorkerId, ReputationScore] = {}
        self.contribution_history: Dict[WorkerId, List[Contribution]] = {}
    
    def record_contribution(self, 
                           worker_id: WorkerId,
                           contribution: Contribution,
                           verification_result: VerificationResult):
        """
        Update worker reputation based on contribution quality.
        """
        if worker_id not in self.reputations:
            self.reputations[worker_id] = ReputationScore(
                worker_id=worker_id,
                total_contributions=0,
                verified_contributions=0,
                consistency_score=0.0,
                reliability_score=0.0,
                stake_amount=0.0
            )
        
        rep = self.reputations[worker_id]
        rep.total_contributions += 1
        
        if verification_result.passed:
            rep.verified_contributions += 1
            rep.consistency_score = 0.9 * rep.consistency_score + 0.1 * verification_result.consistency
        else:
            # Penalize failed verification
            rep.consistency_score *= 0.95
        
        # Update reliability based on on-time completion
        if contribution.completed_on_time:
            rep.reliability_score = 0.95 * rep.reliability_score + 0.05
        else:
            rep.reliability_score *= 0.98
        
        # Store contribution
        if worker_id not in self.contribution_history:
            self.contribution_history[worker_id] = []
        self.contribution_history[worker_id].append(contribution)
    
    def get_trust_score(self, worker_id: WorkerId) -> float:
        """
        Compute overall trust score for task assignment.
        """
        rep = self.reputations.get(worker_id)
        if not rep or rep.total_contributions < 5:
            return 0.5  # New workers start with neutral score
        
        # Weighted combination
        verification_rate = rep.verified_contributions / rep.total_contributions
        
        trust = (
            0.3 * verification_rate +
            0.3 * rep.consistency_score +
            0.2 * rep.reliability_score +
            0.2 * min(rep.stake_amount / 1000, 1.0)  # Stake bonus
        )
        
        return trust
```

---

## 6. System Integration

### 6.1 End-to-End Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SWARM RESEARCH TRAINING FLOW                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INITIALIZATION                                                              │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────────┐                      │
│  │  Init   │───►│   Bootstrap │───►│  Join DHT Ring  │                      │
│  │  Model  │    │   to Network│    │  Announce Caps  │                      │
│  └─────────┘    └─────────────┘    └─────────────────┘                      │
│       │                                                                     │
│       ▼                                                                     │
│  OUTER ROUND LOOP (T iterations)                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ┌──────────────┐    ┌──────────────────────────────────────────┐  │   │
│  │  │ Distribute   │───►│ Workers download model checkpoint        │  │   │
│  │  │ Global Model │    │ via P2P or CDN                           │  │   │
│  │  └──────────────┘    └──────────────────────────────────────────┘  │   │
│  │         │                                                          │   │
│  │         ▼                                                          │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │              INNER OPTIMIZATION (H steps)                    │  │   │
│  │  │  ┌────────┐  ┌────────┐  ┌────────┐        ┌────────┐       │  │   │
│  │  │  │ Step 1 │─►│ Step 2 │─►│ Step 3 │─...─►  │ Step H │       │  │   │
│  │  │  │ Forward│  │ Forward│  │ Forward│        │ Forward│       │  │   │
│  │  │  │ Backward│  │ Backward│  │ Backward│       │ Backward│       │  │   │
│  │  │  │ AdamW  │  │ AdamW  │  │ AdamW  │        │ AdamW  │       │  │   │
│  │  │  └────────┘  └────────┘  └────────┘        └────────┘       │  │   │
│  │  │                                                              │  │   │
│  │  │  Checkpoint every K steps for fault tolerance               │  │   │
│  │  │  Stream partial deltas (Streaming DiLoCo)                   │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │         │                                                          │   │
│  │         ▼                                                          │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │              SYNCHRONIZATION                                   │  │   │
│  │  │  1. Compute outer gradient: Δ = θ^(H) - θ^(0)                │  │   │
│  │  │  2. Compress gradient (8-bit quant + top-k sparsification)   │  │   │
│  │  │  3. Submit to coordinator                                    │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │         │                                                          │   │
│  │         ▼                                                          │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │              AGGREGATION                                       │  │   │
│  │  │  1. Collect gradients from workers (with timeout)            │  │   │
│  │  │  2. Verify consistency (virtual data samples)                │  │   │
│  │  │  3. Filter Byzantine (Multi-Krum)                            │  │   │
│  │  │  4. Aggregate: Δ_global = robust_aggregate(Δ_i)              │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │         │                                                          │   │
│  │         ▼                                                          │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │              OUTER UPDATE                                      │  │   │
│  │  │  θ^(t+1) = NesterovMomentum(θ^(t), Δ_global)                 │  │   │
│  │  │  Save checkpoint to distributed storage                      │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  REPEAT until convergence                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Configuration Summary

```yaml
# SwarmResearch v0.3 Configuration

network:
  discovery:
    protocol: "kademlia-dht"
    bootstrap_nodes:
      - "bootstrap.swarmresearch.org:6881"
      - "bootstrap-eu.swarmresearch.org:6881"
    k_bucket_size: 20
    alpha_parallelism: 3
  
  transport:
    protocol: "quic"
    encryption: "tls1.3+noise"
    compression: "zstd"
    nat_traversal:
      stun_servers: 3
      turn_fallback: true

training:
  algorithm: "diloco"
  inner_optimizer: "adamw"
  outer_optimizer: "nesterov"
  inner_steps: 500  # H parameter
  learning_rate:
    inner: 1e-4
    outer: 0.7
  
  compression:
    quantization_bits: 8
    sparsity_ratio: 0.01
    error_feedback: true
    target_ratio: 100.0
  
  streaming:
    enabled: true
    buffer_size: 10

fault_tolerance:
  checkpoint_interval: 100  # steps
  replication_factor: 3
  timeout_factor: 1.5
  min_participation: 0.6
  failure_detection:
    threshold: 8.0
    min_std_dev: 0.5

security:
  max_byzantine_ratio: 0.33
  aggregation_rule: "multi-krum"
  consistency_verification:
    enabled: true
    virtual_samples: 100
    threshold: 0.9
  reputation:
    min_contributions_for_trust: 5
    stake_required: false  # v0.3 may add staking
```

---

## 7. References

1. Douillard et al. (2023). "DiLoCo: Distributed Low-Communication Training of Language Models." arXiv:2311.08105.

2. Douillard & Donchev (2025). "Eager Updates For Overlapped Communication and Computation in DiLoCo." arXiv:2502.12996.

3. McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS.

4. Blanchard et al. (2017). "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent." NeurIPS.

5. El-Mhamdi et al. (2021). "Distributed Momentum for Byzantine-resilient Stochastic Gradient Descent." ICLR.

6. Cox et al. (2024). "Asynchronous Byzantine Federated Learning." arXiv:2406.01438.

---

*Document Version: 0.3.0*
*Last Updated: 2025*
