# SwarmResearch Inter-Agent Communication Protocol

## Executive Summary

This document defines the inter-agent communication protocol for SwarmResearch, a massively parallel collaborative AI research system. The protocol is designed to enable efficient, scalable, and reliable communication between hundreds of AI agents across multiple LLM providers.

**Key Design Decisions:**
- Hybrid protocol combining A2A-inspired agent discovery with MCP-inspired tool integration
- Multi-pattern communication (pub/sub, direct, broadcast) for different use cases
- Message queuing with backpressure for flow control
- Protocol-agnostic transport layer supporting Redis, RabbitMQ, NATS, and in-memory

---

## 1. Protocol Architecture Overview

### 1.1 Design Principles

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMMUNICATION PROTOCOL PRINCIPLES                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. PATTERN FLEXIBILITY                                                      │
│     • Use the right pattern for the right job                               │
│     • Pub/Sub for events, Direct for RPC, Broadcast for discovery           │
│                                                                              │
│  2. TRANSPORT AGNOSTICISM                                                    │
│     • Core protocol independent of underlying transport                     │
│     • Pluggable adapters for Redis, RabbitMQ, NATS, gRPC, HTTP              │
│                                                                              │
│  3. EFFICIENCY FIRST                                                         │
│     • Binary serialization for high-throughput scenarios                    │
│     • Message batching and compression                                      │
│     • Connection pooling and multiplexing                                   │
│                                                                              │
│  4. RELIABILITY GUARANTEES                                                   │
│     • At-least-once delivery with idempotency                               │
│     • Dead letter queues for failed messages                                │
│     • Circuit breakers for fault isolation                                  │
│                                                                              │
│  5. OBSERVABILITY                                                            │
│     • Distributed tracing across agent boundaries                           │
│     • Metrics collection at all protocol layers                             │
│     • Structured logging with correlation IDs                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Protocol Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROTOCOL STACK LAYERS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Layer 5: Application Layer                                                  │
│  ─────────────────────────                                                   │
│  • Agent Cards (A2A-inspired capability discovery)                          │
│  • Task Lifecycle Management                                                │
│  • Skill Registry & Invocation                                              │
│                                                                              │
│  Layer 4: Messaging Layer                                                    │
│  ───────────────────────                                                     │
│  • Message Formats (JSON/Binary)                                            │
│  • Schema Validation                                                        │
│  • Serialization/Deserialization                                            │
│                                                                              │
│  Layer 3: Routing Layer                                                      │
│  ─────────────────────                                                       │
│  • Topic-based Pub/Sub                                                      │
│  • Direct Message Routing                                                   │
│  • Broadcast & Multicast                                                    │
│  • Message Filtering                                                        │
│                                                                              │
│  Layer 2: Transport Layer                                                    │
│  ───────────────────────                                                     │
│  • Connection Management                                                    │
│  • Flow Control & Backpressure                                              │
│  • Reliability (ACKs, Retries)                                              │
│                                                                              │
│  Layer 1: Network Layer                                                      │
│  ─────────────────────                                                       │
│  • Redis Pub/Sub                                                            │
│  • RabbitMQ AMQP                                                            │
│  • NATS JetStream                                                           │
│  • gRPC/HTTP/2                                                              │
│  • In-Memory (dev/testing)                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Message Formats

### 2.1 Core Message Schema

All messages in SwarmResearch follow a unified envelope format:

```python
# Core message envelope - all messages wrap this structure
class SwarmMessage(BaseModel):
    """Universal message envelope for SwarmResearch"""
    
    # Identity
    message_id: UUID                    # Unique message identifier
    correlation_id: UUID                # Links related messages
    parent_id: Optional[UUID]           # For threaded conversations
    
    # Timing
    timestamp: datetime                 # ISO8601 UTC
    ttl: Optional[int]                  # Time-to-live in seconds
    
    # Routing
    source: AgentAddress                # Sender identity
    destination: RoutingInfo            # Target specification
    
    # Content
    message_type: MessageType           # Type discriminator
    payload: MessagePayload             # Typed payload
    
    # Metadata
    priority: int = 5                   # 1-10, lower = higher priority
    trace_context: Optional[TraceContext]  # Distributed tracing
    compression: Optional[str]          # "gzip", "zstd", None
    
    # Protocol version
    version: str = "1.0"

class AgentAddress(BaseModel):
    """Unique agent identifier with location info"""
    agent_id: UUID
    agent_type: str                     # "researcher", "critic", "orchestrator"
    node_id: str                        # Physical/logical node identifier
    capabilities: List[str]             # Advertised capabilities

class RoutingInfo(BaseModel):
    """Routing specification"""
    routing_type: RoutingType           # DIRECT, PUBSUB, BROADCAST, MULTICAST
    target: Union[str, List[str]]       # Topic, agent_id, or pattern
    headers: Dict[str, str]             # Filter headers for routing

class TraceContext(BaseModel):
    """OpenTelemetry-compatible tracing"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    sampled: bool = True
```

### 2.2 Message Type Hierarchy

```python
class MessageType(str, Enum):
    """All message types in the system"""
    
    # Task Management
    TASK_SUBMIT = "task.submit"         # Submit new task
    TASK_ASSIGN = "task.assign"         # Task assigned to agent
    TASK_UPDATE = "task.update"         # Progress update
    TASK_COMPLETE = "task.complete"     # Task finished
    TASK_FAILED = "task.failed"         # Task error
    TASK_CANCEL = "task.cancel"         # Cancel request
    
    # Agent Lifecycle
    AGENT_SPAWN = "agent.spawn"         # Create new agent
    AGENT_READY = "agent.ready"         # Agent initialized
    AGENT_HEARTBEAT = "agent.heartbeat" # Health ping
    AGENT_SHUTDOWN = "agent.shutdown"   # Graceful exit
    AGENT_FAILED = "agent.failed"       # Agent crash
    
    # Communication Patterns
    DIRECT_MESSAGE = "comm.direct"      # Point-to-point message
    BROADCAST = "comm.broadcast"        # All agents receive
    PUBLISH = "comm.publish"            # Pub/sub message
    REQUEST = "comm.request"            # RPC request
    RESPONSE = "comm.response"          # RPC response
    
    # Discovery & Registry
    AGENT_CARD = "discovery.card"       # Capability advertisement
    SKILL_REGISTER = "discovery.skill"  # Register skill
    SKILL_INVOKE = "discovery.invoke"   # Invoke remote skill
    
    # Coordination
    CONSENSUS_PROPOSE = "coord.propose" # Consensus proposal
    CONSENSUS_VOTE = "coord.vote"       # Vote on proposal
    CONSENSUS_COMMIT = "coord.commit"   # Commit decision
    LOCK_ACQUIRE = "coord.lock_acquire" # Distributed lock
    LOCK_RELEASE = "coord.lock_release" # Release lock
    
    # Data & Results
    RESULT_PARTIAL = "data.partial"     # Partial result
    RESULT_FINAL = "data.final"         # Final result
    STREAM_CHUNK = "data.stream"        # Streaming chunk
    EMBEDDING = "data.embedding"        # Vector embedding
```

### 2.3 Payload Schemas by Type

#### Task Messages

```python
class TaskSubmitPayload(BaseModel):
    """Payload for task submission"""
    task_id: UUID
    task_type: str
    description: str
    requirements: TaskRequirements
    context: Dict[str, Any]
    dependencies: List[UUID]            # Task dependencies
    deadline: Optional[datetime]
    max_retries: int = 3

class TaskUpdatePayload(BaseModel):
    """Progress update payload"""
    task_id: UUID
    status: TaskStatus                  # PENDING, RUNNING, COMPLETED, FAILED
    progress: float                     # 0.0 - 1.0
    current_step: Optional[str]
    partial_results: Optional[List[PartialResult]]
    estimated_completion: Optional[datetime]
    metrics: Optional[TaskMetrics]

class TaskCompletePayload(BaseModel):
    """Task completion payload"""
    task_id: UUID
    result: Result
    execution_time_ms: int
    tokens_used: TokenUsage
    artifacts: List[Artifact]
    citations: List[Citation]
```

#### Agent Card (A2A-inspired)

```python
class AgentCard(BaseModel):
    """Agent capability advertisement (A2A-inspired)"""
    
    # Identity
    agent_id: UUID
    name: str
    description: str
    version: str
    
    # Endpoint
    endpoint_url: str
    authentication: AuthScheme
    
    # Capabilities
    skills: List[Skill]
    modalities: List[str]               # ["text", "image", "code"]
    models: List[str]                   # Supported LLM models
    
    # Limits
    rate_limits: RateLimits
    max_concurrent_tasks: int
    
    # Metadata
    provider: str                       # "kimi", "claude", "openai", "ollama"
    tags: List[str]
    created_at: datetime

class Skill(BaseModel):
    """Individual skill definition"""
    id: str
    name: str
    description: str
    parameters: Dict[str, Any]          # JSON Schema
    return_type: Dict[str, Any]         # JSON Schema
    examples: List[SkillExample]

class AuthScheme(BaseModel):
    """Authentication requirements"""
    scheme: str                         # "api_key", "bearer", "oauth2", "mtls"
    required: bool
    scopes: Optional[List[str]]
```

#### RPC Messages

```python
class RPCRequestPayload(BaseModel):
    """Remote procedure call request"""
    method: str                         # "skill.invoke", "task.query", etc.
    params: Dict[str, Any]
    request_id: UUID
    timeout_ms: int = 30000

class RPCResponsePayload(BaseModel):
    """RPC response"""
    request_id: UUID
    success: bool
    result: Optional[Any]
    error: Optional[RPCError]
    execution_time_ms: int

class RPCError(BaseModel):
    """RPC error details"""
    code: str                           # "TIMEOUT", "NOT_FOUND", "RATE_LIMITED"
    message: str
    details: Optional[Dict[str, Any]]
    retryable: bool
```

### 2.4 Binary Serialization (High Throughput)

For high-throughput scenarios, messages can be serialized using Protocol Buffers or MessagePack:

```python
# Binary message format for performance-critical paths
class BinaryMessage:
    """Compact binary message format"""
    
    HEADER_SIZE = 24
    
    # Header (24 bytes)
    # - message_id: 16 bytes (UUID)
    # - message_type: 1 byte (enum index)
    # - priority: 1 byte
    # - flags: 2 bytes (compression, encryption, etc.)
    # - payload_length: 4 bytes
    
    # Body (variable)
    # - source_agent_id: 16 bytes
    # - correlation_id: 16 bytes
    # - timestamp: 8 bytes (unix nanoseconds)
    # - payload: protobuf/msgpack encoded
```

---

## 3. Communication Patterns

### 3.1 Pattern Selection Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMMUNICATION PATTERN SELECTION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Use Case                    │ Pattern      │ Reliability │ Latency         │
│  ────────────────────────────┼──────────────┼─────────────┼─────────────────│
│  Task submission             │ PUB/SUB      │ At-least-once │ < 10ms        │
│  Direct agent communication  │ DIRECT       │ At-least-once │ < 5ms         │
│  Discovery & announcements   │ BROADCAST    │ Best-effort │ < 50ms          │
│  RPC/skill invocation        │ REQUEST/RESP │ At-least-once │ < 100ms       │
│  Result streaming            │ STREAM       │ At-least-once │ Real-time     │
│  Consensus voting            │ MULTICAST    │ At-least-once │ < 200ms       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Publish/Subscribe Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PUB/SUB PATTERN ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                              ┌─────────────┐                                 │
│                              │  Publisher  │                                 │
│                              │  (Any Agent)│                                 │
│                              └──────┬──────┘                                 │
│                                     │ publish()                               │
│                                     ▼                                         │
│                         ┌───────────────────────┐                            │
│                         │    Message Broker     │                            │
│                         │  (Redis/RabbitMQ/NATS)│                            │
│                         └───────────┬───────────┘                            │
│                                     │                                         │
│           ┌─────────────────────────┼─────────────────────────┐              │
│           │                         │                         │              │
│           ▼                         ▼                         ▼              │
│    ┌─────────────┐           ┌─────────────┐           ┌─────────────┐      │
│    │ Subscriber 1│           │ Subscriber 2│           │ Subscriber N│      │
│    │ (Filter:    │           │ (Filter:    │           │ (Filter:    │      │
│    │  priority>5)│           │  type=research)          │  all)       │      │
│    └─────────────┘           └─────────────┘           └─────────────┘      │
│                                                                              │
│  Topic Hierarchy:                                                            │
│  ────────────────                                                            │
│  • tasks.{new,assigned,completed,failed}                                    │
│  • agents.{heartbeat,spawn,shutdown}                                        │
│  • results.{partial,final}                                                  │
│  • swarm.{rebalance,config}                                                 │
│  • research.{topic}.{subtopic}                                              │
│                                                                              │
│  Wildcard Subscriptions:                                                     │
│  • "tasks.*" - all task events                                              │
│  • "research.ai.*" - all AI research topics                                 │
│  • "agents.>" - all agent lifecycle events (NATS style)                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
class PubSubManager:
    """Publish/Subscribe pattern implementation"""
    
    async def publish(
        self,
        topic: str,
        message: SwarmMessage,
        options: PublishOptions = None
    ) -> PublishResult:
        """Publish message to topic"""
        
        # Apply compression for large payloads
        if len(message.payload) > COMPRESSION_THRESHOLD:
            message.payload = compress(message.payload)
            message.compression = "zstd"
        
        # Serialize based on format preference
        serialized = self.serializer.serialize(message)
        
        # Publish to underlying transport
        return await self.transport.publish(topic, serialized, options)
    
    async def subscribe(
        self,
        pattern: str,
        handler: MessageHandler,
        options: SubscribeOptions = None
    ) -> Subscription:
        """Subscribe to topic pattern"""
        
        # Create message handler wrapper
        async def wrapped_handler(raw_message: bytes):
            message = self.serializer.deserialize(raw_message)
            
            # Decompress if needed
            if message.compression:
                message.payload = decompress(message.payload, message.compression)
            
            # Apply filter if specified
            if options and options.filter:
                if not options.filter.matches(message):
                    return
            
            await handler(message)
        
        return await self.transport.subscribe(pattern, wrapped_handler, options)
```

### 3.3 Direct Message Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DIRECT MESSAGE PATTERN                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                              ┌─────────────┐               │
│  │   Agent A   │                              │   Agent B   │               │
│  │ (Requester) │                              │ (Responder) │               │
│  └──────┬──────┘                              └──────┬──────┘               │
│         │                                            │                        │
│         │  1. send_direct(agent_b_id, message)       │                        │
│         │────────────────────────────────────────────>│                        │
│         │                                            │                        │
│         │         2. Message routed to inbox         │                        │
│         │                                            │                        │
│         │  3. ACK (optional)                         │                        │
│         │<────────────────────────────────────────────│                        │
│         │                                            │                        │
│         │         4. Process message                 │                        │
│         │                                            │                        │
│         │  5. Response (if requested)                │                        │
│         │<────────────────────────────────────────────│                        │
│         │                                            │                        │
│                                                                              │
│  Inbox Queue per Agent:                                                      │
│  ──────────────────────                                                      │
│  Each agent has a dedicated inbox queue: "inbox.{agent_id}"                 │
│  Messages are persisted until acknowledged or expired                        │
│                                                                              │
│  Delivery Guarantees:                                                        │
│  ─────────────────────                                                       │
│  • At-most-once: Fire and forget                                             │
│  • At-least-once: With acknowledgment                                        │
│  • Exactly-once: With deduplication (idempotency keys)                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
class DirectMessaging:
    """Direct point-to-point messaging"""
    
    async def send(
        self,
        target_agent_id: UUID,
        message: SwarmMessage,
        delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE,
        timeout_ms: int = 30000
    ) -> Optional[SwarmMessage]:
        """Send direct message to agent"""
        
        inbox_topic = f"inbox.{target_agent_id}"
        
        if delivery_guarantee == DeliveryGuarantee.AT_MOST_ONCE:
            # Fire and forget
            await self.transport.publish(inbox_topic, message)
            return None
        
        elif delivery_guarantee == DeliveryGuARANTEE.AT_LEAST_ONCE:
            # Wait for acknowledgment
            ack_future = self.pending_acks[message.message_id] = asyncio.Future()
            await self.transport.publish(inbox_topic, message)
            
            try:
                await asyncio.wait_for(ack_future, timeout_ms / 1000)
                return None
            except asyncio.TimeoutError:
                raise DeliveryTimeout(f"Message {message.message_id} not acknowledged")
        
        elif delivery_guarantee == DeliveryGuarantee.EXACTLY_ONCE:
            # Idempotent delivery with deduplication
            message.deduplication_key = generate_dedup_key(message)
            return await self.exactly_once_delivery(target_agent_id, message, timeout_ms)
    
    async def receive(self, handler: MessageHandler) -> None:
        """Start receiving messages to this agent's inbox"""
        
        inbox_topic = f"inbox.{self.agent_id}"
        
        async def inbox_handler(message: SwarmMessage):
            # Check for duplicates
            if await self.dedup_store.is_duplicate(message.deduplication_key):
                return
            
            # Send acknowledgment if requested
            if message.requires_ack:
                await self.send_ack(message.source.agent_id, message.message_id)
            
            # Process message
            await handler(message)
        
        await self.transport.subscribe(inbox_topic, inbox_handler)
```

### 3.4 Broadcast Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BROADCAST PATTERN                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         ┌─────────────┐                                      │
│                         │  Broadcaster│                                      │
│                         │  (Orchestrator)                                   │
│                         └──────┬──────┘                                      │
│                                │ broadcast()                                 │
│                                ▼                                             │
│                    ┌───────────────────────┐                                 │
│                    │   Broadcast Channel   │                                 │
│                    │   (swarm.broadcast)   │                                 │
│                    └───────────┬───────────┘                                 │
│                                │                                             │
│        ┌───────────────────────┼───────────────────────┐                     │
│        │                       │                       │                     │
│        ▼                       ▼                       ▼                     │
│  ┌───────────┐           ┌───────────┐           ┌───────────┐              │
│  │  Agent 1  │           │  Agent 2  │           │  Agent N  │              │
│  │ (receives)│           │ (receives)│           │ (receives)│              │
│  └───────────┘           └───────────┘           └───────────┘              │
│        │                       │                       │                     │
│        │ filter_match?         │ filter_match?         │ filter_match?      │
│        │                       │                       │                     │
│        ▼                       ▼                       ▼                     │
│   [process]              [ignore]                [process]                   │
│                                                                              │
│  Use Cases:                                                                  │
│  ──────────                                                                  │
│  • System-wide announcements (shutdown, reconfiguration)                    │
│  • Discovery broadcasts (new agent available)                               │
│  • Emergency signals (circuit breaker trip)                                 │
│  • Configuration updates                                                    │
│                                                                              │
│  Filtering:                                                                  │
│  ──────────                                                                  │
│  Agents filter broadcasts based on headers:                                  │
│  • capability in message.required_capabilities                              │
│  • agent_type in message.target_types                                       │
│  • tags overlap with message.target_tags                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Request/Response (RPC) Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      REQUEST/RESPONSE (RPC) PATTERN                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                    ┌─────────────┐         │
│  │   Client    │                                    │   Server    │         │
│  │  (Agent A)  │                                    │  (Agent B)  │         │
│  └──────┬──────┘                                    └──────┬──────┘         │
│         │                                                  │                │
│         │  1. RPC Request                                  │                │
│         │  {                                               │                │
│         │    method: "skill.invoke",                       │                │
│         │    params: {...},                                │                │
│         │    request_id: "uuid",                           │                │
│         │    reply_to: "inbox.agent_a"                     │                │
│         │  }                                               │                │
│         │─────────────────────────────────────────────────>│                │
│         │                                                  │                │
│         │                        2. Process request        │                │
│         │                        (may take seconds)        │                │
│         │                                                  │                │
│         │  3. RPC Response                                 │                │
│         │  {                                               │                │
│         │    request_id: "uuid",  // matches request       │                │
│         │    success: true,                                │                │
│         │    result: {...}                                 │                │
│         │  }                                               │                │
│         │<─────────────────────────────────────────────────│                │
│         │                                                  │                │
│                                                                              │
│  Timeout Handling:                                                           │
│  ─────────────────                                                           │
│  • Client sets timeout (default 30s)                                        │
│  • Server can send progress updates for long-running operations             │
│  • On timeout: client may retry with same request_id (idempotent)           │
│                                                                              │
│  Error Codes:                                                                │
│  ────────────                                                                │
│  • METHOD_NOT_FOUND: Skill/method doesn't exist                             │
│  • INVALID_PARAMS: Parameters don't match schema                            │
│  • EXECUTION_ERROR: Runtime error during execution                          │
│  • TIMEOUT: Request timed out                                               │
│  • RATE_LIMITED: Too many requests                                          │
│  • NOT_AVAILABLE: Agent temporarily unavailable                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
class RPCClient:
    """RPC client for agent-to-agent calls"""
    
    def __init__(self, transport: Transport, serializer: Serializer):
        self.transport = transport
        self.serializer = serializer
        self.pending_requests: Dict[UUID, asyncio.Future] = {}
    
    async def call(
        self,
        target_agent_id: UUID,
        method: str,
        params: Dict[str, Any],
        timeout_ms: int = 30000
    ) -> RPCResult:
        """Make RPC call to another agent"""
        
        request_id = uuid4()
        reply_topic = f"inbox.{self.agent_id}"
        
        request = RPCRequestPayload(
            method=method,
            params=params,
            request_id=request_id,
            timeout_ms=timeout_ms
        )
        
        message = SwarmMessage(
            message_id=uuid4(),
            correlation_id=request_id,
            source=self.agent_address,
            destination=RoutingInfo(
                routing_type=RoutingType.DIRECT,
                target=str(target_agent_id)
            ),
            message_type=MessageType.REQUEST,
            payload=request,
            reply_to=reply_topic
        )
        
        # Create future for response
        response_future = asyncio.Future()
        self.pending_requests[request_id] = response_future
        
        try:
            # Send request
            await self.transport.send_direct(target_agent_id, message)
            
            # Wait for response
            response = await asyncio.wait_for(
                response_future,
                timeout_ms / 1000
            )
            
            return RPCResult(
                success=response.payload.success,
                result=response.payload.result,
                error=response.payload.error
            )
            
        except asyncio.TimeoutError:
            return RPCResult(
                success=False,
                error=RPCError(
                    code="TIMEOUT",
                    message=f"RPC call timed out after {timeout_ms}ms",
                    retryable=True
                )
            )
        finally:
            del self.pending_requests[request_id]
    
    async def handle_response(self, message: SwarmMessage):
        """Handle incoming RPC response"""
        request_id = message.payload.request_id
        if request_id in self.pending_requests:
            self.pending_requests[request_id].set_result(message)
```

### 3.6 Streaming Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STREAMING PATTERN                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                    ┌─────────────┐         │
│  │  Producer   │                                    │  Consumer   │         │
│  │  (Agent A)  │                                    │  (Agent B)  │         │
│  └──────┬──────┘                                    └──────┬──────┘         │
│         │                                                  │                │
│         │  1. Stream Init                                  │                │
│         │  { type: STREAM_INIT, stream_id, chunks_expected }                │
│         │─────────────────────────────────────────────────>│                │
│         │                                                  │                │
│         │  2. Stream Chunk 1                               │                │
│         │  { type: STREAM_CHUNK, stream_id, seq: 1, data }                │
│         │═════════════════════════════════════════════════>│                │
│         │  3. Stream Chunk 2                               │                │
│         │  { type: STREAM_CHUNK, stream_id, seq: 2, data }                │
│         │═════════════════════════════════════════════════>│                │
│         │  4. Stream Chunk N                               │                │
│         │  { type: STREAM_CHUNK, stream_id, seq: N, data }                │
│         │═════════════════════════════════════════════════>│                │
│         │                                                  │                │
│         │  5. Stream Complete                              │                │
│         │  { type: STREAM_COMPLETE, stream_id }           │                │
│         │─────────────────────────────────────────────────>│                │
│         │                                                  │                │
│         │<═════════════════════════════════════════════════│                │
│         │  ACK (per-chunk or cumulative)                   │                │
│                                                                              │
│  Flow Control:                                                               │
│  ─────────────                                                               │
│  • Consumer advertises window size (max unacknowledged chunks)              │
│  • Producer pauses when window is full                                      │
│  • Consumer can send WINDOW_UPDATE to increase window                       │
│                                                                              │
│  Use Cases:                                                                  │
│  ──────────                                                                  │
│  • LLM token streaming                                                       │
│  • Large result transfer (chunked)                                           │
│  • Real-time progress updates                                                │
│  • Log/event streaming                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Protocol Efficiency

### 4.1 Message Batching

```python
class MessageBatcher:
    """Batch multiple messages for efficient transmission"""
    
    def __init__(
        self,
        max_batch_size: int = 100,
        max_batch_bytes: int = 1024 * 1024,  # 1MB
        max_wait_ms: int = 10
    ):
        self.max_batch_size = max_batch_size
        self.max_batch_bytes = max_batch_bytes
        self.max_wait_ms = max_wait_ms
        self.batch: List[SwarmMessage] = []
        self.batch_bytes = 0
        self.flush_timer: Optional[asyncio.Task] = None
    
    async def add(self, message: SwarmMessage) -> Optional[BatchMessage]:
        """Add message to batch, return batch if flushed"""
        
        message_bytes = len(self.serializer.serialize(message))
        
        # Check if adding would exceed limits
        if (len(self.batch) >= self.max_batch_size or
            self.batch_bytes + message_bytes > self.max_batch_bytes):
            # Flush current batch first
            batch = await self.flush()
            self.batch = [message]
            self.batch_bytes = message_bytes
            return batch
        
        self.batch.append(message)
        self.batch_bytes += message_bytes
        
        # Start flush timer if not running
        if self.flush_timer is None:
            self.flush_timer = asyncio.create_task(self._delayed_flush())
        
        return None
    
    async def flush(self) -> BatchMessage:
        """Flush current batch"""
        
        if self.flush_timer:
            self.flush_timer.cancel()
            self.flush_timer = None
        
        batch = BatchMessage(
            batch_id=uuid4(),
            messages=self.batch,
            timestamp=datetime.utcnow()
        )
        
        self.batch = []
        self.batch_bytes = 0
        
        return batch
    
    async def _delayed_flush(self):
        """Flush after max_wait_ms"""
        await asyncio.sleep(self.max_wait_ms / 1000)
        if self.batch:
            await self.flush()
```

### 4.2 Compression Strategy

```python
class CompressionManager:
    """Adaptive compression based on payload characteristics"""
    
    COMPRESSION_THRESHOLD = 1024  # 1KB
    
    def __init__(self):
        self.compression_stats: Dict[str, CompressionStats] = {}
    
    async def compress(
        self,
        data: bytes,
        hint: CompressionHint = None
    ) -> Tuple[bytes, Optional[str]]:
        """Compress data with optimal algorithm"""
        
        # Don't compress small payloads
        if len(data) < self.COMPRESSION_THRESHOLD:
            return data, None
        
        # Select algorithm based on hint and data characteristics
        algorithm = self._select_algorithm(data, hint)
        
        if algorithm == "zstd":
            compressed = zstd.compress(data, level=3)
        elif algorithm == "gzip":
            compressed = gzip.compress(data, compresslevel=6)
        elif algorithm == "lz4":
            compressed = lz4.frame.compress(data)
        else:
            return data, None
        
        # Only use compression if it actually reduces size
        if len(compressed) < len(data) * 0.9:
            return compressed, algorithm
        
        return data, None
    
    def _select_algorithm(
        self,
        data: bytes,
        hint: CompressionHint
    ) -> str:
        """Select best compression algorithm"""
        
        if hint:
            return hint.preferred_algorithm
        
        # Heuristic selection
        if len(data) > 100 * 1024:  # > 100KB
            return "zstd"  # Best compression ratio
        elif len(data) > 10 * 1024:  # > 10KB
            return "gzip"  # Good balance
        else:
            return "lz4"   # Fast, low overhead
```

### 4.3 Connection Pooling

```python
class ConnectionPool:
    """Pool connections to message broker for reuse"""
    
    def __init__(
        self,
        broker_url: str,
        min_connections: int = 5,
        max_connections: int = 50,
        max_idle_time: int = 300
    ):
        self.broker_url = broker_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        
        self.available: asyncio.Queue[Connection] = asyncio.Queue()
        self.in_use: Set[Connection] = set()
        self.total_connections = 0
    
    async def acquire(self) -> Connection:
        """Acquire connection from pool"""
        
        # Try to get available connection
        try:
            conn = self.available.get_nowait()
            if await self._is_valid(conn):
                self.in_use.add(conn)
                return conn
            else:
                await self._close(conn)
        except asyncio.QueueEmpty:
            pass
        
        # Create new connection if under limit
        if self.total_connections < self.max_connections:
            conn = await self._create_connection()
            self.total_connections += 1
            self.in_use.add(conn)
            return conn
        
        # Wait for available connection
        conn = await self.available.get()
        self.in_use.add(conn)
        return conn
    
    async def release(self, conn: Connection):
        """Release connection back to pool"""
        
        self.in_use.remove(conn)
        
        if await self._is_valid(conn):
            await self.available.put(conn)
        else:
            await self._close(conn)
            self.total_connections -= 1
```

### 4.4 Protocol Efficiency Metrics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EFFICIENCY TARGETS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Metric                      │ Target    │ Measurement                      │
│  ────────────────────────────┼───────────┼──────────────────────────────────│
│  Message latency (p50)       │ < 5ms     │ End-to-end within datacenter     │
│  Message latency (p99)       │ < 50ms    │ Including retries                │
│  Throughput per agent        │ > 1000/s  │ Messages processed               │
│  Serialization overhead      │ < 10%     │ Compared to raw payload          │
│  Compression ratio           │ > 3x      │ For text-heavy payloads          │
│  Connection reuse rate       │ > 95%     │ Pool hit rate                    │
│  Batch efficiency            │ > 80%     │ Avg batch size / max batch size  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Message Routing

### 5.1 Routing Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MESSAGE ROUTING ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                           ┌─────────────┐                                    │
│                           │   Router    │                                    │
│                           │  (Core)     │                                    │
│                           └──────┬──────┘                                    │
│                                  │                                           │
│           ┌──────────────────────┼──────────────────────┐                    │
│           │                      │                      │                    │
│           ▼                      ▼                      ▼                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Topic Router   │  │  Agent Router   │  │  Load Balancer  │              │
│  │  (Pub/Sub)      │  │  (Direct)       │  │  (Distribution) │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                      │                      │
│           ▼                    ▼                      ▼                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Topic:         │  │  Agent Registry │  │  Worker Pool    │              │
│  │  tasks.new      │  │  (Consistent    │  │  (Round-robin,  │              │
│  │  agents.>       │  │   Hash Ring)    │  │   Least-conn)   │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                              │
│  Routing Strategies:                                                         │
│  ───────────────────                                                         │
│  • Direct: Route to specific agent_id                                       │
│  • Topic: Route to all subscribers of topic                                 │
│  • Broadcast: Route to all agents (with filtering)                          │
│  • Load-balanced: Route to least-loaded agent with capability               │
│  • Consistent-hash: Route based on task_id for locality                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Topic-Based Routing

```python
class TopicRouter:
    """Route messages based on topic patterns"""
    
    def __init__(self):
        self.subscriptions: Dict[str, Set[UUID]] = {}
        self.wildcard_subs: List[Tuple[str, UUID]] = []  # (pattern, agent_id)
    
    def subscribe(self, agent_id: UUID, pattern: str):
        """Subscribe agent to topic pattern"""
        
        if '*' in pattern or '>' in pattern:
            self.wildcard_subs.append((pattern, agent_id))
        else:
            if pattern not in self.subscriptions:
                self.subscriptions[pattern] = set()
            self.subscriptions[pattern].add(agent_id)
    
    def route(self, topic: str) -> Set[UUID]:
        """Find all agents subscribed to topic"""
        
        recipients = set()
        
        # Direct match
        if topic in self.subscriptions:
            recipients.update(self.subscriptions[topic])
        
        # Wildcard match
        for pattern, agent_id in self.wildcard_subs:
            if self._match_wildcard(topic, pattern):
                recipients.add(agent_id)
        
        return recipients
    
    def _match_wildcard(self, topic: str, pattern: str) -> bool:
        """Match topic against wildcard pattern"""
        
        # * matches single token
        # > matches any number of tokens (NATS style)
        
        topic_parts = topic.split('.')
        pattern_parts = pattern.split('.')
        
        t_idx, p_idx = 0, 0
        
        while t_idx < len(topic_parts) and p_idx < len(pattern_parts):
            if pattern_parts[p_idx] == '*':
                t_idx += 1
                p_idx += 1
            elif pattern_parts[p_idx] == '>':
                return True  # > matches rest
            elif pattern_parts[p_idx] == topic_parts[t_idx]:
                t_idx += 1
                p_idx += 1
            else:
                return False
        
        return t_idx == len(topic_parts) and p_idx == len(pattern_parts)
```

### 5.3 Capability-Based Routing

```python
class CapabilityRouter:
    """Route to agents based on required capabilities"""
    
    def __init__(self, agent_registry: AgentRegistry):
        self.registry = agent_registry
        self.capability_index: Dict[str, Set[UUID]] = {}
    
    async def find_agents(
        self,
        required_capabilities: List[str],
        preferences: RoutingPreferences = None
    ) -> List[AgentMatch]:
        """Find agents matching capability requirements"""
        
        # Find intersection of all capability sets
        candidate_ids = None
        for cap in required_capabilities:
            agents_with_cap = self.capability_index.get(cap, set())
            if candidate_ids is None:
                candidate_ids = agents_with_cap
            else:
                candidate_ids &= agents_with_cap
        
        if not candidate_ids:
            return []
        
        # Score and rank candidates
        matches = []
        for agent_id in candidate_ids:
            agent = await self.registry.get_agent(agent_id)
            if not agent or not agent.available:
                continue
            
            score = self._score_agent(agent, required_capabilities, preferences)
            matches.append(AgentMatch(agent_id=agent_id, score=score, agent=agent))
        
        # Sort by score descending
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches
    
    def _score_agent(
        self,
        agent: AgentInfo,
        required_capabilities: List[str],
        preferences: RoutingPreferences
    ) -> float:
        """Score agent suitability (0-1)"""
        
        score = 0.0
        
        # Capability match (40%)
        matching_caps = set(agent.capabilities) & set(required_capabilities)
        score += 0.4 * (len(matching_caps) / len(required_capabilities))
        
        # Load factor (30%) - prefer less loaded agents
        load_score = 1.0 - (agent.active_tasks / agent.max_concurrent_tasks)
        score += 0.3 * load_score
        
        # Latency (20%) - prefer lower latency
        if agent.avg_latency_ms:
            latency_score = max(0, 1.0 - (agent.avg_latency_ms / 1000))
            score += 0.2 * latency_score
        
        # Provider preference (10%)
        if preferences and preferences.preferred_provider:
            if agent.provider == preferences.preferred_provider:
                score += 0.1
        
        return score
```

### 5.4 Consistent Hashing for State Locality

```python
class ConsistentHashRouter:
    """Route related tasks to same agent for cache locality"""
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: SortedDict[int, UUID] = SortedDict()  # hash -> agent_id
        self.agents: Dict[UUID, Set[int]] = {}  # agent_id -> virtual node hashes
    
    def add_agent(self, agent_id: UUID):
        """Add agent to hash ring"""
        
        node_hashes = set()
        for i in range(self.virtual_nodes):
            node_key = f"{agent_id}:{i}"
            hash_val = self._hash(node_key)
            self.ring[hash_val] = agent_id
            node_hashes.add(hash_val)
        
        self.agents[agent_id] = node_hashes
    
    def remove_agent(self, agent_id: UUID):
        """Remove agent from hash ring"""
        
        for hash_val in self.agents.get(agent_id, []):
            del self.ring[hash_val]
        
        del self.agents[agent_id]
    
    def route(self, key: str) -> UUID:
        """Route key to responsible agent"""
        
        if not self.ring:
            raise NoAgentsAvailable()
        
        key_hash = self._hash(key)
        
        # Find first node >= key_hash
        idx = self.ring.bisect_left(key_hash)
        
        # Wrap around if needed
        if idx >= len(self.ring):
            idx = 0
        
        return self.ring.peekitem(idx)[1]
    
    def _hash(self, key: str) -> int:
        """Consistent hash function"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
```

---

## 6. Backpressure Handling

### 6.1 Backpressure Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BACKPRESSURE ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐ │
│  │  Producer   │────>│   Queue     │────>│  Consumer   │────>│  Handler  │ │
│  │  (Agent A)  │     │  (Bounded)  │     │  (Worker)   │     │ (Process) │ │
│  └─────────────┘     └──────┬──────┘     └──────┬──────┘     └───────────┘ │
│                             │                   │                          │
│                             │  Queue Metrics    │  Processing Metrics      │
│                             ▼                   ▼                          │
│                       ┌─────────────────────────────────┐                  │
│                       │      Backpressure Controller    │                  │
│                       │                                 │                  │
│                       │  • Queue depth monitoring       │                  │
│                       │  • Processing rate tracking     │                  │
│                       │  • Flow control decisions       │                  │
│                       │  • Producer throttling          │                  │
│                       └─────────────────────────────────┘                  │
│                                       │                                    │
│                                       ▼                                    │
│                              ┌─────────────────┐                           │
│                              │  Flow Control   │                           │
│                              │  Actions:       │                           │
│                              │  • Throttle     │                           │
│                              │  • Buffer       │                           │
│                              │  • Shed load    │                           │
│                              │  • Scale up     │                           │
│                              └─────────────────┘                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Queue-Based Backpressure

```python
class BackpressureQueue:
    """Bounded queue with backpressure signaling"""
    
    def __init__(
        self,
        max_size: int = 10000,
        high_water_mark: float = 0.8,
        low_water_mark: float = 0.3
    ):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.max_size = max_size
        self.high_water_mark = int(max_size * high_water_mark)
        self.low_water_mark = int(max_size * low_water_mark)
        
        self.backpressure_active = False
        self.backpressure_callbacks: List[Callable] = []
    
    async def put(self, item: Any) -> bool:
        """Put item with potential backpressure delay"""
        
        # Check if backpressure needed
        if self.queue.qsize() >= self.high_water_mark:
            self._activate_backpressure()
        
        # Wait for space with timeout
        try:
            await asyncio.wait_for(
                self.queue.put(item),
                timeout=30.0
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    async def get(self) -> Any:
        """Get item, potentially deactivating backpressure"""
        
        item = await self.queue.get()
        
        # Check if backpressure can be released
        if (self.backpressure_active and
            self.queue.qsize() <= self.low_water_mark):
            self._deactivate_backpressure()
        
        return item
    
    def _activate_backpressure(self):
        """Signal backpressure to producers"""
        
        if not self.backpressure_active:
            self.backpressure_active = True
            for callback in self.backpressure_callbacks:
                callback(BackpressureState.ACTIVE)
    
    def _deactivate_backpressure(self):
        """Release backpressure"""
        
        if self.backpressure_active:
            self.backpressure_active = False
            for callback in self.backpressure_callbacks:
                callback(BackpressureState.INACTIVE)
```

### 6.3 Producer Throttling

```python
class ThrottledProducer:
    """Producer with adaptive rate limiting"""
    
    def __init__(
        self,
        initial_rate: float = 1000.0,  # messages/sec
        min_rate: float = 10.0,
        max_rate: float = 10000.0
    ):
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        
        self.tokens = initial_rate
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def produce(self, message: SwarmMessage) -> bool:
        """Produce message with rate limiting"""
        
        async with self.lock:
            # Update token bucket
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(
                self.current_rate,
                self.tokens + elapsed * self.current_rate
            )
            self.last_update = now
            
            # Check if we can produce
            if self.tokens >= 1:
                self.tokens -= 1
                await self._send(message)
                return True
            else:
                # Need to wait
                wait_time = (1 - self.tokens) / self.current_rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
                await self._send(message)
                return True
    
    def adjust_rate(self, feedback: BackpressureFeedback):
        """Adjust rate based on backpressure feedback"""
        
        if feedback.state == BackpressureState.ACTIVE:
            # Reduce rate multiplicatively
            self.current_rate = max(
                self.min_rate,
                self.current_rate * 0.8
            )
        elif feedback.state == BackpressureState.INACTIVE:
            # Increase rate additively
            self.current_rate = min(
                self.max_rate,
                self.current_rate + 100
            )
```

### 6.4 Load Shedding

```python
class LoadShedder:
    """Shed load when system is overloaded"""
    
    def __init__(
        self,
        drop_threshold: float = 0.95,  # Queue fullness to start dropping
        priority_thresholds: Dict[int, float] = None
    ):
        self.drop_threshold = drop_threshold
        self.priority_thresholds = priority_thresholds or {
            1: 0.99,  # Critical: drop at 99%
            5: 0.95,  # Normal: drop at 95%
            10: 0.85  # Low: drop at 85%
        }
        self.dropped_stats: Dict[int, int] = defaultdict(int)
    
    def should_accept(
        self,
        message: SwarmMessage,
        queue_depth: int,
        queue_capacity: int
    ) -> bool:
        """Decide whether to accept or drop message"""
        
        fullness = queue_depth / queue_capacity
        
        # Get threshold for this priority
        threshold = self.priority_thresholds.get(
            message.priority,
            self.drop_threshold
        )
        
        if fullness > threshold:
            self.dropped_stats[message.priority] += 1
            return False
        
        return True
    
    def get_shedding_stats(self) -> Dict[str, Any]:
        """Get load shedding statistics"""
        
        total_dropped = sum(self.dropped_stats.values())
        return {
            "total_dropped": total_dropped,
            "by_priority": dict(self.dropped_stats),
            "drop_threshold": self.drop_threshold
        }
```

### 6.5 Circuit Breaker Pattern

```python
class CircuitBreaker:
    """Circuit breaker for fault isolation"""
    
    class State(Enum):
        CLOSED = "closed"       # Normal operation
        OPEN = "open"           # Failing, reject requests
        HALF_OPEN = "half_open" # Testing if recovered
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.lock = asyncio.Lock()
    
    async def call(self, operation: Callable) -> Any:
        """Execute operation with circuit breaker protection"""
        
        async with self.lock:
            if self.state == self.State.OPEN:
                if self._should_attempt_reset():
                    self.state = self.State.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpen()
        
        try:
            result = await operation()
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful call"""
        
        async with self.lock:
            if self.state == self.State.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.half_open_max_calls:
                    self.state = self.State.CLOSED
                    self.failure_count = 0
            else:
                self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call"""
        
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            
            if self.state == self.State.HALF_OPEN:
                self.state = self.State.OPEN
            elif self.failure_count >= self.failure_threshold:
                self.state = self.State.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery"""
        
        if self.last_failure_time is None:
            return True
        
        elapsed = time.monotonic() - self.last_failure_time
        return elapsed >= self.recovery_timeout
```

---

## 7. Transport Layer Implementation

### 7.1 Transport Interface

```python
class Transport(ABC):
    """Abstract transport interface"""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to message broker"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection"""
        pass
    
    @abstractmethod
    async def publish(
        self,
        topic: str,
        message: bytes,
        options: PublishOptions = None
    ) -> None:
        """Publish message to topic"""
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        pattern: str,
        handler: Callable[[bytes], Awaitable[None]],
        options: SubscribeOptions = None
    ) -> Subscription:
        """Subscribe to topic pattern"""
        pass
    
    @abstractmethod
    async def send_direct(
        self,
        agent_id: UUID,
        message: bytes
    ) -> None:
        """Send direct message to agent inbox"""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check transport health"""
        pass
```

### 7.2 Redis Transport

```python
class RedisTransport(Transport):
    """Redis-based transport implementation"""
    
    def __init__(self, redis_url: str, pool_size: int = 10):
        self.redis_url = redis_url
        self.pool_size = pool_size
        self.redis: Optional[aioredis.Redis] = None
        self.pubsub: Optional[aioredis.client.PubSub] = None
        self.subscriptions: Dict[str, asyncio.Task] = {}
    
    async def connect(self) -> None:
        """Connect to Redis"""
        
        self.redis = await aioredis.from_url(
            self.redis_url,
            max_connections=self.pool_size,
            decode_responses=False
        )
        self.pubsub = self.redis.pubsub()
    
    async def publish(
        self,
        topic: str,
        message: bytes,
        options: PublishOptions = None
    ) -> None:
        """Publish to Redis channel"""
        
        await self.redis.publish(topic, message)
    
    async def subscribe(
        self,
        pattern: str,
        handler: Callable[[bytes], Awaitable[None]],
        options: SubscribeOptions = None
    ) -> Subscription:
        """Subscribe to Redis channel/pattern"""
        
        if '*' in pattern:
            await self.pubsub.psubscribe(pattern)
        else:
            await self.pubsub.subscribe(pattern)
        
        # Start listener task
        task = asyncio.create_task(
            self._listen(handler, pattern)
        )
        
        subscription = Subscription(
            pattern=pattern,
            cancel=lambda: task.cancel()
        )
        
        self.subscriptions[pattern] = task
        return subscription
    
    async def _listen(
        self,
        handler: Callable[[bytes], Awaitable[None]],
        pattern: str
    ):
        """Listen for messages"""
        
        async for message in self.pubsub.listen():
            if message['type'] in ('message', 'pmessage'):
                await handler(message['data'])
    
    async def send_direct(
        self,
        agent_id: UUID,
        message: bytes
    ) -> None:
        """Send via Redis list (inbox queue)"""
        
        inbox_key = f"inbox:{agent_id}"
        await self.redis.lpush(inbox_key, message)
        await self.redis.expire(inbox_key, 86400)  # 24h TTL
```

### 7.3 NATS Transport

```python
class NATSTransport(Transport):
    """NATS-based transport with JetStream persistence"""
    
    def __init__(self, nats_url: str):
        self.nats_url = nats_url
        self.nc: Optional[nats.aio.client.Client] = None
        self.js: Optional[nats.js.JetStreamContext] = None
    
    async def connect(self) -> None:
        """Connect to NATS server"""
        
        self.nc = await nats.connect(self.nats_url)
        self.js = self.nc.jetstream()
    
    async def publish(
        self,
        topic: str,
        message: bytes,
        options: PublishOptions = None
    ) -> None:
        """Publish to NATS subject"""
        
        if options and options.persistent:
            # Use JetStream for persistence
            await self.js.publish(topic, message)
        else:
            # Core NATS for fire-and-forget
            await self.nc.publish(topic, message)
    
    async def subscribe(
        self,
        pattern: str,
        handler: Callable[[bytes], Awaitable[None]],
        options: SubscribeOptions = None
    ) -> Subscription:
        """Subscribe to NATS subject"""
        
        if options and options.durable:
            # Durable consumer with JetStream
            sub = await self.js.subscribe(
                pattern,
                durable=options.durable_name,
                cb=self._wrap_handler(handler)
            )
        else:
            # Ephemeral subscription
            sub = await self.nc.subscribe(
                pattern,
                cb=self._wrap_handler(handler)
            )
        
        return Subscription(
            pattern=pattern,
            cancel=sub.unsubscribe
        )
    
    def _wrap_handler(
        self,
        handler: Callable[[bytes], Awaitable[None]]
    ) -> Callable:
        """Wrap handler for NATS message format"""
        
        async def wrapped(msg):
            await handler(msg.data)
        
        return wrapped
```

---

## 8. Integration with A2A and MCP

### 8.1 A2A Protocol Compatibility

```python
class A2AAdapter:
    """Adapter for Google A2A protocol compatibility"""
    
    def __init__(self, swarm_protocol: SwarmProtocol):
        self.swarm = swarm_protocol
    
    def to_agent_card(self, agent_info: AgentInfo) -> Dict[str, Any]:
        """Convert internal agent info to A2A Agent Card"""
        
        return {
            "name": agent_info.name,
            "description": agent_info.description,
            "url": f"{self.swarm.base_url}/agents/{agent_info.agent_id}",
            "provider": {
                "organization": "SwarmResearch",
                "url": "https://swarmresearch.ai"
            },
            "version": agent_info.version,
            "documentationUrl": f"{self.swarm.base_url}/docs/{agent_info.agent_id}",
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": True
            },
            "authentication": {
                "schemes": ["bearer"]
            },
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
            "skills": [
                self._to_a2a_skill(skill)
                for skill in agent_info.skills
            ]
        }
    
    def _to_a2a_skill(self, skill: Skill) -> Dict[str, Any]:
        """Convert internal skill to A2A skill format"""
        
        return {
            "id": skill.id,
            "name": skill.name,
            "description": skill.description,
            "tags": skill.tags,
            "examples": skill.examples,
            "inputModes": ["text"],
            "outputModes": ["text"]
        }
    
    async def handle_a2a_task(self, a2a_task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming A2A task request"""
        
        # Convert A2A task to internal format
        task = self._from_a2a_task(a2a_task)
        
        # Submit through SwarmResearch protocol
        result = await self.swarm.submit_task(task)
        
        # Convert result back to A2A format
        return self._to_a2a_task_result(result)
```

### 8.2 MCP Protocol Compatibility

```python
class MCPAdapter:
    """Adapter for Anthropic MCP protocol compatibility"""
    
    def __init__(self, swarm_protocol: SwarmProtocol):
        self.swarm = swarm_protocol
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools (MCP format)"""
        
        skills = await self.swarm.registry.list_skills()
        
        return [
            {
                "name": skill.id,
                "description": skill.description,
                "inputSchema": skill.parameters
            }
            for skill in skills
        ]
    
    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Call tool via MCP interface"""
        
        # Find agent with skill
        agents = await self.swarm.router.find_agents([name])
        if not agents:
            raise ToolNotFound(name)
        
        # Invoke skill via RPC
        result = await self.swarm.rpc.call(
            target_agent_id=agents[0].agent_id,
            method="skill.invoke",
            params={"skill_id": name, "arguments": arguments}
        )
        
        # Convert to MCP content format
        return [{
            "type": "text",
            "text": json.dumps(result.result)
        }]
```

---

## 9. Configuration

### 9.1 Protocol Configuration

```yaml
# communication_protocol.yaml

protocol:
  version: "1.0"
  
  # Serialization
  serialization:
    default_format: "json"  # json, msgpack, protobuf
    binary_threshold: 1024   # Use binary for payloads > 1KB
  
  # Compression
  compression:
    enabled: true
    threshold: 1024          # Compress payloads > 1KB
    algorithm: "zstd"        # zstd, gzip, lz4
    level: 3                 # Compression level
  
  # Message envelope
  envelope:
    include_trace_context: true
    ttl_default: 3600        # Default message TTL in seconds
    max_size: 10485760       # Max message size (10MB)
  
  # Routing
  routing:
    default_strategy: "capability"
    consistent_hash_virtual_nodes: 150
  
  # Backpressure
  backpressure:
    queue_size: 10000
    high_water_mark: 0.8
    low_water_mark: 0.3
    enable_load_shedding: true
  
  # Circuit breaker
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 30
    half_open_max_calls: 3

# Transport configuration
transport:
  type: "redis"  # redis, nats, rabbitmq, memory
  
  redis:
    url: "redis://localhost:6379"
    pool_size: 10
    socket_timeout: 5
    socket_connect_timeout: 5
  
  nats:
    url: "nats://localhost:4222"
    max_reconnect_attempts: 10
    reconnect_time_wait: 2
  
  rabbitmq:
    url: "amqp://guest:guest@localhost:5672/"
    connection_attempts: 3
    retry_delay: 5

# Performance tuning
performance:
  # Batching
  batching:
    enabled: true
    max_size: 100
    max_bytes: 1048576  # 1MB
    max_wait_ms: 10
  
  # Connection pooling
  connection_pool:
    min_size: 5
    max_size: 50
    max_idle_time: 300
  
  # Timeouts
  timeouts:
    default_rpc: 30000
    default_stream: 60000
    health_check: 5000
```

---

## 10. Monitoring & Observability

### 10.1 Metrics

```python
class ProtocolMetrics:
    """Protocol-level metrics collection"""
    
    def __init__(self, metrics_client: MetricsClient):
        self.metrics = metrics_client
        
        # Message counters
        self.messages_sent = Counter('swarm_messages_sent_total')
        self.messages_received = Counter('swarm_messages_received_total')
        self.messages_dropped = Counter('swarm_messages_dropped_total')
        
        # Latency histograms
        self.message_latency = Histogram(
            'swarm_message_latency_seconds',
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        
        # Queue metrics
        self.queue_depth = Gauge('swarm_queue_depth')
        self.queue_fullness = Gauge('swarm_queue_fullness_ratio')
        
        # Backpressure metrics
        self.backpressure_events = Counter('swarm_backpressure_events_total')
        self.load_shedded = Counter('swarm_load_shedded_total')
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge('swarm_circuit_breaker_state')
        self.circuit_breaker_trips = Counter('swarm_circuit_breaker_trips_total')
```

### 10.2 Distributed Tracing

```python
class TracingMiddleware:
    """Distributed tracing for message flow"""
    
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
    
    async def trace_send(
        self,
        message: SwarmMessage,
        operation: Callable
    ):
        """Trace message send operation"""
        
        with self.tracer.start_as_current_span(
            name=f"send.{message.message_type}",
            context=message.trace_context
        ) as span:
            span.set_attribute("message.id", str(message.message_id))
            span.set_attribute("message.type", message.message_type)
            span.set_attribute("source.agent", str(message.source.agent_id))
            
            try:
                result = await operation()
                span.set_status(StatusCode.OK)
                return result
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise
```

---

## 11. Summary

### 11.1 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Message Format | JSON + Binary (MsgPack/Protobuf) | Human-readable for debugging, binary for performance |
| Transport | Redis/NATS/RabbitMQ pluggable | Flexibility across deployment scenarios |
| Routing | Topic + Capability + Consistent Hash | Multiple strategies for different use cases |
| Backpressure | Token bucket + Load shedding | Graceful degradation under load |
| Reliability | At-least-once + Idempotency | Balance between reliability and performance |
| A2A/MCP | Adapter pattern | Maintain flexibility while supporting standards |

### 11.2 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| P50 Latency | < 5ms | Within datacenter |
| P99 Latency | < 50ms | Including retries |
| Throughput | > 1000 msg/s/agent | Per agent |
| Scale | 1000+ agents | Per swarm |
| Availability | 99.9% | With redundancy |

### 11.3 Implementation Phases

1. **Phase 1**: Core message format, Redis transport, Direct messaging
2. **Phase 2**: Pub/Sub, Topic routing, Backpressure
3. **Phase 3**: Circuit breakers, Load shedding, Binary serialization
4. **Phase 4**: A2A/MCP adapters, Advanced routing, Multi-region

---

*Document Version: 1.0*
*Last Updated: 2025*
*Protocol Version: 1.0*
