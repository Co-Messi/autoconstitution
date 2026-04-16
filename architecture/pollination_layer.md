# Cross-Pollination Layer Design
## SwarmResearch Shared Findings Bus

**Version:** 1.0  
**Purpose:** Enable controlled information sharing across agent swarm while preventing premature convergence

---

## 1. Data Structure for Shared Findings

### 1.1 Core Finding Record

```python
class SharedFinding:
    """
    Immutable record of a validated improvement discovered by an agent.
    """
    finding_id: UUID              # Unique identifier
    agent_id: str                 # Originating agent
    timestamp: datetime           # Publication time
    
    # Content
    finding_type: FindingType     # IMPROVEMENT | INSIGHT | PATTERN | NEGATIVE_RESULT
    domain: str                   # Research domain/sub-problem
    description: str              # Human-readable summary
    
    # Validation metadata
    validation_score: float       # 0.0 - 1.0 confidence
    sample_size: int              # Number of evaluations
    statistical_significance: float  # p-value or equivalent
    
    # Reproducibility
    reproduction_hash: str        # Hash of conditions for reproduction
    parameters: Dict[str, Any]    # Key parameters that produced result
    
    # Impact tracking
    novelty_score: float          # Estimated novelty (0-1)
    generality_score: float       # Expected applicability (0-1)
    
    # Lifecycle
    ttl_seconds: int              # Time-to-live before decay
    propagation_count: int        # How many times forwarded
    
    # Provenance
    parent_finding_id: Optional[UUID]  # If derived from another finding
    derivation_chain: List[UUID]  # Full ancestry
```

### 1.2 Finding Type Hierarchy

```
FindingType
├── IMPROVEMENT
│   ├── PARAMETER_OPTIMIZATION
│   ├── ARCHITECTURE_CHANGE
│   └── ALGORITHM_MODIFICATION
├── INSIGHT
│   ├── CAUSAL_RELATIONSHIP
│   ├── CORRELATION_DISCOVERY
│   └── BEHAVIORAL_PATTERN
├── PATTERN
│   ├── SUCCESS_PATTERN
│   ├── FAILURE_PATTERN
│   └── RECURRENT_STRUCTURE
└── NEGATIVE_RESULT
    ├── FAILED_APPROACH
    ├── INEFFECTIVE_PARAMETER
    └── DISPROVEN_HYPOTHESIS
```

### 1.3 Contextual Envelope

```python
class FindingEnvelope:
    """
    Wraps findings with routing and filtering metadata.
    """
    finding: SharedFinding
    
    # Routing
    priority: Priority            # CRITICAL | HIGH | NORMAL | LOW
    target_domains: List[str]     # Specific domains (empty = broadcast)
    exclude_agents: List[str]     # Agents that should NOT receive
    
    # Filtering hints
    relevance_tags: List[str]     # For subscriber filtering
    required_capability: str      # Minimum agent capability level
    
    # Rate limiting
    burst_token: UUID             # For rate limit tracking
    
    # Tracing
    hop_count: int                # Number of relays
    path_trace: List[str]         # Agent IDs that forwarded
```

---

## 2. Update Protocol

### 2.1 Publication Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Agent     │────▶│  Validation  │────▶│   Publish   │────▶│  Broadcast   │
│ Discovers   │     │   Pipeline   │     │   to Bus    │     │   to Subs    │
│ Improvement │     │              │     │             │     │              │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ 1. Local     │
                    │    Validation│
                    │ 2. Significance│
                    │    Check     │
                    │ 3. Novelty   │
                    │    Assessment│
                    └──────────────┘
```

### 2.2 Validation Pipeline Stages

| Stage | Purpose | Threshold | Failure Action |
|-------|---------|-----------|----------------|
| Local Validation | Ensure finding is reproducible | 3/3 successful reproductions | Reject, retry |
| Significance Check | Statistical validity | p < 0.05 or equivalent | Queue for more samples |
| Novelty Assessment | Avoid redundant sharing | novelty > 0.3 | Suppress, increment ref count |
| Impact Estimation | Predict usefulness | impact > 0.2 | Archive only, no broadcast |
| Diversity Check | Prevent convergence | diversity_score > 0.4 | Delay, diversify |

### 2.3 Broadcast Semantics

```python
class BroadcastProtocol:
    """
    Defines how findings propagate through the swarm.
    """
    
    async def publish(self, envelope: FindingEnvelope):
        # 1. Apply rate limiting
        if not self.rate_limiter.allow(envelope.burst_token):
            raise RateLimitExceeded()
        
        # 2. Determine recipients
        recipients = self.subscription_manager.match(envelope)
        
        # 3. Apply diversity filtering
        recipients = self.diversity_filter.apply(recipients, envelope)
        
        # 4. Queue for delivery
        for recipient in recipients:
            await self.delivery_queue.put(recipient, envelope)
    
    async def receive(self, agent_id: str) -> Optional[FindingEnvelope]:
        # 1. Check agent's receive quota
        if not self.receive_quota.available(agent_id):
            return None
        
        # 2. Pop from agent's priority queue
        envelope = await self.agent_queues[agent_id].get()
        
        # 3. Apply information decay
        envelope = self.decay_applier.apply(envelope)
        
        # 4. Update consumption metrics
        self.metrics.record_consumption(agent_id, envelope)
        
        return envelope
```

### 2.4 Delivery Guarantees

| Guarantee | Implementation | Trade-off |
|-----------|---------------|-----------|
| At-Least-Once | Ack + retry (3x) | Duplicates possible |
| Priority Order | Per-agent priority queue | Lower priority may starve |
| Time-Bounded | TTL enforcement | Some findings expire unread |
| Selective Delivery | Subscription filtering | Agents may miss relevant info |

---

## 3. Frequency Controls

### 3.1 Token Bucket Rate Limiting

```python
class TokenBucketRateLimiter:
    """
    Per-agent rate limiting using token buckets.
    Prevents any single agent from flooding the bus.
    """
    
    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        
        # Default configuration
        self.config = {
            'bucket_size': 10,          # Max burst
            'refill_rate': 2.0,          # Tokens per second
            'critical_bonus': 5,         # Extra tokens for critical findings
            'cooldown_period': 60,       # Seconds after exhaustion
        }
    
    def allow(self, agent_id: str, priority: Priority) -> bool:
        bucket = self.buckets.get(agent_id)
        if bucket is None:
            bucket = TokenBucket(**self.config)
            self.buckets[agent_id] = bucket
        
        cost = self._priority_cost(priority)
        return bucket.consume(cost)
    
    def _priority_cost(self, priority: Priority) -> int:
        return {
            Priority.CRITICAL: 1,
            Priority.HIGH: 2,
            Priority.NORMAL: 3,
            Priority.LOW: 5,
        }[priority]
```

### 3.2 Global Flood Protection

```python
class GlobalFloodProtector:
    """
    System-wide protection against information flood.
    """
    
    def __init__(self):
        self.global_tokens = 1000      # System-wide burst capacity
        self.refill_rate = 100         # Tokens per second
        self.domain_limits: Dict[str, int] = {}  # Per-domain limits
    
    def check_global_health(self) -> SystemHealth:
        """
        Monitor system-wide information flow.
        """
        metrics = {
            'findings_per_second': self.metrics.rate(),
            'queue_depth': self.delivery_queue.depth(),
            'agent_saturation': self.metrics.saturation(),
        }
        
        if metrics['findings_per_second'] > self.threshold_critical:
            return SystemHealth.OVERLOADED
        elif metrics['agent_saturation'] > 0.8:
            return SystemHealth.CONGESTED
        return SystemHealth.HEALTHY
    
    def apply_backpressure(self):
        """
        When overloaded, increase filtering aggressiveness.
        """
        self.filtering.diversity_threshold += 0.1
        self.rate_limiter.refill_rate *= 0.8
        self.decay_applier.acceleration += 0.5
```

### 3.3 Adaptive Throttling

```python
class AdaptiveThrottler:
    """
    Dynamically adjusts publication rates based on system state.
    """
    
    def calculate_throttle(self, agent_id: str, finding: SharedFinding) -> float:
        """
        Returns throttle factor (0.0 - 1.0) for this publication.
        """
        factors = {
            # Agent's recent publication rate
            'agent_rate': self._agent_rate_factor(agent_id),
            
            # Finding novelty (novel findings pass easier)
            'novelty': finding.novelty_score,
            
            # System load
            'system_load': self._system_load_factor(),
            
            # Domain congestion
            'domain_congestion': self._domain_factor(finding.domain),
            
            # Agent's historical contribution quality
            'reputation': self._reputation_factor(agent_id),
        }
        
        # Weighted combination
        weights = {
            'agent_rate': 0.2,
            'novelty': 0.3,
            'system_load': 0.25,
            'domain_congestion': 0.15,
            'reputation': 0.1,
        }
        
        throttle = sum(factors[k] * weights[k] for k in factors)
        return min(1.0, max(0.0, throttle))
```

### 3.4 Frequency Control Matrix

| Scenario | Control Applied | Effect |
|----------|----------------|--------|
| Agent publishes >5/min | Exponential backoff | Delay increases: 1s, 2s, 4s, 8s... |
| Same finding pattern | Deduplication | Increment ref count, suppress broadcast |
| Low novelty (<0.3) | Archive only | Available on query, not pushed |
| System overloaded | Priority filter | Only CRITICAL/HIGH propagate |
| Domain saturated | Cross-domain delay | Other domains receive first |
| Agent reputation low | Manual review | Held for moderator approval |

---

## 4. Subscription Management

### 4.1 Subscription Model

```python
class Subscription:
    """
    Defines what findings an agent wants to receive.
    """
    subscriber_id: str
    
    # Content filters
    finding_types: Set[FindingType]  # Which types to receive
    domains: Set[str]                # Relevant domains (empty = all)
    min_validation_score: float      # Minimum confidence
    min_novelty: float               # Minimum novelty
    
    # Rate preferences
    max_findings_per_minute: int     # Receive rate limit
    priority_preference: Priority    # Minimum priority
    
    # Diversity preferences
    diversity_window: int            # Remember last N findings
    similarity_threshold: float      # Skip if too similar to recent
    
    # Temporal preferences
    max_age_seconds: int             # Ignore findings older than
    time_preference: TimePreference  # REALTIME | BATCH | DIGEST
    
    # Capability requirements
    required_capability: str         # Only findings agent can use
```

### 4.2 Subscription Matcher

```python
class SubscriptionMatcher:
    """
    Matches findings to interested subscribers.
    """
    
    def match(self, envelope: FindingEnvelope) -> List[str]:
        """
        Returns list of agent IDs that should receive this finding.
        """
        finding = envelope.finding
        recipients = []
        
        for agent_id, sub in self.subscriptions.items():
            # Skip if in exclude list
            if agent_id in envelope.exclude_agents:
                continue
            
            # Type filter
            if finding.finding_type not in sub.finding_types:
                continue
            
            # Domain filter
            if sub.domains and finding.domain not in sub.domains:
                continue
            
            # Quality filters
            if finding.validation_score < sub.min_validation_score:
                continue
            if finding.novelty_score < sub.min_novelty:
                continue
            
            # Priority filter
            if envelope.priority.value < sub.priority_preference.value:
                continue
            
            # Age filter
            age = (datetime.now() - finding.timestamp).seconds
            if age > sub.max_age_seconds:
                continue
            
            # Similarity filter (diversity)
            if self._too_similar(finding, agent_id, sub):
                continue
            
            recipients.append(agent_id)
        
        return recipients
    
    def _too_similar(self, finding: SharedFinding, agent_id: str, 
                     sub: Subscription) -> bool:
        """
        Check if finding is too similar to recently received findings.
        """
        recent = self.recent_findings[agent_id][-sub.diversity_window:]
        
        for recent_finding in recent:
            similarity = self.similarity_calc.calculate(finding, recent_finding)
            if similarity > sub.similarity_threshold:
                return True
        
        return False
```

### 4.3 Dynamic Subscription Adjustment

```python
class DynamicSubscriptionManager:
    """
    Automatically adjusts subscriptions to prevent convergence.
    """
    
    def adjust_for_diversity(self, agent_id: str):
        """
        Periodically adjust subscription to expose agent to diverse information.
        """
        sub = self.subscriptions[agent_id]
        
        # Analyze what agent has been receiving
        history = self.consumption_history[agent_id]
        domain_distribution = history.domain_distribution()
        type_distribution = history.type_distribution()
        
        # If too concentrated in one domain, expand
        if domain_distribution.entropy() < 0.5:
            new_domains = self._suggest_diverse_domains(agent_id)
            sub.domains.update(new_domains)
        
        # If only receiving improvements, add insights
        if type_distribution.get(FindingType.IMPROVEMENT, 0) > 0.8:
            sub.finding_types.add(FindingType.INSIGHT)
            sub.finding_types.add(FindingType.PATTERN)
        
        # If reputation is high, lower thresholds to see more experimental work
        if self.reputation[agent_id] > 0.8:
            sub.min_validation_score = max(0.5, sub.min_validation_score - 0.1)
    
    def _suggest_diverse_domains(self, agent_id: str) -> Set[str]:
        """
        Suggest domains that might provide useful cross-pollination.
        """
        current_domains = self.subscriptions[agent_id].domains
        
        # Find domains with high success transfer rate to current domains
        candidates = self.domain_transfer_matrix.successful_transfers(current_domains)
        
        # Filter to those with active research
        candidates = [d for d in candidates if self.domain_activity[d] > 0.3]
        
        return set(candidates[:3])  # Suggest top 3
```

### 4.4 Subscription Lifecycle

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│   CREATE    │────▶│   ACTIVE     │────▶│   UPDATE     │────▶│   EXPIRE    │
│  (on join)  │     │  (receiving) │     │  (periodic)  │     │ (on leave)  │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
                           │                                        │
                           │         ┌──────────────┐              │
                           └────────▶│   PAUSE      │──────────────┘
                                     │ (on overload)│
                                     └──────────────┘
```

---

## 5. Information Decay Mechanisms

### 5.1 Decay Model

```python
class InformationDecay:
    """
    Models how finding relevance decreases over time.
    Prevents old findings from dominating agent behavior.
    """
    
    def __init__(self):
        self.decay_strategies = {
            FindingType.IMPROVEMENT: ExponentialDecay(half_life=3600),
            FindingType.INSIGHT: ExponentialDecay(half_life=7200),
            FindingType.PATTERN: LinearDecay(rate=0.1),
            FindingType.NEGATIVE_RESULT: StepDecay(steps=[3600, 7200, 14400]),
        }
    
    def apply_decay(self, finding: SharedFinding, 
                    current_time: datetime) -> DecayedFinding:
        """
        Apply time-based decay to a finding.
        """
        age_seconds = (current_time - finding.timestamp).seconds
        strategy = self.decay_strategies[finding.finding_type]
        
        decay_factor = strategy.calculate(age_seconds)
        
        return DecayedFinding(
            original=finding,
            current_relevance=finding.validation_score * decay_factor,
            decay_factor=decay_factor,
            age_seconds=age_seconds,
        )
```

### 5.2 Decay Strategies

| Strategy | Formula | Best For |
|----------|---------|----------|
| Exponential | relevance = original × e^(-λt) | Rapidly evolving domains |
| Linear | relevance = original - (rate × t) | Stable domains |
| Step | relevance drops at thresholds | Milestone-based findings |
| Adaptive | relevance based on successor count | Areas with rapid progress |

### 5.3 Successor-Based Decay

```python
class SuccessorBasedDecay:
    """
    Accelerates decay when better findings emerge.
    Critical for preventing premature convergence.
    """
    
    def __init__(self):
        self.finding_graph: Dict[UUID, List[UUID]] = {}  # parent -> children
        self.finding_scores: Dict[UUID, float] = {}
    
    def register_successor(self, parent_id: UUID, child: SharedFinding):
        """
        Register that a new finding builds on or improves a previous one.
        """
        if parent_id not in self.finding_graph:
            self.finding_graph[parent_id] = []
        self.finding_graph[parent_id].append(child.finding_id)
        self.finding_scores[child.finding_id] = child.validation_score
    
    def get_effective_relevance(self, finding_id: UUID) -> float:
        """
        Calculate relevance considering successors.
        """
        base_relevance = self._time_decay(finding_id)
        
        # Check if successors have superseded this finding
        successors = self.finding_graph.get(finding_id, [])
        if not successors:
            return base_relevance
        
        best_successor_score = max(
            self.finding_scores[sid] for sid in successors
        )
        
        # If successors are significantly better, accelerate decay
        if best_successor_score > self.finding_scores[finding_id] * 1.2:
            supersession_factor = 0.5  # 50% reduction
        else:
            supersession_factor = 1.0
        
        return base_relevance * supersession_factor
```

### 5.4 Convergence Detection & Mitigation

```python
class ConvergenceMonitor:
    """
    Detects and mitigates premature convergence in the swarm.
    """
    
    def __init__(self):
        self.agent_states: Dict[str, AgentState] = {}
        self.convergence_threshold = 0.7
    
    def measure_diversity(self) -> DiversityMetrics:
        """
        Measure current diversity across the swarm.
        """
        # Collect agent parameter distributions
        parameter_distributions = self._collect_parameters()
        
        # Calculate entropy across key dimensions
        metrics = DiversityMetrics(
            parameter_entropy=self._entropy(parameter_distributions),
            approach_diversity=self._approach_diversity(),
            solution_diversity=self._solution_diversity(),
            exploration_ratio=self._exploration_ratio(),
        )
        
        return metrics
    
    def detect_convergence(self) -> Optional[ConvergenceWarning]:
        """
        Detect if swarm is converging prematurely.
        """
        metrics = self.measure_diversity()
        
        warnings = []
        
        if metrics.parameter_entropy < 0.3:
            warnings.append("Low parameter diversity detected")
        
        if metrics.approach_diversity < 0.4:
            warnings.append("Approach homogenization detected")
        
        if metrics.exploration_ratio < 0.2:
            warnings.append("Exploration critically low")
        
        if warnings:
            return ConvergenceWarning(
                severity=len(warnings),
                messages=warnings,
                metrics=metrics,
            )
        
        return None
    
    def mitigate_convergence(self, warning: ConvergenceWarning):
        """
        Apply countermeasures to restore diversity.
        """
        # 1. Increase information decay rate
        self.decay_applier.acceleration = 2.0
        
        # 2. Boost exploration incentives
        for agent_id in self.agent_states:
            self.agent_states[agent_id].exploration_bonus = 1.5
        
        # 3. Filter out overly popular findings temporarily
        self.subscription_manager.suppress_popular_findings(
            popularity_threshold=0.6,
            duration_seconds=300
        )
        
        # 4. Inject diversity signals
        self._broadcast_diversity_challenge()
        
        # 5. Temporarily partition communication
        self._create_information_silos(duration_seconds=600)
```

### 5.5 Decay Control Matrix

| Condition | Decay Action | Rationale |
|-----------|--------------|-----------|
| Finding age > 1 hour | Normal decay | Standard lifecycle |
| Finding age > 4 hours | Accelerated decay | Prevent stale info |
| Better successor found | Immediate 50% reduction | Superseded findings |
| Convergence detected | Global 2x acceleration | Restore diversity |
| Domain inactive | Gradual decay | Clear unused info |
| Finding referenced often | Slower decay | Valuable reference |
| Negative result | Step decay at 1h, 2h, 4h | Keep initially, fade slowly |

---

## 6. Anti-Convergence Architecture

### 6.1 Diversity Preservation Strategies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ANTI-CONVERGENCE SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   NICHING    │    │   ISLAND     │    │   DIVERSITY  │              │
│  │   MECHANISM  │    │   MODEL      │    │   INJECTION  │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ Maintain     │    │ Periodic     │    │ Introduce    │              │
│  │ sub-populations│   │ migration    │    │ random       │              │
│  │ with limited │    │ between      │    │ perturbations│              │
│  │ cross-talk   │    │ groups       │    │              │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Niching Configuration

```python
class NichingConfiguration:
    """
    Divides swarm into sub-populations with controlled interaction.
    """
    
    num_niches: int = 5
    niche_assignment: str = 'capability_based'  # or 'random', 'domain_based'
    
    # Cross-niche communication
    inter_niche_rate: float = 0.2  # 20% of findings cross niches
    inter_niche_delay: int = 300   # 5 minute delay for cross-niche
    
    # Migration
    migration_interval: int = 1800  # 30 minutes
    migration_size: int = 2         # Agents per migration
```

### 6.3 Controlled Information Flow

```python
class ControlledInformationFlow:
    """
    Manages how information spreads to prevent rapid convergence.
    """
    
    def should_propagate(self, finding: SharedFinding, 
                         from_agent: str, to_agent: str) -> bool:
        """
        Determine if a finding should propagate between specific agents.
        """
        # Same niche: always allow
        if self._same_niche(from_agent, to_agent):
            return True
        
        # Different niches: apply controls
        controls = {
            # Delay cross-niche propagation
            'delay': self._cross_niche_delay(finding),
            
            # Require higher validation for cross-niche
            'validation_threshold': 0.8,
            
            # Limit how many niches receive
            'niche_limit': 3,
            
            # Diversity check
            'diversity_required': True,
        }
        
        return all([
            finding.validation_score >= controls['validation_threshold'],
            self._niche_propagation_count(finding) < controls['niche_limit'],
            not self._would_reduce_diversity(finding, to_agent),
        ])
```

---

## 7. Implementation Reference

### 7.1 Message Flow Summary

```
Agent Discovery
      │
      ▼
┌─────────────┐
│  Validate   │──▶ Local reproduction (3x)
│  (internal) │──▶ Significance test
└──────┬──────┘──▶ Novelty check
       │
       ▼
┌─────────────┐
│   Publish   │──▶ Rate limit check
│   Request   │──▶ Priority assignment
└──────┬──────┘──▶ Envelope creation
       │
       ▼
┌─────────────┐
│   Route &   │──▶ Subscription matching
│   Filter    │──▶ Diversity filtering
└──────┬──────┘──▶ Recipient selection
       │
       ▼
┌─────────────┐
│   Queue &   │──▶ Priority queuing per agent
│   Deliver   │──▶ Delivery with decay applied
└─────────────┘──▶ Consumption tracking
```

### 7.2 Configuration Parameters

```yaml
pollination_layer:
  # Rate limiting
  rate_limit:
    bucket_size: 10
    refill_rate: 2.0
    cooldown_period: 60
  
  # Decay
  decay:
    default_half_life: 3600
    convergence_acceleration: 2.0
    successor_supersession: 0.5
  
  # Diversity
  diversity:
    similarity_threshold: 0.7
    window_size: 20
    min_entropy: 0.5
  
  # Convergence detection
  convergence:
    check_interval: 300
    entropy_threshold: 0.3
    exploration_min: 0.2
  
  # Niching
  niching:
    enabled: true
    num_niches: 5
    inter_niche_rate: 0.2
    migration_interval: 1800
```

### 7.3 Monitoring Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| findings_per_minute | Publication rate | 10-50 |
| average_queue_depth | Delivery backlog | <100 |
| diversity_entropy | Swarm diversity | >0.5 |
| convergence_index | Convergence risk | <0.3 |
| decay_applications | Decay events/min | 5-20 |
| cross_niche_ratio | Inter-niche flow | 0.1-0.3 |

---

## 8. Summary

The Cross-Pollination Layer provides a robust shared findings bus with multiple safeguards against premature convergence:

1. **Structured Data**: Rich finding records with validation, novelty, and provenance metadata
2. **Controlled Propagation**: Multi-stage validation and adaptive rate limiting
3. **Frequency Controls**: Token buckets, global flood protection, and adaptive throttling
4. **Smart Subscriptions**: Dynamic matching with diversity enforcement
5. **Information Decay**: Time-based and successor-based relevance reduction
6. **Anti-Convergence**: Niching, controlled flow, and convergence detection/mitigation

This design enables effective information sharing while maintaining swarm diversity for robust exploration.
