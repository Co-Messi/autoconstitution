# SwarmResearch Observability System Design

## Executive Summary

This document defines the comprehensive observability architecture for SwarmResearch, a massively parallel multi-agent AI research system. The observability system provides deep visibility into agent behavior, system performance, research progress, and operational health across all deployment tiers—from a single Mac Mini M4 to H100 GPU clusters.

---

## 1. Observability Architecture Overview

### 1.1 Design Principles

The observability system follows three core principles:

1. **Three Pillars (Unified View)**: Metrics (Numbers), Logs (Events), and Traces (Journeys) all tied together with unified context (trace_id)

2. **Observability-Driven Development**: Every component emits observable signals by default, instrumentation is built-in, debuggability is first-class

3. **Progressive Observability**:
   - Development: Console + File logging
   - Small Production: + Prometheus metrics
   - Medium Scale: + Distributed tracing, centralized logs
   - Large Scale: + Real-time analytics, ML-based anomaly detection

### 1.2 Observability Stack by Deployment Tier

| Tier | Environment | Metrics | Logs | Tracing | Dashboard | Alerts |
|------|-------------|---------|------|---------|-----------|--------|
| 1 | Development (Mac Mini M4) | prometheus_client (in-process) | structlog → stdout/file | OpenTelemetry (console) | CLI + local Grafana | Console warnings |
| 2 | Small Production | Prometheus (local) + Grafana | structlog → Loki (local) | OpenTelemetry → Jaeger (local) | Grafana unified | Alertmanager (email/Slack) |
| 3 | Medium Scale (K8s) | Prometheus (HA) + Grafana | Fluent Bit → Loki | OTel Collector → Jaeger/Tempo | Grafana multi-tenant | Alertmanager + PagerDuty |
| 4 | Large Scale (H100 Cluster) | Prometheus (federated) + Thanos/Cortex | Vector/Fluentd → Kafka → ClickHouse/Loki | OpenTelemetry → Jaeger (Cassandra/ES) | Grafana global + ML insights | Alertmanager + auto-remediation |

---

## 2. Metrics Collection System

### 2.1 Metrics Architecture

```
Agent Workers → OpenTelemetry SDK → Prometheus Exporter → Prometheus/Thanos
Orchestrator  → (Counter, Gauge, Histogram, UpDownCounter)
Provider Layer
```

### 2.2 Core Metrics Catalog

#### Agent-Level Metrics

```python
# Agent execution metrics
swarm_agent_iterations_total      # Counter[agent_id, agent_type, branch_id, status]
swarm_agent_iteration_duration_seconds  # Histogram[agent_id, agent_type]
swarm_agent_tokens_total          # Counter[agent_id, agent_type, provider, token_type]
swarm_agent_tool_calls_total      # Counter[agent_id, tool_name, status]
swarm_agent_tool_duration_seconds # Histogram[tool_name]
swarm_agent_state                 # Gauge[agent_id, agent_type, state]
swarm_agent_knowledge_contributions_total  # Counter[agent_id, contribution_type]
swarm_agent_gradient_score        # Gauge[agent_id, branch_id]
```

#### Orchestrator-Level Metrics

```python
swarm_active_agents               # Gauge[orchestrator_id, branch_id]
swarm_branch_count                # Gauge[orchestrator_id, branch_status]
swarm_queue_depth                 # Gauge[orchestrator_id, queue_name, priority]
swarm_reallocations_total         # Counter[orchestrator_id, source_branch, target_branch, reason]
swarm_events_total                # Counter[orchestrator_id, event_type]
swarm_ratchet_best_score          # Gauge[orchestrator_id]
swarm_ratchet_version             # Gauge[orchestrator_id]
swarm_ratchet_update_latency_seconds  # Histogram[orchestrator_id]
```

#### Provider-Level Metrics

```python
swarm_provider_requests_total     # Counter[provider, model, endpoint, status]
swarm_provider_latency_seconds    # Histogram[provider, model, endpoint]
swarm_provider_tokens_total       # Counter[provider, model, token_type]
swarm_provider_cost_usd           # Counter[provider, model]
swarm_provider_errors_total       # Counter[provider, error_type, status_code]
swarm_provider_rate_limit_hits_total  # Counter[provider]
swarm_provider_health             # Gauge[provider]
swarm_provider_throughput_tps     # Gauge[provider, model]
```

#### Research-Specific Metrics

```python
swarm_research_hypotheses_total   # Counter[orchestrator_id, branch_id, status]
swarm_research_findings_total     # Counter[orchestrator_id, branch_id, finding_type]
swarm_research_convergence_score  # Gauge[orchestrator_id, branch_id]
swarm_research_depth              # Gauge[orchestrator_id, branch_id]
swarm_research_coverage_percent   # Gauge[orchestrator_id]
swarm_research_budget_used_percent  # Gauge[orchestrator_id]
```

### 2.3 Metrics Collection Implementation

```python
# metrics_collector.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from contextlib import contextmanager
import time

class SwarmMetricsCollector:
    """Centralized metrics collection for SwarmResearch."""
    
    def __init__(self, service_name: str = "swarm-research"):
        self.service_name = service_name
        self.registry = CollectorRegistry()
        self._init_metrics()
    
    def _init_metrics(self):
        # Agent metrics
        self.agent_iterations = Counter(
            'swarm_agent_iterations_total',
            'Total agent iterations',
            ['agent_id', 'agent_type', 'status'],
            registry=self.registry
        )
        self.agent_duration = Histogram(
            'swarm_agent_iteration_duration_seconds',
            'Agent iteration duration',
            ['agent_id', 'agent_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
            registry=self.registry
        )
        self.agent_tokens = Counter(
            'swarm_agent_tokens_total',
            'Token usage by agent',
            ['agent_id', 'provider', 'token_type'],
            registry=self.registry
        )
        # Provider metrics
        self.provider_latency = Histogram(
            'swarm_provider_latency_seconds',
            'Provider request latency',
            ['provider', 'model', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        self.provider_requests = Counter(
            'swarm_provider_requests_total',
            'Total provider requests',
            ['provider', 'model', 'status'],
            registry=self.registry
        )
        # Swarm metrics
        self.active_agents = Gauge(
            'swarm_active_agents',
            'Number of active agents',
            ['orchestrator_id'],
            registry=self.registry
        )
        self.queue_depth = Gauge(
            'swarm_queue_depth',
            'Task queue depth',
            ['orchestrator_id', 'priority'],
            registry=self.registry
        )
    
    @contextmanager
    def timed_iteration(self, agent_id: str, agent_type: str):
        """Context manager for timing agent iterations."""
        start_time = time.time()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            self.agent_duration.labels(agent_id=agent_id, agent_type=agent_type).observe(duration)
            self.agent_iterations.labels(agent_id=agent_id, agent_type=agent_type, status=status).inc()
    
    def record_token_usage(self, agent_id: str, provider: str, input_tokens: int, output_tokens: int):
        """Record token usage for an agent."""
        self.agent_tokens.labels(agent_id=agent_id, provider=provider, token_type="input").inc(input_tokens)
        self.agent_tokens.labels(agent_id=agent_id, provider=provider, token_type="output").inc(output_tokens)
    
    @contextmanager
    def timed_provider_request(self, provider: str, model: str, endpoint: str):
        """Context manager for timing provider requests."""
        start_time = time.time()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            self.provider_latency.labels(provider=provider, model=model, endpoint=endpoint).observe(duration)
            self.provider_requests.labels(provider=provider, model=model, status=status).inc()
```

---

## 3. Logging Strategy

### 3.1 Log Levels and Usage

| Level | When to Use | Examples |
|-------|-------------|----------|
| DEBUG | Detailed debugging (dev only) | Function entry/exit, variable values |
| INFO | Normal operational events | Agent spawned, task completed, ratchet updated |
| WARNING | Unexpected but handled | Rate limit hit, retry attempted, queue backlog |
| ERROR | Failed operations | Agent crashed, provider error, task failed |
| CRITICAL | System-wide failures | All providers down, out of memory, disk full |

### 3.2 Structured Log Schema

```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "logger": "swarm.agent_worker",
  "service": "swarm-research",
  "version": "1.2.3",
  "environment": "production",
  "trace_id": "abc123def456",
  "span_id": "span789",
  "correlation_id": "req-xyz789",
  "orchestrator_id": "orch-001",
  "agent_id": "agent-123",
  "agent_type": "explorer",
  "branch_id": "branch-456",
  "event": {
    "type": "agent_iteration_completed",
    "duration_ms": 1500,
    "tokens_used": 2500,
    "provider": "claude",
    "model": "claude-3-5-sonnet-20241022",
    "status": "success"
  },
  "error": {
    "type": "ProviderError",
    "message": "Rate limit exceeded",
    "context": {"provider": "openai", "retry_count": 3}
  }
}
```

### 3.3 Logging Implementation

```python
# logging_config.py
import structlog
import logging
import sys

def configure_logging(level: str = "INFO", environment: str = "production"):
    """Configure structured logging for SwarmResearch."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level))
    
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    if environment == "development":
        structlog.configure(
            processors=shared_processors + [structlog.dev.ConsoleRenderer(colors=True)],
            wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        structlog.configure(
            processors=shared_processors + [
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

# Agent logger mixin
class AgentLoggerMixin:
    """Mixin to add structured logging to agents."""
    
    @property
    def logger(self):
        if not hasattr(self, '_logger') or self._logger is None:
            base_logger = structlog.get_logger()
            self._logger = base_logger.bind(
                agent_id=getattr(self, 'agent_id', 'unknown'),
                agent_type=getattr(self, 'agent_type', 'unknown'),
                branch_id=getattr(self, 'branch_id', 'unknown'),
                orchestrator_id=getattr(self, 'orchestrator_id', 'unknown')
            )
        return self._logger
    
    def log_iteration_start(self, iteration: int, hypothesis: str):
        self.logger.info("agent_iteration_started", event={
            "type": "agent_iteration_started",
            "iteration": iteration,
            "hypothesis": hypothesis[:100] + "..." if len(hypothesis) > 100 else hypothesis
        })
    
    def log_iteration_complete(self, iteration: int, duration_ms: float, tokens_used: int, result: dict):
        self.logger.info("agent_iteration_completed", event={
            "type": "agent_iteration_completed",
            "iteration": iteration,
            "duration_ms": duration_ms,
            "tokens_used": tokens_used,
            "result_summary": self._summarize_result(result)
        })
    
    def log_error(self, error: Exception, context: dict = None):
        self.logger.error("agent_error", event={
            "type": "agent_error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }, exc_info=True)
```

### 3.4 Log Retention Policy

```yaml
retention_policy:
  hot:  # Loki - fast search
    retention: 7d
    max_size: 100GB
    indices: [level, service, agent_id, orchestrator_id, trace_id]
  
  warm:  # ClickHouse - analytics
    retention: 90d
    max_size: 1TB
    aggregation: [hourly_metrics, daily_summaries, error_patterns]
  
  cold:  # S3 - archive
    retention: 1y
    compression: gzip
    partitioning: [by_date: YYYY/MM/DD, by_service]
    lifecycle:
      - transition_to_glacier: 90d
      - expire: 365d
```

---

## 4. Distributed Tracing

### 4.1 Trace Flow Example

```
User Request
    └── API Gateway (span: api_request, 30s)
        └── Orchestrator (span: orchestrate_research, 29.5s)
            ├── Decomposer (span: decompose_problem, 2s)
            ├── Agent Spawner (span: spawn_agents)
            │   ├── Agent 1 (span: agent_1_iteration)
            │   │   ├── LLM Call (span: llm_call, claude)
            │   │   └── Tool Call (span: tool_call, search)
            │   ├── Agent 2 (span: agent_2_iteration)
            │   └── Agent 3 (span: agent_3_iteration)
            ├── Aggregator (span: aggregate_results, 1s)
            └── Formatter (span: format_response, 0.5s)
```

### 4.2 Tracing Implementation

```python
# tracing_config.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from contextlib import contextmanager

class SwarmTracing:
    """Distributed tracing configuration for SwarmResearch."""
    
    def __init__(self, service_name: str = "swarm-research", otlp_endpoint: str = "http://localhost:4317", sampling_rate: float = 1.0):
        self.service_name = service_name
        set_global_textmap(CompositePropagator([TraceContextTextMapPropagator()]))
        
        provider = TracerProvider(sampler=trace.sampling.TraceIdRatioBased(sampling_rate))
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        span_processor = BatchSpanProcessor(otlp_exporter, max_queue_size=2048, max_export_batch_size=512)
        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(service_name)
    
    def start_research_trace(self, research_id: str, problem_statement: str, attributes: dict = None):
        """Start a root trace for a research request."""
        return self.tracer.start_as_current_span(
            name="research_request",
            kind=trace.SpanKind.SERVER,
            attributes={
                "research.id": research_id,
                "research.problem": problem_statement[:100],
                "service.name": self.service_name,
                **(attributes or {})
            }
        )
    
    @contextmanager
    def agent_span(self, agent_id: str, agent_type: str, branch_id: str, iteration: int = None):
        """Context manager for agent execution span."""
        attributes = {"agent.id": agent_id, "agent.type": agent_type, "branch.id": branch_id}
        if iteration is not None:
            attributes["agent.iteration"] = iteration
        with self.tracer.start_as_current_span(
            name=f"agent.{agent_type}.execute",
            kind=trace.SpanKind.INTERNAL,
            attributes=attributes
        ) as span:
            yield span
    
    @contextmanager
    def provider_span(self, provider: str, model: str, operation: str):
        """Context manager for LLM provider call span."""
        with self.tracer.start_as_current_span(
            name=f"llm.{provider}.{operation}",
            kind=trace.SpanKind.CLIENT,
            attributes={"llm.provider": provider, "llm.model": model, "llm.operation": operation}
        ) as span:
            start_time = time.time()
            try:
                yield span
                span.set_attribute("llm.success", True)
            except Exception as e:
                span.set_attribute("llm.success", False)
                span.set_attribute("error.type", type(e).__name__)
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise
            finally:
                span.set_attribute("llm.duration_ms", (time.time() - start_time) * 1000)
```

### 4.3 Sampling Strategy

```python
# sampling_config.py
from opentelemetry.sdk.trace.sampling import Sampler, SamplingResult, Decision, ParentBasedTraceIdRatio

class SwarmAdaptiveSampler(Sampler):
    """Adaptive sampling based on request characteristics."""
    
    def __init__(self, base_rate: float = 0.1, error_rate: float = 1.0, slow_request_threshold_ms: float = 5000):
        self.base_rate = base_rate
        self.error_rate = error_rate
        self.slow_request_threshold_ms = slow_request_threshold_ms
        self._parent_based = ParentBasedTraceIdRatio(base_rate)
    
    def should_sample(self, parent_context, trace_id, name, kind, attributes, links, trace_state):
        # Always sample errors
        if attributes and attributes.get("error"):
            return SamplingResult(Decision.RECORD_AND_SAMPLE)
        # Always sample research requests (root spans)
        if name == "research_request":
            return SamplingResult(Decision.RECORD_AND_SAMPLE)
        # Sample slow requests at higher rate
        if attributes and attributes.get("duration_ms", 0) > self.slow_request_threshold_ms:
            return self._sample_at_rate(self.error_rate, trace_id)
        return self._parent_based.should_sample(parent_context, trace_id, name, kind, attributes, links, trace_state)

# Sampling configuration by environment
SAMPLING_CONFIG = {
    "development": {"sampler": "always_on", "rate": 1.0},
    "staging": {"sampler": "adaptive", "base_rate": 0.5, "error_rate": 1.0},
    "production": {"sampler": "adaptive", "base_rate": 0.1, "error_rate": 1.0}
}
```


---

## 5. Dashboard Requirements

### 5.1 Core Dashboards

#### System Overview Dashboard

Key panels:
- **Active Agents** (stat): `swarm_active_agents{orchestrator_id=~"$orchestrator"}`
- **Queue Depth** (stat): `swarm_queue_depth{orchestrator_id=~"$orchestrator"}`
- **Research Progress** (gauge): `swarm_research_budget_used_percent`
- **Best Score** (stat): `swarm_ratchet_best_score`
- **Agent Iterations** (graph): `rate(swarm_agent_iterations_total[5m])`
- **Provider Latency p99** (graph): `histogram_quantile(0.99, rate(swarm_provider_latency_seconds_bucket[5m]))`
- **Provider Health Status** (stat): `swarm_provider_health`

#### Research Progress Dashboard

Key panels:
- **Best Score Over Time** (graph): `swarm_ratchet_best_score`
- **Score Improvement Rate** (graph): `rate(swarm_ratchet_best_score[5m])`
- **Branch Status Distribution** (pie): `swarm_branch_count`
- **Branch Gradient Scores** (graph): `swarm_agent_gradient_score`
- **Hypotheses Generated** (graph): `rate(swarm_research_hypotheses_total[5m])`
- **Convergence Score** (gauge): `swarm_research_convergence_score`

#### Provider Performance Dashboard

Key panels:
- **Requests per Second** (graph): `rate(swarm_provider_requests_total[1m])`
- **Latency Percentiles** (graph): p50, p95, p99
- **Error Rate by Provider** (graph): `rate(swarm_provider_errors_total[5m])`
- **Cost per Provider** (graph): `swarm_provider_cost_usd`
- **Token Usage** (graph): `rate(swarm_provider_tokens_total[5m])`

#### Agent Detail Dashboard

Key panels:
- **Agent State Timeline** (graph): `swarm_agent_state{agent_id=~"$agent"}`
- **Iteration Duration Heatmap**: `swarm_agent_iteration_duration_seconds_bucket`
- **Tool Calls** (graph): `rate(swarm_agent_tool_calls_total{agent_id=~"$agent"}[5m])`
- **Recent Logs** (logs panel): Loki query `{agent_id="$agent"}`
- **Trace Timeline** (traces panel): Jaeger query `agent_id="$agent"`

### 5.2 Custom Research Visualizations

```python
# Custom Grafana panels for SwarmResearch

KNOWLEDGE_GRAPH_PANEL = {
    "type": "nodeGraph",
    "title": "Knowledge Graph",
    "query": """
        SELECT node_id as id, node_type, node_label as title, confidence as mainStat
        FROM knowledge_nodes WHERE orchestrator_id = '$orchestrator'
    """,
    "edges_query": """
        SELECT source_id as source, target_id as target, relation_type, strength
        FROM knowledge_edges WHERE orchestrator_id = '$orchestrator'
    """
}

BRANCH_TREE_PANEL = {
    "type": "flamegraph",
    "title": "Research Branch Tree",
    "query": """
        SELECT branch_id, parent_branch_id, focus_area, gradient_score, agent_count
        FROM research_branches WHERE orchestrator_id = '$orchestrator'
    """
}

HYPOTHESIS_EVOLUTION_PANEL = {
    "type": "heatmap",
    "title": "Hypothesis Score Evolution",
    "query": """
        SELECT timestamp, hypothesis_id, score
        FROM hypothesis_history WHERE orchestrator_id = '$orchestrator'
    """
}

RESEARCH_STREAM_PANEL = {
    "type": "logs",
    "title": "Live Research Activity",
    "expr": '{orchestrator_id="$orchestrator"} | json | event_type=~"hypothesis_generated|finding_recorded|ratchet_updated"'
}
```

---

## 6. Alerting System

### 6.1 Critical Alerts (Immediate Response)

```yaml
# alerts_critical.yaml
groups:
  - name: swarm_critical
    rules:
      - alert: AllProvidersDown
        expr: sum(swarm_provider_health) == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "All LLM providers are down"
          description: "No healthy providers available. Research system cannot function."
          runbook_url: "https://wiki/runbooks/all-providers-down"
      
      - alert: HighErrorRate
        expr: |
          sum(rate(swarm_provider_errors_total[5m])) / 
          sum(rate(swarm_provider_requests_total[5m])) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"
      
      - alert: AgentCrashLoop
        expr: rate(swarm_agent_iterations_total{status="error"}[5m]) > 0.5
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "Agent {{ $labels.agent_id }} is in crash loop"
          description: "Agent is failing iterations at high rate"
      
      - alert: OutOfMemory
        expr: |
          swarm_system_memory_bytes{type="used"} / 
          (swarm_system_memory_bytes{type="used"} + swarm_system_memory_bytes{type="free"}) > 0.95
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Instance {{ $labels.instance }} is out of memory"
          description: "Memory usage is above 95%"
```

### 6.2 Warning Alerts (Investigation Required)

```yaml
# alerts_warning.yaml
groups:
  - name: swarm_warning
    rules:
      - alert: SlowProviderResponse
        expr: histogram_quantile(0.95, rate(swarm_provider_latency_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Provider {{ $labels.provider }} is responding slowly"
          description: "p95 latency is {{ $value }}s"
      
      - alert: QueueBacklog
        expr: swarm_queue_depth > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Task queue has significant backlog"
          description: "Queue depth is {{ $value }} tasks"
      
      - alert: HighReallocationRate
        expr: rate(swarm_reallocations_total[5m]) > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High agent reallocation rate"
          description: "Agents are being frequently reallocated, possible instability"
      
      - alert: StagnantResearch
        expr: time() - swarm_ratchet_version > 1800
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Research has not progressed in 30 minutes"
          description: "No improvements to global ratchet state"
      
      - alert: CostSpike
        expr: rate(swarm_provider_cost_usd[1h]) > 10
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Unusual cost spike detected"
          description: "Cost is {{ $value }}/hour, higher than normal"
```

### 6.3 Alert Routing

```yaml
# alertmanager_config.yaml
global:
  smtp_smarthost: 'smtp.example.com:587'
  smtp_from: 'alerts@swarmresearch.io'
  slack_api_url: '${SLACK_WEBHOOK_URL}'

route:
  group_by: ['alertname', 'orchestrator_id', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default'
  
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true
    - match:
        severity: warning
      receiver: 'slack-warning'
      continue: true
    - match:
        severity: info
      receiver: 'slack-info'

receivers:
  - name: 'default'
    email_configs:
      - to: 'oncall@swarmresearch.io'
  
  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '${PAGERDUTY_SERVICE_KEY}'
        severity: critical
  
  - name: 'slack-warning'
    slack_configs:
      - channel: '#swarm-alerts'
        title: 'Warning: {{ .GroupLabels.alertname }}'
        color: 'warning'
  
  - name: 'slack-info'
    slack_configs:
      - channel: '#swarm-info'
        title: '{{ .GroupLabels.alertname }}'
        color: 'good'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'orchestrator_id']
```

### 6.4 Auto-Remediation

```python
# auto_remediation.py
from enum import Enum
from dataclasses import dataclass

class RemediationAction(Enum):
    RESTART_AGENT = "restart_agent"
    SWITCH_PROVIDER = "switch_provider"
    SCALE_WORKERS = "scale_workers"
    CLEAR_QUEUE = "clear_queue"
    NOTIFY_TEAM = "notify_team"

@dataclass
class RemediationResult:
    action: RemediationAction
    success: bool
    message: str
    context: dict

class AutoRemediation:
    """Auto-remediation system for SwarmResearch alerts."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.remediation_map = {
            'AgentCrashLoop': RemediationAction.RESTART_AGENT,
            'SlowProviderResponse': RemediationAction.SWITCH_PROVIDER,
            'QueueBacklog': RemediationAction.SCALE_WORKERS,
            'HighReallocationRate': RemediationAction.NOTIFY_TEAM,
        }
    
    async def handle_alert(self, alert: dict) -> RemediationResult:
        alert_name = alert.get('alertname')
        action = self.remediation_map.get(alert_name)
        
        if action == RemediationAction.RESTART_AGENT:
            return await self._restart_agent(alert)
        elif action == RemediationAction.SWITCH_PROVIDER:
            return await self._switch_provider(alert)
        elif action == RemediationAction.SCALE_WORKERS:
            return await self._scale_workers(alert)
        
        return None
    
    async def _restart_agent(self, alert: dict) -> RemediationResult:
        agent_id = alert.get('labels', {}).get('agent_id')
        try:
            agent = self.orchestrator.agents.get(agent_id)
            if not agent:
                return RemediationResult(action=RemediationAction.RESTART_AGENT, success=False, message=f"Agent {agent_id} not found", context={})
            
            config = agent.config
            branch_id = alert.get('labels', {}).get('branch_id')
            await self.orchestrator.terminate_agent(agent_id)
            new_agent_id = await self.orchestrator.spawn_agent(config, branch_id)
            
            return RemediationResult(
                action=RemediationAction.RESTART_AGENT,
                success=True,
                message=f"Agent {agent_id} restarted as {new_agent_id}",
                context={'old_agent_id': agent_id, 'new_agent_id': new_agent_id}
            )
        except Exception as e:
            return RemediationResult(action=RemediationAction.RESTART_AGENT, success=False, message=str(e), context={'agent_id': agent_id})
    
    async def _switch_provider(self, alert: dict) -> RemediationResult:
        provider = alert.get('labels', {}).get('provider')
        try:
            await self.orchestrator.health_monitor.mark_unhealthy(provider)
            new_provider = await self.orchestrator.provider_router.failover(provider)
            return RemediationResult(
                action=RemediationAction.SWITCH_PROVIDER,
                success=True,
                message=f"Switched from {provider} to {new_provider}",
                context={'old_provider': provider, 'new_provider': new_provider}
            )
        except Exception as e:
            return RemediationResult(action=RemediationAction.SWITCH_PROVIDER, success=False, message=str(e), context={'provider': provider})
    
    async def _scale_workers(self, alert: dict) -> RemediationResult:
        try:
            current_workers = len(self.orchestrator.workers)
            target_workers = min(current_workers * 2, self.orchestrator.max_workers)
            await self.orchestrator.scale_workers(target_workers)
            return RemediationResult(
                action=RemediationAction.SCALE_WORKERS,
                success=True,
                message=f"Scaled workers from {current_workers} to {target_workers}",
                context={'old_count': current_workers, 'new_count': target_workers}
            )
        except Exception as e:
            return RemediationResult(action=RemediationAction.SCALE_WORKERS, success=False, message=str(e), context={})
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- Set up Prometheus for metrics collection
- Configure structlog for structured logging
- Implement basic metrics collector
- Add logging to agent worker and orchestrator
- Create basic Grafana dashboard
- **Deliverables**: Metrics endpoint, JSON logs, basic system dashboard

### Phase 2: Enhancement (Weeks 3-4)
- Implement OpenTelemetry tracing
- Set up Jaeger for trace collection
- Add distributed context propagation
- Implement provider-level metrics
- Create research-specific dashboards
- Set up Loki for log aggregation
- **Deliverables**: End-to-end tracing, provider dashboards, centralized logs

### Phase 3: Production (Weeks 5-6)
- Configure Alertmanager
- Define alert rules (critical, warning, info)
- Set up notification channels (Slack, PagerDuty)
- Implement auto-remediation actions
- Add log retention policies
- Create runbooks for common alerts
- **Deliverables**: Production alerting, auto-remediation, runbooks

### Phase 4: Scale (Weeks 7-8)
- Set up Thanos/Cortex for long-term metrics
- Implement custom research visualizations
- Add ML-based anomaly detection
- Create real-time research analytics
- Implement cross-cluster observability
- Add cost optimization dashboards
- **Deliverables**: Global observability, ML insights, cost tracking

---

## 8. Best Practices

### Metrics
- Use meaningful metric names: `swarm_component_metric_unit`
- Include relevant labels for filtering and grouping
- Use appropriate metric types (Counter, Gauge, Histogram)
- Define histogram buckets based on actual observed values
- Export metrics at `/metrics` endpoint for scraping
- Keep cardinality under control (avoid unbounded labels)

### Logging
- Always use structured logging (JSON) in production
- Include correlation IDs for request tracing
- Log at appropriate levels (don't spam INFO)
- Include context (agent_id, branch_id, etc.) in every log
- Never log sensitive information (API keys, PII)
- Use log aggregation for centralized search

### Tracing
- Create spans for significant operations
- Propagate context across service boundaries
- Add attributes to spans for debugging
- Set appropriate span kinds (SERVER, CLIENT, INTERNAL)
- Use sampling in production to control overhead
- Link related traces (research → sub-tasks)

### Alerting
- Alert on symptoms, not causes
- Use appropriate severity levels
- Include runbook links in alert annotations
- Set reasonable thresholds based on baselines
- Use inhibition to reduce alert noise
- Test alert routing regularly

### Dashboards
- Create role-specific dashboards
- Use consistent color schemes
- Include links to related dashboards
- Add annotations for significant events
- Optimize query performance
- Document dashboard purpose and usage

---

## 9. Summary

This observability design provides a comprehensive framework for monitoring, logging, tracing, and alerting in the SwarmResearch multi-agent AI research system. Key components include:

1. **Metrics Collection**: Prometheus-based metrics with custom instruments for agent performance, provider health, and research progress

2. **Logging Strategy**: Structured logging with structlog, supporting both development (pretty) and production (JSON) formats

3. **Distributed Tracing**: OpenTelemetry-based tracing for end-to-end request visibility across agent boundaries

4. **Dashboards**: Grafana dashboards for system overview, research progress, provider performance, and agent details

5. **Alerting**: Multi-tier alerting with Alertmanager, supporting critical, warning, and informational alerts with auto-remediation

The design follows a progressive enhancement approach, starting with basic observability in development and scaling to enterprise-grade monitoring in production deployments.

---

*Document Version: 1.0*
*Last Updated: 2024*
*Author: Observability Architect*
