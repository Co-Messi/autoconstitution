# Constitutional Critic Subsystem Design
## SwarmResearch - Architecture Specification

---

## Executive Summary

The Constitutional Critic subsystem is a defensive layer in the SwarmResearch architecture designed to prevent degradation of code quality, generalisability, and scientific validity. Critics are specialized agents whose sole purpose is to challenge proposed improvements through structured adversarial analysis.

**Core Principle**: *Every improvement must earn its place through rigorous critique.*

---

## 1. Critic Agent Interface

### 1.1 Agent Definition

```python
@dataclass
class CriticAgent:
    """
    Constitutional Critic - Adversarial validator of proposed changes.
    
    A critic has no incentive to approve changes. Their success is measured
    by the quality of objections raised, not by approval throughput.
    """
    agent_id: str                    # Unique identifier
    specialization: CriticSpecialization  # Primary critique domain
    constitutional_principles: List[Principle]  # Assigned principles
    critique_history: List[CritiqueRecord]      # Past critiques for calibration
    
    # Behavioral parameters
    skepticism_level: float          # 0.0-1.0, default 0.7
    thoroughness: float              # 0.0-1.0, default 0.8
    
class CriticSpecialization(Enum):
    """Domain expertise for specialized critique."""
    BENCHMARK_INTEGRITY = "benchmark"      # Detects overfitting
    GENERALISABILITY = "generalise"        # Cross-domain validation
    METRIC_SANITY = "metric"               # Gaming detection
    CODE_QUALITY = "quality"               # Maintainability analysis
    SCIENTIFIC_VALIDITY = "science"        # Methodological rigor
    SAFETY = "safety"                      # Risk assessment
```

### 1.2 Interface Contract

```python
class CriticInterface(ABC):
    """
    All critic implementations must satisfy this interface.
    """
    
    @abstractmethod
    async def critique(
        self,
        proposal: ImprovementProposal,
        context: ResearchContext,
        config: CritiqueConfig
    ) -> CritiqueReport:
        """
        Generate structured critique of a proposed improvement.
        
        Args:
            proposal: The improvement being challenged
            context: Research state, benchmarks, history
            config: Critique parameters and constraints
            
        Returns:
            Complete critique report with verdict and reasoning
        """
        pass
    
    @abstractmethod
    def get_confidence(
        self,
        critique: CritiqueReport
    ) -> ConfidenceScore:
        """
        Calculate confidence in critique conclusions.
        
        Confidence reflects certainty, not approval likelihood.
        """
        pass
    
    @abstractmethod
    def calibrate(
        self,
        outcomes: List[Tuple[CritiqueReport, ActualOutcome]]
    ) -> CalibrationMetrics:
        """
        Update internal parameters based on historical accuracy.
        """
        pass
```

### 1.3 Input Specification

```python
@dataclass
class ImprovementProposal:
    """The target of critique."""
    proposal_id: str
    description: str
    code_changes: CodeDiff
    expected_improvement: Dict[str, float]  # metric -> expected delta
    supporting_evidence: List[ExperimentResult]
    author_agent: str
    timestamp: datetime
    
@dataclass
class ResearchContext:
    """Context for informed critique."""
    current_benchmarks: Dict[str, BenchmarkResult]
    historical_changes: List[CommittedChange]
    codebase_state: CodebaseSnapshot
    related_work: List[PaperReference]
    failure_modes: List[KnownFailure]
    
@dataclass
class CritiqueConfig:
    """Parameters controlling critique behavior."""
    max_critique_time: timedelta = timedelta(minutes=5)
    min_objection_depth: int = 2  # Minimum levels of "why"
    require_evidence: bool = True
    enable_speculative: bool = True  # Allow "what if" objections
```

---

## 2. Structured Critique Format

### 2.1 Critique Report Schema

```python
@dataclass
class CritiqueReport:
    """
    Structured output from critic analysis.
    
    The report must be machine-parseable and human-auditable.
    """
    report_id: str
    critic_id: str
    proposal_id: str
    timestamp: datetime
    
    # Core verdict
    verdict: Verdict
    confidence: ConfidenceScore
    
    # Structured objections
    objections: List[Objection]
    
    # Positive findings (critics can note genuine improvements)
    acknowledgments: List[Acknowledgment]
    
    # Recommendations
    recommendations: List[Recommendation]
    
    # Metadata
    critique_duration: timedelta
    principles_invoked: List[str]
    
class Verdict(Enum):
    """Final judgment on proposal."""
    REJECT = "reject"           # Fatal flaws identified
    CONDITIONAL = "conditional" # Acceptable with modifications
    ACCEPT = "accept"           # No significant objections
    ABSTAIN = "abstain"         # Insufficient expertise/context
```

### 2.2 Objection Structure

```python
@dataclass
class Objection:
    """
    Individual critique point with full reasoning chain.
    """
    objection_id: str
    category: ObjectionCategory
    severity: Severity
    
    # Structured argument
    claim: str                    # What is wrong
    reasoning: str                # Why it's wrong
    evidence: List[Evidence]      # Supporting data
    counterfactual: str           # What could go wrong
    
    # Traceability
    principle_violated: Optional[str]
    related_objections: List[str] # Links to other objections
    
    # Resolution path
    mitigatable: bool
    mitigation_suggestion: Optional[str]
    
class ObjectionCategory(Enum):
    """Taxonomy of critique types."""
    # Benchmark concerns
    OVERFITTING = "overfitting"           # Fits benchmark too well
    BENCHMARK_SPECIFIC = "benchmark_specific"  # Doesn't generalize
    
    # Generalisability concerns  
    DOMAIN_OVERFIT = "domain_overfit"     # Narrow applicability
    DISTRIBUTION_SHIFT = "dist_shift"     # Brittle to changes
    
    # Metric concerns
    METRIC_GAMING = "metric_gaming"       # Optimizes wrong thing
    PROXY_FAILURE = "proxy_failure"       # Metric doesn't measure goal
    
    # Code concerns
    COMPLEXITY = "complexity"             # Unnecessary complexity
    MAINTAINABILITY = "maintainability"   # Hard to maintain
    READABILITY = "readability"           # Hard to understand
    TESTABILITY = "testability"           # Hard to test
    
    # Scientific concerns
    METHODOLOGY = "methodology"           # Flawed approach
    REPRODUCIBILITY = "reproducibility"   # Can't be replicated
    STATISTICAL = "statistical"           # Insufficient evidence
    
    # Safety concerns
    REGRESSION = "regression"             # Breaks existing functionality
    SIDE_EFFECT = "side_effect"           # Unintended consequences
    
class Severity(Enum):
    """Impact level of objection."""
    CRITICAL = 4      # Must be addressed
    HIGH = 3          # Should be addressed
    MEDIUM = 2        # Worth considering
    LOW = 1           # Minor concern
    INFO = 0          # For awareness only
```

### 2.3 Evidence Structure

```python
@dataclass
class Evidence:
    """Supporting data for objections."""
    evidence_type: EvidenceType
    description: str
    data: Any  # Structured evidence data
    source: str  # Where evidence came from
    confidence: float  # 0.0-1.0
    
class EvidenceType(Enum):
    """Types of supporting evidence."""
    STATISTICAL_TEST = "statistical"      # p-values, confidence intervals
    CODE_ANALYSIS = "code"                # Static analysis results
    EXPERIMENT_RESULT = "experiment"      # Empirical findings
    LITERATURE_REFERENCE = "literature"   # Published research
    SIMULATION = "simulation"             # Simulated scenarios
    ANALOGY = "analogy"                   # Similar past failures
    THEORETICAL = "theoretical"           # First-principles reasoning
```

### 2.4 Example Critique Report (JSON)

```json
{
  "report_id": "critique-2024-001-a7f3",
  "critic_id": "critic-benchmark-integrity-01",
  "proposal_id": "prop-2024-0892",
  "timestamp": "2024-01-15T14:32:11Z",
  "verdict": "CONDITIONAL",
  "confidence": {
    "score": 0.78,
    "calibration": "well_calibrated",
    "uncertainty_sources": ["limited_test_data", "novel_technique"]
  },
  "objections": [
    {
      "objection_id": "obj-001",
      "category": "OVERFITTING",
      "severity": "HIGH",
      "claim": "The proposed optimization shows 95% improvement on the target benchmark but only 12% improvement on held-out test sets",
      "reasoning": "The pattern of high target performance with low generalization is characteristic of benchmark overfitting. The optimization appears to exploit specific characteristics of the evaluation set rather than improving the underlying capability.",
      "evidence": [
        {
          "evidence_type": "STATISTICAL_TEST",
          "description": "Performance gap significance test",
          "data": {"p_value": 0.003, "effect_size": 0.83},
          "source": "internal_statistical_analysis",
          "confidence": 0.92
        }
      ],
      "counterfactual": "If we change the benchmark distribution slightly, the improvement may vanish or become negative",
      "principle_violated": "PRINCIPLE_GENERALISABILITY",
      "mitigatable": true,
      "mitigation_suggestion": "Validate on at least 3 additional held-out test sets from different distributions"
    }
  ],
  "acknowledgments": [
    {
      "description": "The proposed caching mechanism is well-designed and follows established patterns",
      "category": "CODE_QUALITY"
    }
  ],
  "recommendations": [
    {
      "priority": "HIGH",
      "description": "Conduct cross-validation across multiple benchmark distributions",
      "rationale": "Will validate whether improvement is genuine or benchmark-specific"
    }
  ],
  "principles_invoked": ["PRINCIPLE_GENERALISABILITY", "PRINCIPLE_STATISTICAL_RIGOR"]
}
```

---

## 3. Confidence Scoring

### 3.1 Confidence Score Structure

```python
@dataclass
class ConfidenceScore:
    """
    Multi-dimensional confidence assessment.
    
    Confidence reflects the critic's certainty in their analysis,
    NOT the probability of proposal success.
    """
    # Overall confidence
    score: float  # 0.0-1.0
    
    # Calibration status
    calibration: CalibrationStatus
    
    # Component confidences
    components: Dict[str, float]
    
    # Uncertainty sources
    uncertainty_sources: List[str]
    
    # Confidence history for this critic
    historical_accuracy: Optional[float]
    
class CalibrationStatus(Enum):
    """How well-calibrated the confidence is."""
    WELL_CALIBRATED = "well_calibrated"    # Historical accuracy matches confidence
    OVERCONFIDENT = "overconfident"        # Usually wrong when confident
    UNDERCONFIDENT = "underconfident"      # Usually right when uncertain
    UNCALIBRATED = "uncalibrated"          # Insufficient history
    
# Component confidence breakdown
CONFIDENCE_COMPONENTS = {
    "evidence_quality": "Quality of supporting evidence",
    "reasoning_soundness": "Logical validity of arguments", 
    "domain_expertise": "Critic's expertise in relevant domain",
    "context_completeness": "Completeness of available context",
    "statistical_power": "Statistical strength of claims",
    "reproducibility": "Ability to reproduce findings"
}
```

### 3.2 Confidence Calculation

```python
class ConfidenceCalculator:
    """
    Computes confidence scores using multiple signals.
    """
    
    def calculate(
        self,
        critique: CritiqueReport,
        critic_history: List[CritiqueRecord]
    ) -> ConfidenceScore:
        """
        Compute multi-dimensional confidence score.
        """
        components = {}
        
        # Evidence quality (0.0-1.0)
        components["evidence_quality"] = self._score_evidence(
            critique.objections
        )
        
        # Reasoning soundness
        components["reasoning_soundness"] = self._check_reasoning(
            critique.objections
        )
        
        # Domain expertise match
        components["domain_expertise"] = self._match_expertise(
            critique.critic_id,
            critique.proposal_id
        )
        
        # Context completeness
        components["context_completeness"] = self._assess_context(
            critique
        )
        
        # Statistical power
        components["statistical_power"] = self._statistical_power(
            critique.objections
        )
        
        # Overall score (weighted combination)
        weights = {
            "evidence_quality": 0.25,
            "reasoning_soundness": 0.25,
            "domain_expertise": 0.20,
            "context_completeness": 0.15,
            "statistical_power": 0.15
        }
        
        overall = sum(
            components[k] * weights[k] for k in weights
        )
        
        # Uncertainty sources
        uncertainties = self._identify_uncertainties(
            critique, components
        )
        
        # Calibration status
        calibration = self._assess_calibration(critic_history)
        
        return ConfidenceScore(
            score=overall,
            calibration=calibration,
            components=components,
            uncertainty_sources=uncertainties,
            historical_accuracy=self._historical_accuracy(critic_history)
        )
```

### 3.3 Calibration Tracking

```python
@dataclass
class CalibrationRecord:
    """Tracks how accurate critic predictions were."""
    critique_id: str
    predicted_verdict: Verdict
    predicted_confidence: float
    actual_outcome: ActualOutcome
    outcome_timestamp: datetime
    
class ActualOutcome(Enum):
    """What actually happened to the proposal."""
    COMMITTED_AND_SUCCESSFUL = "committed_success"
    COMMITTED_AND_FAILED = "committed_failed"
    COMMITTED_AND_MIXED = "committed_mixed"
    REJECTED = "rejected"
    MODIFIED_THEN_COMMITTED = "modified"
    ABANDONED = "abandoned"

def compute_calibration_metrics(
    records: List[CalibrationRecord]
) -> Dict[str, float]:
    """
    Compute calibration statistics.
    
    Returns metrics indicating whether confidence scores
    accurately reflect prediction accuracy.
    """
    # Bin by confidence level
    bins = defaultdict(list)
    for r in records:
        bin_key = round(r.predicted_confidence * 10) / 10
        bins[bin_key].append(r)
    
    metrics = {}
    for conf_level, bin_records in bins.items():
        # Accuracy at this confidence level
        correct = sum(
            1 for r in bin_records
            if verdict_matches_outcome(
                r.predicted_verdict, r.actual_outcome
            )
        )
        accuracy = correct / len(bin_records)
        
        # Calibration error
        metrics[f"calibration_error_{conf_level}"] = abs(
            accuracy - conf_level
        )
    
    # Expected calibration error
    metrics["expected_calibration_error"] = np.mean([
        metrics[k] for k in metrics if k.startswith("calibration_error_")
    ])
    
    return metrics
```

---

## 4. Multi-Critic Consensus Mechanism

### 4.1 Consensus Architecture

```python
@dataclass
class ConsensusConfig:
    """Configuration for multi-critic consensus."""
    
    # Critic selection
    min_critics: int = 3
    max_critics: int = 7
    required_specializations: List[CriticSpecialization]
    
    # Consensus rules
    consensus_threshold: float = 0.67  # % agreement required
    unanimity_required_for: List[Verdict] = [Verdict.REJECT]
    
    # Confidence weighting
    weight_by_confidence: bool = True
    weight_by_calibration: bool = True
    
    # Dispute resolution
    enable_deliberation: bool = True
    max_deliberation_rounds: int = 2
    
    # Tie-breaking
    tie_breaker: TieBreaker = TieBreaker.CONSERVATIVE
    
class TieBreaker(Enum):
    """How to resolve ties."""
    CONSERVATIVE = "conservative"    # Prefer rejection/revision
    OPTIMISTIC = "optimistic"        # Prefer acceptance
    RANDOM = "random"                # Random selection
    EXPERT_PANEL = "expert_panel"    # Escalate to senior critics
```

### 4.2 Consensus Process

```python
class ConsensusEngine:
    """
    Orchestrates multi-critic consensus.
    """
    
    async def reach_consensus(
        self,
        proposal: ImprovementProposal,
        config: ConsensusConfig
    ) -> ConsensusResult:
        """
        Execute multi-critic consensus process.
        """
        # Phase 1: Select critics
        critics = self._select_critics(proposal, config)
        
        # Phase 2: Initial critiques (parallel)
        initial_critiques = await asyncio.gather(*[
            critic.critique(proposal, self.context, CritiqueConfig())
            for critic in critics
        ])
        
        # Phase 3: Check for early consensus
        preliminary = self._aggregate_critiques(
            initial_critiques, config
        )
        
        if preliminary.consensus_reached:
            return self._finalize_consensus(preliminary)
        
        # Phase 4: Deliberation (if enabled)
        if config.enable_deliberation:
            deliberated = await self._deliberation_phase(
                initial_critiques, proposal, config
            )
            
            final = self._aggregate_critiques(deliberated, config)
            
            if final.consensus_reached:
                return self._finalize_consensus(final)
        
        # Phase 5: Dispute resolution
        resolved = self._resolve_dispute(final, config)
        return self._finalize_consensus(resolved)
    
    def _select_critics(
        self,
        proposal: ImprovementProposal,
        config: ConsensusConfig
    ) -> List[CriticAgent]:
        """
        Select diverse critic panel.
        """
        selected = []
        
        # Ensure required specializations
        for spec in config.required_specializations:
            critic = self._best_available_critic(spec, proposal)
            selected.append(critic)
        
        # Add additional critics for diversity
        remaining_slots = config.max_critics - len(selected)
        additional = self._diverse_critics(
            proposal, remaining_slots, exclude=selected
        )
        selected.extend(additional)
        
        return selected[:config.max_critics]
    
    def _aggregate_critiques(
        self,
        critiques: List[CritiqueReport],
        config: ConsensusConfig
    ) -> PreliminaryConsensus:
        """
        Aggregate multiple critiques into preliminary consensus.
        """
        # Weight by confidence and calibration
        weights = self._compute_weights(critiques, config)
        
        # Weighted vote on verdict
        verdict_votes = defaultdict(float)
        for critique, weight in zip(critiques, weights):
            verdict_votes[critique.verdict] += weight
        
        total_weight = sum(weights)
        verdict_distribution = {
            v: w / total_weight for v, w in verdict_votes.items()
        }
        
        # Check consensus threshold
        max_verdict = max(verdict_distribution, key=verdict_votes.get)
        max_support = verdict_distribution[max_verdict]
        
        consensus_reached = max_support >= config.consensus_threshold
        
        # Check unanimity requirements
        if max_verdict in config.unanimity_required_for:
            consensus_reached = max_support == 1.0
        
        return PreliminaryConsensus(
            verdict_distribution=verdict_distribution,
            leading_verdict=max_verdict if consensus_reached else None,
            consensus_reached=consensus_reached,
            confidence=self._aggregate_confidence(critiques, weights),
            objections_merged=self._merge_objections(critiques),
            critique_reports=critiques
        )
```

### 4.3 Deliberation Phase

```python
async def _deliberation_phase(
    self,
    initial_critiques: List[CritiqueReport],
    proposal: ImprovementProposal,
    config: ConsensusConfig
) -> List[CritiqueReport]:
    """
    Allow critics to respond to each other's critiques.
    """
    deliberated = initial_critiques.copy()
    
    for round_num in range(config.max_deliberation_rounds):
        # Identify points of disagreement
        disagreements = self._find_disagreements(deliberated)
        
        if not disagreements:
            break  # Consensus achieved through deliberation
        
        # Critics respond to disagreements
        responses = await asyncio.gather(*[
            self._generate_response(
                critic, disagreements, deliberated, proposal
            )
            for critic in self.critics
        ])
        
        # Update critiques with responses
        deliberated = self._update_with_responses(
            deliberated, responses
        )
    
    return deliberated

def _find_disagreements(
    critiques: List[CritiqueReport]
) -> List[Disagreement]:
    """
    Identify areas where critics disagree.
    """
    disagreements = []
    
    # Verdict disagreements
    verdicts = [c.verdict for c in critiques]
    if len(set(verdicts)) > 1:
        disagreements.append(Disagreement(
            type="verdict",
            description=f"Critics split: {Counter(verdicts)}",
            severity="high"
        ))
    
    # Objection disagreements
    all_objections = []
    for c in critiques:
        all_objections.extend(c.objections)
    
    # Group related objections
    grouped = self._group_related_objections(all_objections)
    
    for group in grouped:
        severities = [o.severity for o in group]
        if len(set(severities)) > 1:
            disagreements.append(Disagreement(
                type="severity",
                description=f"Severity disagreement on: {group[0].claim[:50]}...",
                severity="medium",
                related_objections=[o.objection_id for o in group]
            ))
    
    return disagreements
```

### 4.4 Consensus Result

```python
@dataclass
class ConsensusResult:
    """Final output of consensus process."""
    
    # Decision
    final_verdict: Verdict
    confidence: ConfidenceScore
    
    # Process metadata
    num_critics: int
    critic_ids: List[str]
    deliberation_rounds: int
    
    # Detailed breakdown
    verdict_distribution: Dict[Verdict, float]
    all_objections: List[Objection]
    all_acknowledgments: List[Acknowledgment]
    
    # Dissent
    dissenting_opinions: List[Dissent]
    
    # Conditions (for CONDITIONAL verdict)
    conditions: List[Condition]
    
    # Audit trail
    process_log: List[ProcessEvent]
    
@dataclass
class Dissent:
    """Record of disagreement from consensus."""
    critic_id: str
    their_verdict: Verdict
    consensus_verdict: Verdict
    reasoning: str
    confidence_in_dissent: float
    
@dataclass
class Condition:
    """Required condition for CONDITIONAL acceptance."""
    condition_id: str
    description: str
    verification_method: str
    deadline: Optional[datetime]
    blocking: bool  # Must be met before commit
```

---

## 5. Constitutional Principles

### 5.1 Principle Hierarchy

```python
@dataclass
class ConstitutionalPrinciple:
    """
    Fundamental principle that guides critique.
    
    Principles are organized hierarchically and can be invoked
    to justify objections.
    """
    principle_id: str
    name: str
    statement: str
    rationale: str
    
    # Hierarchy
    parent: Optional[str]  # Parent principle
    children: List[str]    # Sub-principles
    
    # Application
    applies_to: List[ProposalType]
    
    # Priority
    priority: Priority  # For conflict resolution
    
    # Examples
    positive_examples: List[str]  # What adherence looks like
    negative_examples: List[str]  # What violation looks like
    
class Priority(Enum):
    """Principle priority for conflict resolution."""
    CRITICAL = 5    # Never violated
    HIGH = 4        # Violated only with extraordinary justification
    MEDIUM = 3      # Important but can be balanced
    LOW = 2         # Preferential guidance
    ASPIRATIONAL = 1  # Nice to have
```

### 5.2 Core Principles

```python
# =============================================================================
# TIER 1: CRITICAL PRINCIPLES (Never Violated)
# =============================================================================

PRINCIPLE_SCIENTIFIC_INTEGRITY = ConstitutionalPrinciple(
    principle_id="PRINCIPLE_SCIENTIFIC_INTEGRITY",
    name="Scientific Integrity",
    statement="""
    All claims must be supported by evidence. No fabrication, 
    cherry-picking, or misleading presentation of results.
    """,
    rationale="""
    Scientific progress depends on trust. Violations destroy 
    the foundation of cumulative knowledge.
    """,
    parent=None,
    children=[
        "PRINCIPLE_REPRODUCIBILITY",
        "PRINCIPLE_STATISTICAL_RIGOR",
        "PRINCIPLE_HONEST_REPORTING"
    ],
    applies_to=[ProposalType.ALL],
    priority=Priority.CRITICAL,
    positive_examples=[
        "Reporting all experiments, including failed ones",
        "Including confidence intervals with point estimates",
        "Clearly stating assumptions and limitations"
    ],
    negative_examples=[
        "Only reporting the best of 20 random seeds",
        "Omitting negative results that contradict claims",
        "Using p-hacking to achieve significance"
    ]
)

PRINCIPLE_SAFETY = ConstitutionalPrinciple(
    principle_id="PRINCIPLE_SAFETY",
    name="Safety First",
    statement="""
    Changes must not introduce unacceptable risks to system 
    integrity, data security, or user safety.
    """,
    rationale="""
    A broken system cannot generate knowledge. Safety is 
    prerequisite to all other activities.
    """,
    parent=None,
    children=[
        "PRINCIPLE_NO_REGRESSIONS",
        "PRINCIPLE_SECURE_BY_DEFAULT"
    ],
    applies_to=[ProposalType.ALL],
    priority=Priority.CRITICAL,
    positive_examples=[
        "Adding comprehensive tests for new functionality",
        "Performing security review of external dependencies",
        "Validating input sanitization"
    ],
    negative_examples=[
        "Disabling safety checks to improve performance",
        "Introducing eval() on user input",
        "Removing test coverage for critical paths"
    ]
)

# =============================================================================
# TIER 2: HIGH PRIORITY PRINCIPLES
# =============================================================================

PRINCIPLE_GENERALISABILITY = ConstitutionalPrinciple(
    principle_id="PRINCIPLE_GENERALISABILITY",
    name="True Generalisation",
    statement="""
    Improvements must demonstrate generalisation beyond the 
    specific benchmark or test case used for development.
    """,
    rationale="""
    Research aims for general knowledge, not benchmark 
    optimization. Overfitting to benchmarks is a form of 
    intellectual fraud.
    """,
    parent=None,
    children=[
        "PRINCIPLE_CROSS_DOMAIN",
        "PRINCIPLE_DISTRIBUTION_ROBUSTNESS",
        "PRINCIPLE_NO_BENCHMARK_OVERFIT"
    ],
    applies_to=[
        ProposalType.ALGORITHM,
        ProposalType.MODEL,
        ProposalType.OPTIMIZATION
    ],
    priority=Priority.HIGH,
    positive_examples=[
        "Testing on multiple independent benchmarks",
        "Using held-out test sets not seen during development",
        "Demonstrating improvement across different data distributions"
    ],
    negative_examples=[
        "Tuning hyperparameters specifically for the test set",
        "Using test set information during model selection",
        "Only evaluating on the development benchmark"
    ]
)

PRINCIPLE_METRIC_VALIDITY = ConstitutionalPrinciple(
    principle_id="PRINCIPLE_METRIC_VALIDITY",
    name="Valid Metrics Only",
    statement="""
    Optimized metrics must actually measure the intended 
    construct. Gaming metrics is not progress.
    """,
    rationale="""
    Metrics are proxies for goals. When we optimize the 
    proxy instead of the goal, we achieve the wrong thing.
    """,
    parent=None,
    children=[
        "PRINCIPLE_NO_METRIC_GAMING",
        "PRINCIPLE_PROXY_AWARENESS"
    ],
    applies_to=[
        ProposalType.METRIC,
        ProposalType.OPTIMIZATION,
        ProposalType.EVALUATION
    ],
    priority=Priority.HIGH,
    positive_examples=[
        "Validating that metric improvements correlate with human judgment",
        "Using multiple complementary metrics",
        "Periodically auditing metric alignment with goals"
    ],
    negative_examples=[
        "Exploiting loopholes in metric definitions",
        "Optimizing BLEU when human quality is the real goal",
        "Creating artificial metric improvements without real improvement"
    ]
)

PRINCIPLE_CODE_QUALITY = ConstitutionalPrinciple(
    principle_id="PRINCIPLE_CODE_QUALITY",
    name="Maintainable Code",
    statement="""
    Code must be readable, testable, and maintainable. 
    Cleverness that sacrifices clarity is not improvement.
    """,
    rationale="""
    Research code is read more than written. Future researchers
    (including future you) depend on clear, maintainable code.
    """,
    parent=None,
    children=[
        "PRINCIPLE_READABILITY",
        "PRINCIPLE_TESTABILITY",
        "PRINCIPLE_SIMPLICITY",
        "PRINCIPLE_DOCUMENTATION"
    ],
    applies_to=[
        ProposalType.CODE,
        ProposalType.REFACTOR,
        ProposalType.ALGORITHM
    ],
    priority=Priority.HIGH,
    positive_examples=[
        "Writing clear docstrings for all public functions",
        "Following established style guides",
        "Adding unit tests for new functionality",
        "Refactoring complex code into understandable pieces"
    ],
    negative_examples=[
        "One-liners that require 10 minutes to understand",
        "Removing comments to reduce line count",
        "Using obscure language features unnecessarily",
        "Copy-pasting code instead of creating reusable functions"
    ]
)

# =============================================================================
# TIER 3: MEDIUM PRIORITY PRINCIPLES
# =============================================================================

PRINCIPLE_EFFICIENCY = ConstitutionalPrinciple(
    principle_id="PRINCIPLE_EFFICIENCY",
    name="Reasonable Efficiency",
    statement="""
    Solutions should be appropriately efficient. Unnecessary 
    waste of resources is not acceptable.
    """,
    rationale="""
    Computational resources are finite. Inefficient solutions
    limit scalability and reproducibility.
    """,
    parent=None,
    children=[
        "PRINCIPLE_TIME_COMPLEXITY",
        "PRINCIPLE_SPACE_COMPLEXITY",
        "PRINCIPLE_ENERGY_EFFICIENCY"
    ],
    applies_to=[
        ProposalType.ALGORITHM,
        ProposalType.OPTIMIZATION,
        ProposalType.INFRASTRUCTURE
    ],
    priority=Priority.MEDIUM,
    positive_examples=[
        "Using appropriate data structures",
        "Profiling before optimizing",
        "Considering computational cost of experiments"
    ],
    negative_examples=[
        "O(n²) solutions when O(n log n) is straightforward",
        "Loading entire datasets into memory unnecessarily",
        "Running experiments with 10x the needed samples"
    ]
)

PRINCIPLE_REPRODUCIBILITY = ConstitutionalPrinciple(
    principle_id="PRINCIPLE_REPRODUCIBILITY",
    name="Reproducible Research",
    statement="""
    Results must be reproducible by others. This requires 
    clear documentation, fixed dependencies, and shared code.
    """,
    rationale="""
    Science depends on verification. Irreproducible results
    cannot be built upon.
    """,
    parent="PRINCIPLE_SCIENTIFIC_INTEGRITY",
    children=[],
    applies_to=[ProposalType.ALL],
    priority=Priority.MEDIUM,
    positive_examples=[
        "Pinning all dependency versions",
        "Providing random seeds",
        "Documenting hardware and software environment",
        "Sharing code and data"
    ],
    negative_examples=[
        "Using 'latest' versions of dependencies",
        "Not recording experimental conditions",
        "Keeping code private after publication"
    ]
)

# =============================================================================
# TIER 4: LOW PRIORITY / ASPIRATIONAL PRINCIPLES
# =============================================================================

PRINCIPLE_ELEGANCE = ConstitutionalPrinciple(
    principle_id="PRINCIPLE_ELEGANCE",
    name="Elegant Solutions",
    statement="""
    Prefer elegant solutions when equally effective. Beauty 
    in code often indicates correctness.
    """,
    rationale="""
    Elegant solutions are often simpler, more general, and 
    easier to understand. However, correctness trumps elegance.
    """,
    parent=None,
    children=[],
    applies_to=[ProposalType.ALL],
    priority=Priority.LOW,
    positive_examples=[
        "Finding the minimal sufficient solution",
        "Using mathematical insights to simplify",
        "Following established design patterns"
    ],
    negative_examples=[
        "Sacrificing clarity for cleverness",
        "Forcing elegance where simplicity suffices"
    ]
)
```

### 5.3 Principle Application

```python
class PrincipleEnforcer:
    """
    Applies constitutional principles to critique.
    """
    
    def check_compliance(
        self,
        proposal: ImprovementProposal,
        principle: ConstitutionalPrinciple
    ) -> PrincipleAssessment:
        """
        Assess proposal compliance with a principle.
        """
        # Check if principle applies
        if not self._applies_to(proposal, principle):
            return PrincipleAssessment(
                principle_id=principle.principle_id,
                applicable=False,
                compliance=None,
                reasoning="Principle does not apply to this proposal type"
            )
        
        # Evaluate compliance
        violations = self._find_violations(proposal, principle)
        adherence = self._find_adherence(proposal, principle)
        
        # Determine compliance level
        if violations:
            compliance = ComplianceLevel.VIOLATED
        elif adherence:
            compliance = ComplianceLevel.ADHERED
        else:
            compliance = ComplianceLevel.NEUTRAL
        
        return PrincipleAssessment(
            principle_id=principle.principle_id,
            applicable=True,
            compliance=compliance,
            violations=violations,
            adherence_points=adherence,
            reasoning=self._generate_reasoning(
                principle, violations, adherence
            )
        )
    
    def resolve_conflict(
        self,
        p1: ConstitutionalPrinciple,
        p2: ConstitutionalPrinciple
    ) -> ConstitutionalPrinciple:
        """
        Resolve conflicts between principles using priority.
        """
        if p1.priority.value > p2.priority.value:
            return p1
        elif p2.priority.value > p1.priority.value:
            return p2
        else:
            # Same priority - check hierarchy
            if p2.principle_id in p1.children:
                return p1  # Parent overrides child
            elif p1.principle_id in p2.children:
                return p2
            else:
                # No clear resolution - escalate
                raise PrincipleConflictError(p1, p2)
```

---

## 6. Integration with Ratchet System

### 6.1 Critique-to-Ratchet Flow

```python
@dataclass
class RatchetGate:
    """
    Gateway between critique and ratchet systems.
    
    Only proposals passing critique can reach the ratchet.
    """
    
    async def evaluate_for_ratchet(
        self,
        proposal: ImprovementProposal
    ) -> RatchetDecision:
        """
        Determine if proposal should be committed to ratchet.
        """
        # Run consensus
        consensus = await self.consensus_engine.reach_consensus(
            proposal, self.consensus_config
        )
        
        # Map consensus to ratchet decision
        if consensus.final_verdict == Verdict.REJECT:
            return RatchetDecision(
                action=RatchetAction.REJECT,
                proposal=proposal,
                consensus=consensus,
                reason="Consensus rejection from critic panel"
            )
        
        elif consensus.final_verdict == Verdict.CONDITIONAL:
            return RatchetDecision(
                action=RatchetAction.CONDITIONAL,
                proposal=proposal,
                consensus=consensus,
                conditions=consensus.conditions,
                reason="Conditional acceptance - conditions must be met"
            )
        
        elif consensus.final_verdict == Verdict.ACCEPT:
            return RatchetDecision(
                action=RatchetAction.ACCEPT,
                proposal=proposal,
                consensus=consensus,
                reason="Consensus acceptance from critic panel"
            )
        
        else:  # ABSTAIN
            return RatchetDecision(
                action=RatchetAction.ESCALATE,
                proposal=proposal,
                consensus=consensus,
                reason="Critics abstained - requires human review"
            )
```

### 6.2 Audit Trail

```python
@dataclass
class CritiqueAuditRecord:
    """
    Complete audit trail for critique process.
    """
    record_id: str
    proposal_id: str
    timestamp_start: datetime
    timestamp_end: datetime
    
    # Critic details
    critic_panel: List[str]
    specializations_represented: List[str]
    
    # Process details
    deliberation_rounds: int
    consensus_reached: bool
    
    # Decisions
    individual_verdicts: Dict[str, Verdict]
    final_verdict: Verdict
    
    # Principles
    principles_invoked: List[str]
    principles_violated: List[str]
    
    # Objections
    objections_raised: int
    objections_addressed: int
    
    # Outcome
    ratchet_action: RatchetAction
    conditions: List[str]
    
    # Full records
    full_critiques: List[CritiqueReport]
    consensus_result: ConsensusResult
```

---

## 7. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Implement CriticAgent base class
- [ ] Implement CritiqueReport schema
- [ ] Implement ConfidenceScore calculator
- [ ] Create principle registry

### Phase 2: Specialization
- [ ] Implement BenchmarkIntegrityCritic
- [ ] Implement GeneralisabilityCritic
- [ ] Implement MetricSanityCritic
- [ ] Implement CodeQualityCritic
- [ ] Implement ScientificValidityCritic

### Phase 3: Consensus
- [ ] Implement ConsensusEngine
- [ ] Implement deliberation logic
- [ ] Implement dispute resolution
- [ ] Create tie-breaking mechanisms

### Phase 4: Integration
- [ ] Connect to ratchet system
- [ ] Implement audit logging
- [ ] Create monitoring dashboards
- [ ] Add calibration tracking

### Phase 5: Validation
- [ ] Test with historical proposals
- [ ] Measure calibration accuracy
- [ ] Validate consensus quality
- [ ] Performance benchmarking

---

## 8. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| False Positive Rate | < 10% | Good proposals rejected |
| False Negative Rate | < 5% | Bad proposals accepted |
| Calibration Error | < 0.15 | ECE on confidence scores |
| Consensus Time | < 10 min | End-to-end critique time |
| Coverage | 100% | All proposals critiqued |
| Principle Invocation | > 80% | Critiques citing principles |

---

## Appendix A: Configuration Templates

### Minimal Config
```yaml
critic_system:
  min_critics: 3
  required_specializations:
    - benchmark
    - quality
    - metric
  consensus_threshold: 0.67
  enable_deliberation: false
```

### Standard Config
```yaml
critic_system:
  min_critics: 5
  max_critics: 7
  required_specializations:
    - benchmark
    - generalise
    - metric
    - quality
    - science
  consensus_threshold: 0.75
  unanimity_required_for:
    - reject
  enable_deliberation: true
  max_deliberation_rounds: 2
  weight_by_confidence: true
  weight_by_calibration: true
```

### Maximum Scrutiny Config
```yaml
critic_system:
  min_critics: 7
  max_critics: 11
  required_specializations:
    - benchmark
    - generalise
    - metric
    - quality
    - science
    - safety
  consensus_threshold: 0.80
  unanimity_required_for:
    - reject
    - accept
  enable_deliberation: true
  max_deliberation_rounds: 3
  weight_by_confidence: true
  weight_by_calibration: true
  tie_breaker: expert_panel
```

---

*Document Version: 1.0*
*Last Updated: 2024*
*Status: Design Specification*
