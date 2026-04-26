# autoconstitution Orchestrator Design
## PARL-Based Multi-Agent Research Orchestration System

---

## 1. Overview

The autoconstitution Orchestrator is a master agent that coordinates parallel research exploration using the PARL (Parallel Autoregressive Learning) paradigm. It decomposes complex research problems, spawns specialized sub-agents, monitors their performance, and dynamically reallocates computational resources to maximize research progress.

### Core Principles
- **Parallel Exploration**: Multiple research branches explore simultaneously
- **Gradient-Guided Allocation**: Resources flow to high-progress branches
- **Global Ratchet**: Best findings are preserved and built upon
- **Dynamic Rebalancing**: Agents migrate from stagnant to promising areas

---

## 2. Orchestrator Class Interface

### 2.1 Main Orchestrator Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, AsyncIterator
from enum import Enum, auto
from datetime import datetime
import asyncio
from uuid import UUID, uuid4

class AgentStatus(Enum):
    """Lifecycle states for research agents."""
    PENDING = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    REALLOCATING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TERMINATED = auto()

class BranchStatus(Enum):
    """Status of a research branch."""
    EXPLORING = auto()      # Active exploration
    CONVERGING = auto()     # Nearing local optimum
    STAGNANT = auto()       # No progress detected
    BREAKTHROUGH = auto()   # Significant finding
    MERGED = auto()         # Consolidated into another branch
    ABANDONED = auto()      # Terminated due to low potential

@dataclass
class ResearchContext:
    """Global context shared across all research branches."""
    problem_statement: str
    domain: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    max_depth: int = 5
    exploration_budget: int = 100  # Total agent-iterations allowed
    
@dataclass
class AgentConfig:
    """Configuration for spawning a research agent."""
    agent_type: str  # 'explorer', 'critic', 'synthesizer', 'verifier'
    specialization: str  # Domain expertise
    depth: int  # Exploration depth in tree
    parent_id: Optional[UUID] = None
    hypotheses: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    
@dataclass
class PerformanceMetrics:
    """Metrics tracked for each agent/branch."""
    agent_id: UUID
    branch_id: UUID
    start_time: datetime
    iterations: int = 0
    hypotheses_generated: int = 0
    hypotheses_validated: int = 0
    knowledge_contributions: List[Dict] = field(default_factory=list)
    gradient_score: float = 0.0  # Rate of progress
    last_improvement_time: Optional[datetime] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)

@dataclass
class GlobalRatchetState:
    """The global best state that only improves (ratchet mechanism)."""
    best_hypothesis: Optional[str] = None
    best_score: float = 0.0
    best_evidence: List[Dict] = field(default_factory=list)
    knowledge_graph: Dict[str, Any] = field(default_factory=dict)
    consolidated_findings: List[Dict] = field(default_factory=list)
    version: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def update_if_better(self, hypothesis: str, score: float, evidence: List[Dict]) -> bool:
        """Only update if new result is strictly better (ratchet)."""
        if score > self.best_score:
            self.best_hypothesis = hypothesis
            self.best_score = score
            self.best_evidence = evidence
            self.version += 1
            self.timestamp = datetime.now()
            return True
        return False

class autoconstitutionOrchestrator:
    """
    Master orchestrator for parallel research exploration.
    
    Implements PARL (Parallel Autoregressive Learning) for research:
    - Decomposes problems into parallel exploration branches
    - Spawns and manages specialized research agents
    - Monitors gradient (progress rate) of each branch
    - Dynamically reallocates agents based on gradient signals
    - Maintains global ratchet state of best findings
    """
    
    def __init__(
        self,
        context: ResearchContext,
        max_concurrent_agents: int = 10,
        reallocation_threshold: float = 0.1,
        stagnation_timeout_seconds: int = 60
    ):
        self.context = context
        self.max_concurrent_agents = max_concurrent_agents
        self.reallocation_threshold = reallocation_threshold
        self.stagnation_timeout_seconds = stagnation_timeout_seconds
        
        # Core state
        self.global_state = GlobalRatchetState()
        self.branches: Dict[UUID, 'ResearchBranch'] = {}
        self.agents: Dict[UUID, 'ResearchAgent'] = {}
        self.metrics: Dict[UUID, PerformanceMetrics] = {}
        
        # Event system
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.subscribers: Dict[str, List[Callable]] = {}
        
        # Control
        self._running = False
        self._reallocation_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
    
    # ==================== Core Orchestration Methods ====================
    
    async def start_research(self) -> GlobalRatchetState:
        """
        Begin the research process.
        
        1. Decompose problem into initial branches
        2. Spawn seed agents
        3. Start monitoring and reallocation loops
        4. Return final state when complete
        """
        self._running = True
        
        # Start background tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._reallocation_task = asyncio.create_task(self._reallocation_loop())
        
        # Initial decomposition and agent spawning
        initial_branches = await self.decompose_problem(self.context.problem_statement)
        
        for branch in initial_branches:
            await self.spawn_branch(branch)
        
        # Wait for completion or budget exhaustion
        await self._wait_for_completion()
        
        return self.global_state
    
    async def stop_research(self) -> GlobalRatchetState:
        """Gracefully terminate all research activities."""
        self._running = False
        
        # Cancel background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._reallocation_task:
            self._reallocation_task.cancel()
        
        # Terminate all agents
        for agent in list(self.agents.values()):
            await agent.terminate()
        
        return self.global_state
    
    async def pause_branch(self, branch_id: UUID) -> None:
        """Pause all agents in a branch."""
        branch = self.branches.get(branch_id)
        if branch:
            branch.status = BranchStatus.STAGNANT
            for agent_id in branch.agent_ids:
                await self.pause_agent(agent_id)
    
    async def resume_branch(self, branch_id: UUID) -> None:
        """Resume a paused branch."""
        branch = self.branches.get(branch_id)
        if branch:
            branch.status = BranchStatus.EXPLORING
            for agent_id in branch.agent_ids:
                await self.resume_agent(agent_id)
    
    # ==================== Abstract Methods for Implementation ====================
    
    @abstractmethod
    async def decompose_problem(self, problem: str) -> List['ResearchBranch']:
        """
        Decompose research problem into parallel exploration branches.
        
        Strategies:
        - Hypothesis-driven: Split by competing hypotheses
        - Methodology-driven: Split by different approaches
        - Subproblem-driven: Split by problem components
        - Abstraction-level: Split by granularity
        """
        pass
    
    @abstractmethod
    async def spawn_agent(self, config: AgentConfig) -> 'ResearchAgent':
        """Create and initialize a new research agent."""
        pass
    
    @abstractmethod
    async def evaluate_gradient(self, branch_id: UUID) -> float:
        """
        Calculate progress gradient for a branch.
        
        Returns score in [0, 1] representing rate of meaningful progress.
        Higher = more promising branch.
        """
        pass
    
    @abstractmethod
    async def should_reallocate(
        self, 
        source_branch: UUID, 
        target_branch: UUID
    ) -> bool:
        """Determine if reallocation from source to target is beneficial."""
        pass
```

### 2.2 Research Branch and Agent Classes

```python
@dataclass
class ResearchBranch:
    """A parallel line of research exploration."""
    branch_id: UUID = field(default_factory=uuid4)
    parent_branch_id: Optional[UUID] = None
    status: BranchStatus = BranchStatus.EXPLORING
    
    # Exploration parameters
    focus_area: str = ""  # What this branch investigates
    approach: str = ""    # Methodology being used
    depth: int = 0        # Tree depth
    
    # Agent management
    agent_ids: List[UUID] = field(default_factory=list)
    max_agents: int = 3
    
    # Progress tracking
    hypotheses: List[Dict] = field(default_factory=list)
    findings: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Gradient tracking
    gradient_history: List[float] = field(default_factory=list)
    stagnation_count: int = 0

class ResearchAgent(ABC):
    """Base class for research sub-agents."""
    
    def __init__(
        self,
        agent_id: UUID,
        config: AgentConfig,
        orchestrator: autoconstitutionOrchestrator
    ):
        self.agent_id = agent_id
        self.config = config
        self.orchestrator = orchestrator
        self.status = AgentStatus.PENDING
        
        # State
        self.current_hypothesis: Optional[str] = None
        self.iteration_count = 0
        self.findings: List[Dict] = []
        
        # Communication
        self.message_inbox: asyncio.Queue = asyncio.Queue()
        self.result_outbox: asyncio.Queue = asyncio.Queue()
    
    @abstractmethod
    async def initialize(self) -> None:
        """Prepare agent for research."""
        pass
    
    @abstractmethod
    async def research_step(self) -> Dict[str, Any]:
        """Execute one research iteration."""
        pass
    
    @abstractmethod
    async def handle_message(self, message: Dict[str, Any]) -> None:
        """Process incoming messages from orchestrator or peers."""
        pass
    
    @abstractmethod
    async def synthesize_findings(self) -> Dict[str, Any]:
        """Compile findings into structured output."""
        pass
    
    async def run(self) -> None:
        """Main agent execution loop."""
        self.status = AgentStatus.ACTIVE
        await self.initialize()
        
        while self.status == AgentStatus.ACTIVE:
            try:
                # Check for messages
                if not self.message_inbox.empty():
                    msg = await asyncio.wait_for(
                        self.message_inbox.get(), 
                        timeout=0.1
                    )
                    await self.handle_message(msg)
                
                # Execute research step
                result = await self.research_step()
                self.iteration_count += 1
                
                # Report results
                await self.result_outbox.put({
                    'agent_id': self.agent_id,
                    'iteration': self.iteration_count,
                    'result': result,
                    'timestamp': datetime.now()
                })
                
                # Check for pause/termination
                if self.status != AgentStatus.ACTIVE:
                    break
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.status = AgentStatus.FAILED
                await self.orchestrator.event_queue.put({
                    'type': 'agent_failed',
                    'agent_id': self.agent_id,
                    'error': str(e)
                })
    
    async def pause(self) -> None:
        """Pause agent execution."""
        if self.status == AgentStatus.ACTIVE:
            self.status = AgentStatus.PAUSED
    
    async def resume(self) -> None:
        """Resume agent execution."""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.ACTIVE
    
    async def terminate(self) -> None:
        """Gracefully terminate agent."""
        self.status = AgentStatus.TERMINATED
```

---

## 3. Task Decomposition Strategies

### 3.1 Decomposition Interface

```python
from typing import Protocol

class DecompositionStrategy(Protocol):
    """Protocol for problem decomposition strategies."""
    
    async def decompose(
        self, 
        problem: str, 
        context: ResearchContext
    ) -> List[ResearchBranch]:
        """Decompose problem into parallel branches."""
        ...
    
    async def estimate_complexity(self, subproblem: str) -> float:
        """Estimate complexity score for resource allocation."""
        ...

class HypothesisDrivenDecomposition:
    """
    Decompose by generating competing hypotheses.
    
    Each branch explores a different hypothesis about the solution.
    Best for: Scientific discovery, root cause analysis
    """
    
    async def decompose(
        self,
        problem: str,
        context: ResearchContext
    ) -> List[ResearchBranch]:
        # Generate diverse hypotheses
        hypotheses = await self._generate_hypotheses(problem, context)
        
        branches = []
        for i, hypothesis in enumerate(hypotheses):
            branch = ResearchBranch(
                focus_area=f"Hypothesis {i+1}: {hypothesis[:100]}...",
                approach="hypothesis_validation",
                depth=0,
                max_agents=2 + (len(hypotheses) - i) // 3  # More agents for top hypotheses
            )
            branch.hypotheses.append({
                'text': hypothesis,
                'confidence': 1.0 - (i * 0.1),
                'status': 'untested'
            })
            branches.append(branch)
        
        return branches
    
    async def _generate_hypotheses(
        self, 
        problem: str, 
        context: ResearchContext
    ) -> List[str]:
        """Generate ranked list of competing hypotheses."""
        # Implementation uses LLM to generate diverse hypotheses
        pass

class MethodologyDrivenDecomposition:
    """
    Decompose by applying different methodologies.
    
    Each branch uses a different approach to solve the same problem.
    Best for: Engineering problems, optimization tasks
    """
    
    METHODOLOGIES = [
        'analytical',      # Formal analysis and proof
        'empirical',       # Data-driven experimentation
        'simulation',      # Model-based testing
        'literature',      # Knowledge synthesis
        'analogical',      # Cross-domain transfer
        'abductive',       # Inference to best explanation
    ]
    
    async def decompose(
        self,
        problem: str,
        context: ResearchContext
    ) -> List[ResearchBranch]:
        branches = []
        
        for method in self.METHODOLOGIES:
            if await self._is_applicable(method, problem, context):
                branch = ResearchBranch(
                    focus_area=f"Method: {method}",
                    approach=method,
                    depth=0,
                    max_agents=2
                )
                branches.append(branch)
        
        return branches
    
    async def _is_applicable(
        self, 
        method: str, 
        problem: str, 
        context: ResearchContext
    ) -> bool:
        """Check if methodology is applicable to problem."""
        # Implementation uses domain knowledge
        pass

class SubproblemDrivenDecomposition:
    """
    Decompose problem into independent subproblems.
    
    Each branch tackles a component of the larger problem.
    Best for: Complex systems, multi-faceted problems
    """
    
    async def decompose(
        self,
        problem: str,
        context: ResearchContext
    ) -> List[ResearchBranch]:
        # Identify subproblems
        subproblems = await self._identify_subproblems(problem, context)
        
        branches = []
        for subproblem in subproblems:
            complexity = await self.estimate_complexity(subproblem)
            branch = ResearchBranch(
                focus_area=subproblem,
                approach="component_solving",
                depth=0,
                max_agents=max(1, int(complexity * 2))
            )
            branches.append(branch)
        
        return branches
    
    async def _identify_subproblems(
        self, 
        problem: str, 
        context: ResearchContext
    ) -> List[str]:
        """Break problem into independent components."""
        pass
    
    async def estimate_complexity(self, subproblem: str) -> float:
        """Estimate complexity on 0-1 scale."""
        pass

class AbstractionLevelDecomposition:
    """
    Decompose by exploring at different abstraction levels.
    
    Branches explore from high-level concepts to low-level details.
    Best for: Architecture design, system understanding
    """
    
    LEVELS = [
        ('conceptual', 'Abstract principles and relationships'),
        ('architectural', 'System structure and components'),
        ('algorithmic', 'Procedures and algorithms'),
        ('implementation', 'Concrete details and code'),
    ]
    
    async def decompose(
        self,
        problem: str,
        context: ResearchContext
    ) -> List[ResearchBranch]:
        branches = []
        
        for level_name, description in self.LEVELS:
            branch = ResearchBranch(
                focus_area=f"Level: {level_name}",
                approach=f"abstraction_{level_name}",
                depth=0,
                max_agents=2
            )
            branches.append(branch)
        
        return branches
```

### 3.2 Adaptive Decomposition

```python
class AdaptiveDecomposer:
    """
    Dynamically selects and combines decomposition strategies.
    
    Analyzes problem characteristics to choose optimal strategy.
    """
    
    def __init__(self):
        self.strategies = {
            'hypothesis': HypothesisDrivenDecomposition(),
            'methodology': MethodologyDrivenDecomposition(),
            'subproblem': SubproblemDrivenDecomposition(),
            'abstraction': AbstractionLevelDecomposition(),
        }
    
    async def analyze_and_decompose(
        self,
        problem: str,
        context: ResearchContext
    ) -> List[ResearchBranch]:
        """Analyze problem and apply best decomposition strategy."""
        
        # Analyze problem characteristics
        characteristics = await self._analyze_problem(problem)
        
        # Select primary strategy
        primary_strategy = self._select_strategy(characteristics)
        
        # Apply decomposition
        branches = await primary_strategy.decompose(problem, context)
        
        # Potentially apply secondary decomposition to complex branches
        for branch in branches:
            complexity = await primary_strategy.estimate_complexity(
                branch.focus_area
            )
            if complexity > 0.7 and context.max_depth > branch.depth:
                sub_branches = await self._further_decompose(branch, context)
                # Link sub-branches as children
                for sub in sub_branches:
                    sub.parent_branch_id = branch.branch_id
        
        return branches
    
    async def _analyze_problem(self, problem: str) -> Dict[str, float]:
        """Extract problem characteristics."""
        return {
            'hypothesis_space_size': 0.0,  # How many competing explanations
            'methodology_diversity': 0.0,  # How many approaches applicable
            'component_coupling': 0.0,     # How independent are subproblems
            'abstraction_range': 0.0,      # Range of relevant abstraction levels
        }
    
    def _select_strategy(
        self, 
        characteristics: Dict[str, float]
    ) -> DecompositionStrategy:
        """Choose best strategy based on characteristics."""
        scores = {
            'hypothesis': characteristics.get('hypothesis_space_size', 0),
            'methodology': characteristics.get('methodology_diversity', 0),
            'subproblem': 1 - characteristics.get('component_coupling', 0.5),
            'abstraction': characteristics.get('abstraction_range', 0),
        }
        
        best = max(scores, key=scores.get)
        return self.strategies[best]
```

---

## 4. Agent Lifecycle Management

### 4.1 Lifecycle Manager

```python
class AgentLifecycleManager:
    """Manages the complete lifecycle of research agents."""
    
    def __init__(self, orchestrator: autoconstitutionOrchestrator):
        self.orchestrator = orchestrator
        self.agent_factory = AgentFactory()
        self.active_tasks: Dict[UUID, asyncio.Task] = {}
    
    async def spawn_agent(
        self, 
        config: AgentConfig,
        branch_id: UUID
    ) -> UUID:
        """
        Create and start a new agent.
        
        Lifecycle: PENDING -> INITIALIZING -> ACTIVE
        """
        # Check resource limits
        if len(self.orchestrator.agents) >= self.orchestrator.max_concurrent_agents:
            raise ResourceLimitError("Maximum concurrent agents reached")
        
        # Create agent
        agent_id = uuid4()
        agent = self.agent_factory.create(config, agent_id, self.orchestrator)
        
        # Register
        self.orchestrator.agents[agent_id] = agent
        self.orchestrator.branches[branch_id].agent_ids.append(agent_id)
        
        # Initialize metrics
        self.orchestrator.metrics[agent_id] = PerformanceMetrics(
            agent_id=agent_id,
            branch_id=branch_id,
            start_time=datetime.now()
        )
        
        # Start agent task
        agent.status = AgentStatus.INITIALIZING
        task = asyncio.create_task(self._run_agent_wrapper(agent))
        self.active_tasks[agent_id] = task
        
        # Emit event
        await self.orchestrator._emit_event('agent_spawned', {
            'agent_id': agent_id,
            'branch_id': branch_id,
            'config': config
        })
        
        return agent_id
    
    async def _run_agent_wrapper(self, agent: ResearchAgent) -> None:
        """Wrapper to handle agent execution and cleanup."""
        try:
            await agent.run()
        except asyncio.CancelledError:
            agent.status = AgentStatus.TERMINATED
        except Exception as e:
            agent.status = AgentStatus.FAILED
            await self.orchestrator.event_queue.put({
                'type': 'agent_error',
                'agent_id': agent.agent_id,
                'error': str(e)
            })
        finally:
            # Cleanup
            if agent.agent_id in self.active_tasks:
                del self.active_tasks[agent.agent_id]
    
    async def pause_agent(self, agent_id: UUID) -> None:
        """Pause agent execution (preserves state)."""
        agent = self.orchestrator.agents.get(agent_id)
        if agent:
            await agent.pause()
            
            # Cancel task but keep agent
            task = self.active_tasks.get(agent_id)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def resume_agent(self, agent_id: UUID) -> None:
        """Resume paused agent."""
        agent = self.orchestrator.agents.get(agent_id)
        if agent and agent.status == AgentStatus.PAUSED:
            task = asyncio.create_task(self._run_agent_wrapper(agent))
            self.active_tasks[agent_id] = task
    
    async def terminate_agent(
        self, 
        agent_id: UUID,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Terminate agent and return final findings.
        
        Lifecycle: ACTIVE -> TERMINATED
        """
        agent = self.orchestrator.agents.get(agent_id)
        if not agent:
            return {}
        
        # Cancel task
        task = self.active_tasks.get(agent_id)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Get final synthesis
        findings = await agent.synthesize_findings()
        
        # Update status
        agent.status = AgentStatus.TERMINATED
        
        # Emit event
        await self.orchestrator._emit_event('agent_terminated', {
            'agent_id': agent_id,
            'reason': reason,
            'findings': findings
        })
        
        return findings
    
    async def migrate_agent(
        self,
        agent_id: UUID,
        source_branch: UUID,
        target_branch: UUID
    ) -> None:
        """
        Move agent from one branch to another.
        
        Lifecycle: ACTIVE -> REALLOCATING -> ACTIVE
        """
        agent = self.orchestrator.agents.get(agent_id)
        if not agent:
            return
        
        # Mark as reallocating
        old_status = agent.status
        agent.status = AgentStatus.REALLOCATING
        
        # Update branch memberships
        self.orchestrator.branches[source_branch].agent_ids.remove(agent_id)
        self.orchestrator.branches[target_branch].agent_ids.append(agent_id)
        
        # Update metrics
        self.orchestrator.metrics[agent_id].branch_id = target_branch
        
        # Reconfigure agent for new branch
        await self._reconfigure_agent(agent, target_branch)
        
        # Restore status
        agent.status = old_status
        
        # Emit event
        await self.orchestrator._emit_event('agent_migrated', {
            'agent_id': agent_id,
            'source_branch': source_branch,
            'target_branch': target_branch
        })
    
    async def _reconfigure_agent(
        self, 
        agent: ResearchAgent, 
        branch_id: UUID
    ) -> None:
        """Reconfigure agent for new branch context."""
        branch = self.orchestrator.branches[branch_id]
        
        # Update agent's hypotheses and focus
        agent.config.hypotheses = [h['text'] for h in branch.hypotheses]
        
        # Send reconfiguration message
        await agent.handle_message({
            'type': 'reconfigure',
            'new_focus': branch.focus_area,
            'context': branch.findings
        })

class AgentFactory:
    """Factory for creating specialized research agents."""
    
    AGENT_TYPES = {
        'explorer': ExplorerAgent,
        'critic': CriticAgent,
        'synthesizer': SynthesizerAgent,
        'verifier': VerifierAgent,
    }
    
    def create(
        self,
        config: AgentConfig,
        agent_id: UUID,
        orchestrator: autoconstitutionOrchestrator
    ) -> ResearchAgent:
        """Create agent instance based on configuration."""
        agent_class = self.AGENT_TYPES.get(config.agent_type, ExplorerAgent)
        return agent_class(agent_id, config, orchestrator)
```

### 4.2 Specialized Agent Types

```python
class ExplorerAgent(ResearchAgent):
    """Agent that explores new hypotheses and generates candidates."""
    
    async def research_step(self) -> Dict[str, Any]:
        """Generate and evaluate new hypotheses."""
        # Generate candidate hypotheses
        candidates = await self._generate_candidates()
        
        # Evaluate each candidate
        evaluated = []
        for candidate in candidates:
            score = await self._evaluate_candidate(candidate)
            evaluated.append({
                'hypothesis': candidate,
                'score': score,
                'evidence': []
            })
        
        # Return best candidates
        evaluated.sort(key=lambda x: x['score'], reverse=True)
        return {
            'candidates': evaluated[:3],
            'exploration_coverage': len(candidates)
        }
    
    async def _generate_candidates(self) -> List[str]:
        """Generate candidate hypotheses."""
        pass
    
    async def _evaluate_candidate(self, candidate: str) -> float:
        """Score candidate hypothesis."""
        pass

class CriticAgent(ResearchAgent):
    """Agent that evaluates and challenges hypotheses."""
    
    async def research_step(self) -> Dict[str, Any]:
        """Critique current hypotheses in branch."""
        branch = self.orchestrator.branches[self.config.parent_branch_id]
        
        critiques = []
        for hypothesis in branch.hypotheses:
            critique = await self._critique_hypothesis(hypothesis)
            critiques.append(critique)
        
        return {
            'critiques': critiques,
            'identified_flaws': sum(1 for c in critiques if c['severity'] > 0.5)
        }
    
    async def _critique_hypothesis(self, hypothesis: Dict) -> Dict:
        """Analyze hypothesis for weaknesses."""
        pass

class SynthesizerAgent(ResearchAgent):
    """Agent that consolidates findings across branches."""
    
    async def research_step(self) -> Dict[str, Any]:
        """Synthesize findings from multiple sources."""
        # Gather findings from related branches
        findings = await self._gather_findings()
        
        # Identify patterns and connections
        synthesis = await self._synthesize(findings)
        
        # Update global knowledge
        await self._update_knowledge_graph(synthesis)
        
        return {
            'synthesis': synthesis,
            'connections_found': len(synthesis.get('connections', []))
        }
    
    async def _gather_findings(self) -> List[Dict]:
        """Collect findings from peer branches."""
        pass
    
    async def _synthesize(self, findings: List[Dict]) -> Dict:
        """Create unified understanding from findings."""
        pass

class VerifierAgent(ResearchAgent):
    """Agent that validates hypotheses with evidence."""
    
    async def research_step(self) -> Dict[str, Any]:
        """Verify hypotheses through validation."""
        branch = self.orchestrator.branches[self.config.parent_branch_id]
        
        verifications = []
        for hypothesis in branch.hypotheses:
            if hypothesis['status'] == 'untested':
                result = await self._verify_hypothesis(hypothesis)
                verifications.append(result)
        
        return {
            'verifications': verifications,
            'validated_count': sum(1 for v in verifications if v['validated'])
        }
    
    async def _verify_hypothesis(self, hypothesis: Dict) -> Dict:
        """Attempt to validate hypothesis."""
        pass
```

---

## 5. Performance Monitoring

### 5.1 Metrics Collection

```python
class PerformanceMonitor:
    """Monitors and analyzes agent/branch performance."""
    
    def __init__(self, orchestrator: autoconstitutionOrchestrator):
        self.orchestrator = orchestrator
        self.metric_history: Dict[UUID, List[PerformanceMetrics]] = {}
        self.gradient_window_size = 5
    
    async def record_iteration(
        self,
        agent_id: UUID,
        result: Dict[str, Any]
    ) -> None:
        """Record metrics from agent iteration."""
        metrics = self.orchestrator.metrics.get(agent_id)
        if not metrics:
            return
        
        # Update basic counters
        metrics.iterations += 1
        metrics.hypotheses_generated += result.get('candidates', [])
        metrics.hypotheses_validated += result.get('validated_count', 0)
        
        # Record knowledge contribution
        if 'candidates' in result:
            for candidate in result['candidates']:
                metrics.knowledge_contributions.append({
                    'type': 'hypothesis',
                    'content': candidate['hypothesis'],
                    'score': candidate['score'],
                    'timestamp': datetime.now()
                })
        
        # Update history
        if agent_id not in self.metric_history:
            self.metric_history[agent_id] = []
        self.metric_history[agent_id].append(metrics)
        
        # Trim history
        if len(self.metric_history[agent_id]) > 100:
            self.metric_history[agent_id] = self.metric_history[agent_id][-100:]
    
    async def calculate_branch_gradient(self, branch_id: UUID) -> float:
        """
        Calculate progress gradient for a branch.
        
        Gradient measures rate of meaningful progress.
        High gradient = branch making rapid discoveries
        Low gradient = branch stagnating
        """
        branch = self.orchestrator.branches.get(branch_id)
        if not branch:
            return 0.0
        
        # Collect agent metrics for branch
        agent_metrics = [
            self.orchestrator.metrics[aid]
            for aid in branch.agent_ids
            if aid in self.orchestrator.metrics
        ]
        
        if not agent_metrics:
            return 0.0
        
        # Calculate multi-factor gradient
        factors = {
            'knowledge_rate': self._knowledge_rate(agent_metrics),
            'quality_improvement': self._quality_improvement(agent_metrics),
            'exploration_efficiency': self._exploration_efficiency(agent_metrics),
            'resource_efficiency': self._resource_efficiency(agent_metrics),
        }
        
        # Weighted combination
        weights = {
            'knowledge_rate': 0.35,
            'quality_improvement': 0.30,
            'exploration_efficiency': 0.20,
            'resource_efficiency': 0.15,
        }
        
        gradient = sum(
            factors[k] * weights[k] 
            for k in factors
        )
        
        # Update branch history
        branch.gradient_history.append(gradient)
        if len(branch.gradient_history) > self.gradient_window_size:
            branch.gradient_history = branch.gradient_history[-self.gradient_window_size:]
        
        return gradient
    
    def _knowledge_rate(self, metrics: List[PerformanceMetrics]) -> float:
        """Rate of new knowledge generation."""
        if not metrics:
            return 0.0
        
        total_contributions = sum(
            len(m.knowledge_contributions) 
            for m in metrics
        )
        total_iterations = sum(m.iterations for m in metrics)
        
        if total_iterations == 0:
            return 0.0
        
        return min(1.0, total_contributions / (total_iterations * 0.5))
    
    def _quality_improvement(self, metrics: List[PerformanceMetrics]) -> float:
        """Rate of quality improvement in contributions."""
        if not metrics or not any(m.knowledge_contributions for m in metrics):
            return 0.0
        
        # Track score trends
        all_scores = []
        for m in metrics:
            for contrib in m.knowledge_contributions:
                all_scores.append(contrib['score'])
        
        if len(all_scores) < 2:
            return 0.5  # Neutral if insufficient data
        
        # Calculate trend
        recent_avg = sum(all_scores[-5:]) / min(5, len(all_scores[-5:]))
        older_avg = sum(all_scores[:-5]) / max(1, len(all_scores[:-5]))
        
        improvement = recent_avg - older_avg
        return 0.5 + (improvement * 2)  # Center at 0.5
    
    def _exploration_efficiency(self, metrics: List[PerformanceMetrics]) -> float:
        """Efficiency of exploration (novel discoveries per iteration)."""
        if not metrics:
            return 0.0
        
        total_hypotheses = sum(m.hypotheses_generated for m in metrics)
        validated = sum(m.hypotheses_validated for m in metrics)
        
        if total_hypotheses == 0:
            return 0.0
        
        return validated / total_hypotheses
    
    def _resource_efficiency(self, metrics: List[PerformanceMetrics]) -> float:
        """Efficiency of resource utilization."""
        if not metrics:
            return 0.0
        
        # Simple metric: contributions per unit time
        total_contributions = sum(
            len(m.knowledge_contributions) 
            for m in metrics
        )
        
        elapsed = (
            datetime.now() - metrics[0].start_time
        ).total_seconds()
        
        if elapsed == 0:
            return 0.0
        
        rate = total_contributions / (elapsed / 60)  # per minute
        return min(1.0, rate / 5)  # Normalize to 5 contributions/min
    
    async def identify_stagnant_branches(self) -> List[UUID]:
        """Identify branches with low or decreasing gradient."""
        stagnant = []
        
        for branch_id, branch in self.orchestrator.branches.items():
            if len(branch.gradient_history) < 3:
                continue
            
            # Check for consistently low gradient
            recent_gradients = branch.gradient_history[-3:]
            avg_gradient = sum(recent_gradients) / len(recent_gradients)
            
            if avg_gradient < self.orchestrator.reallocation_threshold:
                branch.stagnation_count += 1
                
                # Mark as stagnant after multiple periods
                if branch.stagnation_count >= 2:
                    branch.status = BranchStatus.STAGNANT
                    stagnant.append(branch_id)
            else:
                branch.stagnation_count = max(0, branch.stagnation_count - 1)
        
        return stagnant
    
    async def identify_high_potential_branches(self) -> List[Tuple[UUID, float]]:
        """Identify branches with high gradient (sorted by potential)."""
        potentials = []
        
        for branch_id, branch in self.orchestrator.branches.items():
            if len(branch.gradient_history) < 2:
                continue
            
            recent_gradients = branch.gradient_history[-2:]
            avg_gradient = sum(recent_gradients) / len(recent_gradients)
            
            if avg_gradient > 0.5:  # Above average
                # Bonus for increasing gradient
                trend = recent_gradients[-1] - recent_gradients[0]
                potential = avg_gradient + (trend * 0.5)
                
                potentials.append((branch_id, potential))
        
        # Sort by potential (descending)
        potentials.sort(key=lambda x: x[1], reverse=True)
        
        return potentials
```

### 5.2 Monitoring Loop

```python
async def _monitoring_loop(self) -> None:
    """Background task for continuous performance monitoring."""
    monitor = PerformanceMonitor(self)
    
    while self._running:
        try:
            # Process result queue
            while not self.result_queue.empty():
                result = await asyncio.wait_for(
                    self.result_queue.get(), 
                    timeout=0.1
                )
                await monitor.record_iteration(
                    result['agent_id'],
                    result['result']
                )
            
            # Update branch gradients
            for branch_id in self.branches:
                gradient = await monitor.calculate_branch_gradient(branch_id)
                self.branches[branch_id].gradient_score = gradient
            
            # Identify status changes
            stagnant = await monitor.identify_stagnant_branches()
            high_potential = await monitor.identify_high_potential_branches()
            
            # Emit monitoring events
            if stagnant:
                await self._emit_event('branches_stagnant', {
                    'branch_ids': [str(b) for b in stagnant]
                })
            
            if high_potential:
                await self._emit_event('high_potential_branches', {
                    'branches': [
                        {'id': str(bid), 'potential': pot}
                        for bid, pot in high_potential[:3]
                    ]
                })
            
            # Check for breakthroughs
            await self._detect_breakthroughs()
            
            await asyncio.sleep(5)  # Monitoring interval
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            await self._emit_event('monitoring_error', {'error': str(e)})
            await asyncio.sleep(5)
```

---

## 6. Dynamic Reallocation Logic

### 6.1 Reallocation Engine

```python
class ReallocationEngine:
    """Decides and executes agent reallocation between branches."""
    
    def __init__(self, orchestrator: autoconstitutionOrchestrator):
        self.orchestrator = orchestrator
        self.reallocation_cooldown: Dict[UUID, datetime] = {}
        self.cooldown_seconds = 30
    
    async def evaluate_reallocations(self) -> List[Dict]:
        """
        Identify beneficial agent reallocations.
        
        Returns list of reallocation decisions:
        {
            'agent_id': UUID,
            'source_branch': UUID,
            'target_branch': UUID,
            'expected_improvement': float,
            'reason': str
        }
        """
        decisions = []
        
        # Get branch statuses
        stagnant = await self._get_stagnant_branches()
        high_potential = await self._get_high_potential_branches()
        
        # Don't reallocate if no clear gradient difference
        if not stagnant or not high_potential:
            return decisions
        
        # For each stagnant branch, consider reallocating agents
        for stale_branch_id in stagnant:
            agents = self.orchestrator.branches[stale_branch_id].agent_ids
            
            for agent_id in list(agents):
                # Check cooldown
                if not self._check_cooldown(agent_id):
                    continue
                
                # Find best target branch
                target = await self._find_best_target(
                    agent_id, 
                    stale_branch_id,
                    high_potential
                )
                
                if target:
                    improvement = await self._estimate_improvement(
                        agent_id,
                        stale_branch_id,
                        target
                    )
                    
                    if improvement > 0.2:  # Threshold for reallocation
                        decisions.append({
                            'agent_id': agent_id,
                            'source_branch': stale_branch_id,
                            'target_branch': target,
                            'expected_improvement': improvement,
                            'reason': f'Migrating from stagnant branch to high-potential branch'
                        })
        
        # Sort by expected improvement
        decisions.sort(key=lambda x: x['expected_improvement'], reverse=True)
        
        return decisions
    
    async def _get_stagnant_branches(self) -> List[UUID]:
        """Get branches marked as stagnant."""
        return [
            bid for bid, branch in self.orchestrator.branches.items()
            if branch.status == BranchStatus.STAGNANT
        ]
    
    async def _get_high_potential_branches(self) -> List[Tuple[UUID, float]]:
        """Get branches with high gradient potential."""
        potentials = []
        
        for bid, branch in self.orchestrator.branches.items():
            if branch.status == BranchStatus.STAGNANT:
                continue
            
            if len(branch.gradient_history) >= 2:
                avg = sum(branch.gradient_history[-2:]) / 2
                if avg > 0.4:
                    potentials.append((bid, avg))
        
        potentials.sort(key=lambda x: x[1], reverse=True)
        return potentials
    
    async def _find_best_target(
        self,
        agent_id: UUID,
        source_branch: UUID,
        candidates: List[Tuple[UUID, float]]
    ) -> Optional[UUID]:
        """Find best target branch for agent migration."""
        agent = self.orchestrator.agents.get(agent_id)
        if not agent:
            return None
        
        best_target = None
        best_score = 0.0
        
        for target_id, potential in candidates:
            if target_id == source_branch:
                continue
            
            target_branch = self.orchestrator.branches[target_id]
            
            # Skip if target is at capacity
            if len(target_branch.agent_ids) >= target_branch.max_agents:
                continue
            
            # Calculate fit score
            fit = await self._calculate_agent_branch_fit(agent, target_branch)
            score = potential * fit
            
            if score > best_score:
                best_score = score
                best_target = target_id
        
        return best_target
    
    async def _calculate_agent_branch_fit(
        self,
        agent: ResearchAgent,
        branch: ResearchBranch
    ) -> float:
        """Calculate how well agent fits target branch (0-1)."""
        # Factor 1: Specialization match
        spec_match = self._specialization_match(
            agent.config.specialization,
            branch.focus_area
        )
        
        # Factor 2: Agent type suitability
        type_suitability = self._agent_type_suitability(
            agent.config.agent_type,
            branch.approach
        )
        
        # Factor 3: Depth appropriateness
        depth_fit = 1.0 - abs(agent.config.depth - branch.depth) * 0.2
        
        return (spec_match * 0.4 + type_suitability * 0.4 + depth_fit * 0.2)
    
    def _specialization_match(self, agent_spec: str, branch_focus: str) -> float:
        """Calculate specialization match score."""
        # Simple implementation - can be enhanced with embeddings
        agent_terms = set(agent_spec.lower().split())
        branch_terms = set(branch_focus.lower().split())
        
        if not agent_terms or not branch_terms:
            return 0.5
        
        intersection = agent_terms & branch_terms
        union = agent_terms | branch_terms
        
        return len(intersection) / len(union)
    
    def _agent_type_suitability(self, agent_type: str, approach: str) -> float:
        """Calculate agent type suitability for approach."""
        suitability_map = {
            'explorer': {
                'hypothesis_validation': 0.9,
                'component_solving': 0.8,
                'abstraction_conceptual': 0.9,
            },
            'critic': {
                'hypothesis_validation': 0.9,
                'analytical': 0.8,
            },
            'synthesizer': {
                'abstraction_architectural': 0.9,
                'literature': 0.8,
            },
            'verifier': {
                'empirical': 0.9,
                'simulation': 0.8,
            }
        }
        
        return suitability_map.get(agent_type, {}).get(approach, 0.5)
    
    async def _estimate_improvement(
        self,
        agent_id: UUID,
        source: UUID,
        target: UUID
    ) -> float:
        """Estimate improvement from reallocation."""
        source_branch = self.orchestrator.branches[source]
        target_branch = self.orchestrator.branches[target]
        
        # Improvement = (target_gradient - source_gradient) * fit
        source_gradient = source_branch.gradient_score
        target_gradient = target_branch.gradient_score
        
        agent = self.orchestrator.agents[agent_id]
        fit = await self._calculate_agent_branch_fit(agent, target_branch)
        
        improvement = (target_gradient - source_gradient) * fit
        return max(0.0, improvement)
    
    def _check_cooldown(self, agent_id: UUID) -> bool:
        """Check if agent has cooled down from previous reallocation."""
        last_reallocation = self.reallocation_cooldown.get(agent_id)
        if not last_reallocation:
            return True
        
        elapsed = (datetime.now() - last_reallocation).total_seconds()
        return elapsed > self.cooldown_seconds
    
    async def execute_reallocation(self, decision: Dict) -> bool:
        """Execute a reallocation decision."""
        try:
            agent_id = decision['agent_id']
            source = decision['source_branch']
            target = decision['target_branch']
            
            # Execute migration
            await self.orchestrator.lifecycle_manager.migrate_agent(
                agent_id, source, target
            )
            
            # Update cooldown
            self.reallocation_cooldown[agent_id] = datetime.now()
            
            # Emit event
            await self.orchestrator._emit_event('agent_reallocated', {
                'agent_id': str(agent_id),
                'source_branch': str(source),
                'target_branch': str(target),
                'expected_improvement': decision['expected_improvement'],
                'reason': decision['reason']
            })
            
            return True
            
        except Exception as e:
            await self.orchestrator._emit_event('reallocation_failed', {
                'decision': decision,
                'error': str(e)
            })
            return False
```

### 6.2 Reallocation Loop

```python
async def _reallocation_loop(self) -> None:
    """Background task for dynamic agent reallocation."""
    engine = ReallocationEngine(self)
    
    while self._running:
        try:
            # Evaluate potential reallocations
            decisions = await engine.evaluate_reallocations()
            
            # Execute top reallocations (limit to prevent churn)
            max_reallocations = max(1, len(self.agents) // 5)
            
            for decision in decisions[:max_reallocations]:
                success = await engine.execute_reallocation(decision)
                
                if success:
                    # Brief pause between reallocations
                    await asyncio.sleep(2)
            
            # Wait before next evaluation
            await asyncio.sleep(10)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            await self._emit_event('reallocation_error', {'error': str(e)})
            await asyncio.sleep(10)
```

---

## 7. State Management

### 7.1 Global Ratchet State

```python
class StateManager:
    """Manages global state with ratchet semantics."""
    
    def __init__(self, orchestrator: autoconstitutionOrchestrator):
        self.orchestrator = orchestrator
        self.checkpoint_interval = 30  # seconds
        self._checkpoint_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start state management."""
        self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())
    
    async def stop(self) -> None:
        """Stop state management."""
        if self._checkpoint_task:
            self._checkpoint_task.cancel()
    
    async def propose_update(
        self,
        hypothesis: str,
        score: float,
        evidence: List[Dict],
        source_branch: UUID
    ) -> bool:
        """
        Propose an update to global state.
        
        Returns True if update was accepted (ratchet advanced).
        """
        state = self.orchestrator.global_state
        
        # Validate evidence
        validated_evidence = await self._validate_evidence(evidence)
        
        # Attempt ratchet update
        updated = state.update_if_better(
            hypothesis, 
            score, 
            validated_evidence
        )
        
        if updated:
            # Record contribution
            state.consolidated_findings.append({
                'hypothesis': hypothesis,
                'score': score,
                'evidence': validated_evidence,
                'source_branch': str(source_branch),
                'timestamp': datetime.now(),
                'state_version': state.version
            })
            
            # Emit breakthrough event
            await self.orchestrator._emit_event('ratchet_advanced', {
                'new_score': score,
                'hypothesis': hypothesis[:200],
                'state_version': state.version,
                'source_branch': str(source_branch)
            })
            
            # Notify all branches of improvement
            await self._broadcast_improvement(hypothesis, score, evidence)
        
        return updated
    
    async def _validate_evidence(self, evidence: List[Dict]) -> List[Dict]:
        """Validate and filter evidence."""
        validated = []
        
        for item in evidence:
            # Check evidence quality
            if self._is_valid_evidence(item):
                validated.append(item)
        
        return validated
    
    def _is_valid_evidence(self, evidence: Dict) -> bool:
        """Check if evidence meets quality standards."""
        required_fields = ['type', 'content', 'confidence']
        
        if not all(field in evidence for field in required_fields):
            return False
        
        # Confidence threshold
        if evidence.get('confidence', 0) < 0.3:
            return False
        
        return True
    
    async def _broadcast_improvement(
        self,
        hypothesis: str,
        score: float,
        evidence: List[Dict]
    ) -> None:
        """Notify all agents of global improvement."""
        message = {
            'type': 'global_improvement',
            'hypothesis': hypothesis,
            'score': score,
            'evidence_summary': [
                {'type': e['type'], 'confidence': e['confidence']}
                for e in evidence[:3]
            ],
            'state_version': self.orchestrator.global_state.version
        }
        
        for agent in self.orchestrator.agents.values():
            if agent.status == AgentStatus.ACTIVE:
                await agent.message_inbox.put(message)
    
    async def merge_branch_findings(self, branch_id: UUID) -> None:
        """Merge a branch's findings into global state."""
        branch = self.orchestrator.branches.get(branch_id)
        if not branch:
            return
        
        # Process each finding
        for finding in branch.findings:
            score = finding.get('score', 0.0)
            hypothesis = finding.get('hypothesis', '')
            evidence = finding.get('evidence', [])
            
            await self.propose_update(
                hypothesis, 
                score, 
                evidence, 
                branch_id
            )
        
        # Mark branch as merged
        branch.status = BranchStatus.MERGED
    
    async def get_state_snapshot(self) -> Dict[str, Any]:
        """Get serializable state snapshot."""
        state = self.orchestrator.global_state
        
        return {
            'version': state.version,
            'timestamp': state.timestamp.isoformat(),
            'best_score': state.best_score,
            'best_hypothesis': state.best_hypothesis,
            'best_evidence_count': len(state.best_evidence),
            'consolidated_findings_count': len(state.consolidated_findings),
            'knowledge_graph_nodes': len(state.knowledge_graph.get('nodes', [])),
            'knowledge_graph_edges': len(state.knowledge_graph.get('edges', [])),
        }
    
    async def _checkpoint_loop(self) -> None:
        """Periodic state checkpointing."""
        while self.orchestrator._running:
            try:
                await asyncio.sleep(self.checkpoint_interval)
                
                snapshot = await self.get_state_snapshot()
                
                await self.orchestrator._emit_event('state_checkpoint', snapshot)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.orchestrator._emit_event('checkpoint_error', {
                    'error': str(e)
                })
```

### 7.2 Knowledge Graph Management

```python
class KnowledgeGraphManager:
    """Manages the evolving knowledge graph."""
    
    def __init__(self, state: GlobalRatchetState):
        self.state = state
        self._ensure_structure()
    
    def _ensure_structure(self) -> None:
        """Ensure knowledge graph has required structure."""
        if 'nodes' not in self.state.knowledge_graph:
            self.state.knowledge_graph['nodes'] = []
        if 'edges' not in self.state.knowledge_graph:
            self.state.knowledge_graph['edges'] = []
        if 'concepts' not in self.state.knowledge_graph:
            self.state.knowledge_graph['concepts'] = {}
    
    async def add_finding(self, finding: Dict) -> str:
        """Add a finding to the knowledge graph."""
        node_id = f"finding_{len(self.state.knowledge_graph['nodes'])}"
        
        node = {
            'id': node_id,
            'type': 'finding',
            'content': finding.get('hypothesis', ''),
            'score': finding.get('score', 0.0),
            'timestamp': datetime.now().isoformat(),
            'evidence': finding.get('evidence', []),
            'metadata': finding.get('metadata', {})
        }
        
        self.state.knowledge_graph['nodes'].append(node)
        
        # Extract and link concepts
        concepts = await self._extract_concepts(finding.get('hypothesis', ''))
        for concept in concepts:
            await self._link_concept(node_id, concept)
        
        return node_id
    
    async def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Implementation uses NLP to extract concepts
        pass
    
    async def _link_concept(self, node_id: str, concept: str) -> None:
        """Create link between finding and concept."""
        concepts = self.state.knowledge_graph['concepts']
        
        if concept not in concepts:
            concepts[concept] = {
                'id': f"concept_{len(concepts)}",
                'name': concept,
                'first_seen': datetime.now().isoformat(),
                'related_findings': []
            }
        
        concepts[concept]['related_findings'].append(node_id)
        
        # Create edge
        edge = {
            'source': node_id,
            'target': concepts[concept]['id'],
            'type': 'mentions',
            'weight': 1.0
        }
        
        self.state.knowledge_graph['edges'].append(edge)
    
    async def find_related_findings(self, concept: str) -> List[Dict]:
        """Find all findings related to a concept."""
        concepts = self.state.knowledge_graph.get('concepts', {})
        
        if concept not in concepts:
            return []
        
        finding_ids = concepts[concept]['related_findings']
        nodes = self.state.knowledge_graph.get('nodes', [])
        
        return [
            node for node in nodes 
            if node['id'] in finding_ids
        ]
    
    async def get_concept_clusters(self) -> List[List[str]]:
        """Cluster related concepts."""
        # Simple clustering based on shared findings
        concepts = self.state.knowledge_graph.get('concepts', {})
        
        # Build similarity matrix
        similarities = {}
        for c1, data1 in concepts.items():
            for c2, data2 in concepts.items():
                if c1 >= c2:
                    continue
                
                shared = set(data1['related_findings']) & set(data2['related_findings'])
                if shared:
                    similarities[(c1, c2)] = len(shared)
        
        # Group into clusters (simple greedy approach)
        clusters = []
        used = set()
        
        for (c1, c2), strength in sorted(similarities.items(), key=lambda x: -x[1]):
            if c1 in used or c2 in used:
                continue
            
            cluster = [c1, c2]
            used.add(c1)
            used.add(c2)
            
            # Add more related concepts
            for c3 in concepts:
                if c3 in used:
                    continue
                if (c1, c3) in similarities or (c3, c1) in similarities:
                    cluster.append(c3)
                    used.add(c3)
            
            clusters.append(cluster)
        
        return clusters
```

---

## 8. Event System

```python
class EventSystem:
    """Event publishing and subscription system."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[Dict] = []
        self.max_history = 1000
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from events."""
        if event_type in self.subscribers:
            self.subscribers[event_type] = [
                h for h in self.subscribers[event_type] 
                if h != handler
            ]
    
    async def emit(self, event_type: str, data: Dict) -> None:
        """Emit an event to all subscribers."""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Notify subscribers
        handlers = self.subscribers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                # Log but don't fail
                print(f"Event handler error: {e}")

# Add to orchestrator
async def _emit_event(self, event_type: str, data: Dict) -> None:
    """Emit event through event system."""
    await self.event_system.emit(event_type, data)
```

---

## 9. Complete Orchestrator Implementation

```python
class autoconstitutionOrchestratorImpl(autoconstitutionOrchestrator):
    """Complete implementation of the autoconstitution Orchestrator."""
    
    def __init__(self, context: ResearchContext, **kwargs):
        super().__init__(context, **kwargs)
        
        # Subsystems
        self.lifecycle_manager = AgentLifecycleManager(self)
        self.state_manager = StateManager(self)
        self.event_system = EventSystem()
        
        # Queues
        self.result_queue: asyncio.Queue = asyncio.Queue()
    
    # ==================== Decomposition Implementation ====================
    
    async def decompose_problem(self, problem: str) -> List[ResearchBranch]:
        """Decompose using adaptive strategy."""
        decomposer = AdaptiveDecomposer()
        return await decomposer.analyze_and_decompose(problem, self.context)
    
    # ==================== Agent Spawning Implementation ====================
    
    async def spawn_agent(self, config: AgentConfig) -> ResearchAgent:
        """Spawn agent through lifecycle manager."""
        agent_id = await self.lifecycle_manager.spawn_agent(
            config, 
            config.parent_branch_id
        )
        return self.agents[agent_id]
    
    async def spawn_branch(self, branch: ResearchBranch) -> None:
        """Spawn initial agents for a branch."""
        self.branches[branch.branch_id] = branch
        
        # Spawn seed agents
        configs = self._generate_agent_configs(branch)
        
        for config in configs:
            config.parent_branch_id = branch.branch_id
            await self.spawn_agent(config)
    
    def _generate_agent_configs(self, branch: ResearchBranch) -> List[AgentConfig]:
        """Generate agent configurations for a branch."""
        configs = []
        
        # Primary explorer
        configs.append(AgentConfig(
            agent_type='explorer',
            specialization=branch.focus_area,
            depth=branch.depth,
            hypotheses=[h['text'] for h in branch.hypotheses],
            timeout_seconds=300
        ))
        
        # Critic for hypothesis validation
        if branch.approach == 'hypothesis_validation':
            configs.append(AgentConfig(
                agent_type='critic',
                specialization=branch.focus_area,
                depth=branch.depth,
                timeout_seconds=200
            ))
        
        # Verifier for empirical approaches
        if branch.approach in ['empirical', 'simulation']:
            configs.append(AgentConfig(
                agent_type='verifier',
                specialization=branch.focus_area,
                depth=branch.depth,
                timeout_seconds=400
            ))
        
        return configs
    
    # ==================== Gradient Evaluation Implementation ====================
    
    async def evaluate_gradient(self, branch_id: UUID) -> float:
        """Evaluate branch gradient using performance monitor."""
        monitor = PerformanceMonitor(self)
        return await monitor.calculate_branch_gradient(branch_id)
    
    # ==================== Reallocation Decision Implementation ====================
    
    async def should_reallocate(
        self, 
        source_branch: UUID, 
        target_branch: UUID
    ) -> bool:
        """Determine if reallocation is beneficial."""
        source = self.branches[source_branch]
        target = self.branches[target_branch]
        
        # Check gradient difference
        gradient_diff = target.gradient_score - source.gradient_score
        
        # Check capacity
        has_capacity = len(target.agent_ids) < target.max_agents
        
        return gradient_diff > 0.2 and has_capacity
    
    # ==================== Completion Detection ====================
    
    async def _wait_for_completion(self) -> None:
        """Wait for research completion or budget exhaustion."""
        budget_used = 0
        
        while self._running:
            # Count total iterations
            budget_used = sum(
                m.iterations 
                for m in self.metrics.values()
            )
            
            # Check budget
            if budget_used >= self.context.exploration_budget:
                await self._emit_event('budget_exhausted', {
                    'budget': self.context.exploration_budget,
                    'final_state_version': self.global_state.version
                })
                break
            
            # Check for convergence (all branches stagnant or merged)
            active_branches = [
                b for b in self.branches.values()
                if b.status in [BranchStatus.EXPLORING, BranchStatus.CONVERGING]
            ]
            
            if not active_branches and self.branches:
                await self._emit_event('research_converged', {
                    'final_state_version': self.global_state.version
                })
                break
            
            await asyncio.sleep(1)
    
    async def _detect_breakthroughs(self) -> None:
        """Detect and handle breakthrough discoveries."""
        for branch_id, branch in self.branches.items():
            if branch.status == BranchStatus.BREAKTHROUGH:
                continue
            
            # Check for breakthrough indicators
            if len(branch.gradient_history) >= 3:
                recent = branch.gradient_history[-3:]
                
                # Rapidly increasing gradient
                if recent[2] > recent[1] > recent[0] > 0.6:
                    branch.status = BranchStatus.BREAKTHROUGH
                    
                    await self._emit_event('breakthrough_detected', {
                        'branch_id': str(branch_id),
                        'gradient_history': recent,
                        'focus_area': branch.focus_area
                    })
                    
                    # Allocate additional resources
                    await self._boost_branch(branch_id)
    
    async def _boost_branch(self, branch_id: UUID) -> None:
        """Allocate additional resources to breakthrough branch."""
        branch = self.branches[branch_id]
        
        # Increase agent capacity
        branch.max_agents += 2
        
        # Spawn additional agents
        for _ in range(2):
            config = AgentConfig(
                agent_type='explorer',
                specialization=branch.focus_area,
                depth=branch.depth,
                parent_branch_id=branch_id,
                timeout_seconds=300
            )
            await self.spawn_agent(config)
        
        await self._emit_event('branch_boosted', {
            'branch_id': str(branch_id),
            'new_max_agents': branch.max_agents
        })
```

---

## 10. Usage Example

```python
async def main():
    # Define research context
    context = ResearchContext(
        problem_statement="""
        Develop a novel algorithm for distributed consensus that achieves:
        - Byzantine fault tolerance for up to f failures in 3f+1 nodes
        - Latency under 100ms for geo-distributed deployments
        - Throughput of 100k+ transactions per second
        """,
        domain="distributed_systems",
        constraints={
            'network_model': 'partially_synchronous',
            'cryptography': 'elliptic_curve',
            'max_nodes': 100
        },
        success_criteria=[
            'Formal safety proof',
            'Formal liveness proof',
            'Experimental validation',
            'Complexity analysis'
        ],
        max_depth=4,
        exploration_budget=500
    )
    
    # Create orchestrator
    orchestrator = autoconstitutionOrchestratorImpl(
        context=context,
        max_concurrent_agents=15,
        reallocation_threshold=0.15,
        stagnation_timeout_seconds=120
    )
    
    # Subscribe to events
    orchestrator.event_system.subscribe('ratchet_advanced', on_breakthrough)
    orchestrator.event_system.subscribe('agent_reallocated', on_reallocation)
    orchestrator.event_system.subscribe('breakthrough_detected', on_major_find)
    
    # Start research
    final_state = await orchestrator.start_research()
    
    # Output results
    print(f"Research complete!")
    print(f"Best hypothesis: {final_state.best_hypothesis}")
    print(f"Best score: {final_state.best_score}")
    print(f"Total findings: {len(final_state.consolidated_findings)}")
    
    return final_state

async def on_breakthrough(event):
    print(f"🎯 Breakthrough! Score: {event['data']['new_score']}")

async def on_reallocation(event):
    data = event['data']
    print(f"🔄 Agent migrated: {data['source_branch'][:8]} -> {data['target_branch'][:8]}")

async def on_major_find(event):
    print(f"🚀 Major breakthrough in branch: {event['data']['focus_area'][:50]}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 11. Summary

The autoconstitution Orchestrator implements a PARL-based multi-agent research system with the following key components:

| Component | Responsibility |
|-----------|---------------|
| **Orchestrator** | Master coordination, lifecycle management |
| **Decomposition** | Problem splitting into parallel branches |
| **Agent Lifecycle** | Spawn, pause, resume, migrate, terminate |
| **Performance Monitor** | Track gradients, identify stagnation |
| **Reallocation Engine** | Move agents from low to high-gradient branches |
| **State Manager** | Global ratchet, knowledge graph, checkpoints |
| **Event System** | Async communication between components |

### Key Design Decisions

1. **Gradient-Guided Allocation**: Resources flow to branches showing progress
2. **Global Ratchet**: Best findings only improve, never regress
3. **Dynamic Rebalancing**: Agents migrate based on real-time performance
4. **Hierarchical Decomposition**: Problems split recursively by need
5. **Event-Driven Architecture**: Loose coupling between components
