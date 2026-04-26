# autoconstitution Parallel Branch Architecture

## Executive Summary

This document defines the Git-based parallel branch architecture for autoconstitution, designed to support 100+ simultaneous research directions with automated merging, conflict resolution, and meta-agent arbitration.

---

## 1. Branching Strategy

### 1.1 Core Branch Hierarchy

```
main (production-stable)
├── develop (integration branch)
│   ├── research/ (active research branches)
│   │   ├── research/{agent-id}/{direction-id}/{timestamp}
│   │   └── research/meta/{arbitration-branch}
│   ├── experiment/ (experimental implementations)
│   │   └── experiment/{agent-id}/{exp-id}
│   └── validation/ (validation branches)
│       └── validation/{batch-id}/{test-suite}
├── release/ (release candidates)
│   └── release/{version}
└── hotfix/ (critical fixes)
    └── hotfix/{issue-id}
```

### 1.2 Naming Convention Specification

| Branch Type | Pattern | Example |
|-------------|---------|---------|
| Research | `research/{agent-id}/{direction}/{YYYYMMDD}-{seq}` | `research/alpha-7/llm-reasoning/20250115-003` |
| Experiment | `experiment/{agent-id}/{exp-name}/{timestamp}` | `experiment/beta-3/attention-mechanism/20250115-143022` |
| Validation | `validation/{batch-id}/{test-type}` | `validation/batch-47/comprehensive` |
| Integration | `integration/{sprint-id}/{feature-set}` | `integration/sprint-12/multi-modal` |
| Hotfix | `hotfix/{severity}/{issue-id}` | `hotfix/critical/SWARM-2847` |
| Archive | `archive/{date}/{original-name}` | `archive/2025-01/research-alpha-7-legacy` |

### 1.3 Agent-Specific Namespace

```
research/{agent-id}/
├── {agent-id}/hypothesis-{n}/     # Hypothesis testing branches
├── {agent-id}/impl-{n}/           # Implementation branches
├── {agent-id}/analysis-{n}/       # Analysis branches
└── {agent-id}/synthesis-{n}/      # Synthesis branches
```

### 1.4 Research Direction Categorization

```python
DIRECTION_CATEGORIES = {
    "algorithm": "alg",      # Algorithm improvements
    "architecture": "arch",  # System architecture changes
    "data": "data",          # Data pipeline changes
    "evaluation": "eval",    # Evaluation methodology
    "theory": "theory",      # Theoretical foundations
    "application": "app",    # Application-specific
    "integration": "int",    # Integration work
    "documentation": "doc"   # Documentation improvements
}
```

---

## 2. Git Worktree Architecture

### 2.1 Worktree Layout for Parallel Development

```
/autoconstitution/
├── main/                    # Primary worktree (main branch)
├── develop/                 # Integration worktree
├── worktrees/               # Agent worktrees (up to 100)
│   ├── agent-001/          # Worktree for agent-001
│   ├── agent-002/          # Worktree for agent-002
│   ├── ...
│   └── agent-100/          # Worktree for agent-100
├── shared/                  # Shared resources (not versioned)
│   ├── datasets/
│   ├── models/
│   └── cache/
└── .git/                   # Bare repository
```

### 2.2 Worktree Management Commands

```bash
# Initialize worktree for new agent
swarm-worktree init <agent-id> <branch-name>

# Add worktree for existing branch
git worktree add ../worktrees/agent-{id} research/{agent-id}/{direction}

# Prune stale worktrees
git worktree prune --verbose

# List all worktrees with branch info
git worktree list --porcelain
```

### 2.3 Automated Worktree Lifecycle

```yaml
worktree_policy:
  max_concurrent: 100
  auto_cleanup:
    inactive_after: "7d"
    archived_after: "30d"
  resource_limits:
    max_disk_per_worktree: "10GB"
    max_memory_per_worktree: "8GB"
  scheduling:
    priority_levels: [critical, high, normal, low]
    preemption: true
```

---

## 3. Merge Protocol for Validated Improvements

### 3.1 Validation-Gated Merge Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Research  │───▶│  Validation │───▶│   Review    │───▶│   Merge     │
│   Branch    │    │   Pipeline  │    │   Queue     │    │   to Main   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
  Auto-tests         Performance         Meta-agent        Octopus
  Unit tests         benchmarks          arbitration       merge
  Integration        Safety checks       Consensus         Squash
  tests              Reproducibility     voting            options
```

### 3.2 Merge Requirements Matrix

| Branch Type | Tests | Review | Benchmarks | Consensus |
|-------------|-------|--------|------------|-----------|
| hotfix/critical | Required | 1 approver | Skip | Fast-track |
| research/impl | Required | 2 approvers | Required | 60% |
| experiment | Required | 1 approver | Optional | 51% |
| validation | Required | Auto | Required | Auto |
| integration | Required | 3 approvers | Required | 75% |

### 3.3 Octopus Merge Strategy

For batch integration of multiple validated branches:

```bash
# Octopus merge for non-conflicting improvements
git checkout develop
git merge -s octopus \
  research/alpha-7/llm-reasoning/20250115-003 \
  research/beta-3/attention-mechanism/20250115-002 \
  research/gamma-1/data-pipeline/20250115-001 \
  -m "Batch integration: Sprint 12 research directions"
```

### 3.4 Merge Automation Script

```python
#!/usr/bin/env python3
"""autoconstitution Automated Merge Protocol"""

class MergeProtocol:
    def __init__(self):
        self.validation_threshold = 0.75
        self.conflict_resolution = "meta-arbitrate"
    
    def validate_branch(self, branch: str) -> ValidationResult:
        """Run full validation suite on branch."""
        return ValidationResult(
            tests=self.run_test_suite(branch),
            benchmarks=self.run_benchmarks(branch),
            safety=self.run_safety_checks(branch),
            reproducibility=self.verify_reproducibility(branch)
        )
    
    def attempt_merge(self, branches: List[str]) -> MergeResult:
        """Attempt merge with conflict detection."""
        # Check for conflicts first
        conflicts = self.detect_conflicts(branches)
        
        if not conflicts:
            return self.octopus_merge(branches)
        
        # Route to conflict resolution
        return self.resolve_conflicts(branches, conflicts)
    
    def octopus_merge(self, branches: List[str]) -> MergeResult:
        """Perform octopus merge for multiple branches."""
        merge_commit = git.merge(
            strategy="octopus",
            branches=branches,
            message=self.generate_merge_message(branches)
        )
        return MergeResult(success=True, commit=merge_commit)
```

---

## 4. Conflict Resolution for Incompatible Improvements

### 4.1 Conflict Classification

```python
class ConflictType(Enum):
    # Technical conflicts
    SYNTAX = "syntax"           # Code syntax conflicts
    SEMANTIC = "semantic"       # Semantic code conflicts
    API = "api"                 # API interface conflicts
    
    # Research conflicts
    METHODOLOGY = "methodology" # Different methodological approaches
    ASSUMPTION = "assumption"   # Conflicting assumptions
    HYPOTHESIS = "hypothesis"   # Competing hypotheses
    
    # Resource conflicts
    COMPUTE = "compute"         # Compute resource conflicts
    DATA = "data"               # Data access conflicts
    
    # Priority conflicts
    STRATEGY = "strategy"       # Strategic direction conflicts
```

### 4.2 Meta-Agent Arbitration System

```
┌─────────────────────────────────────────────────────────────────┐
│                    META-AGENT ARBITRATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Conflict   │───▶│  Analysis   │───▶│  Strategy   │         │
│  │  Detection  │    │  Engine     │    │  Selector   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                  │                  │
│         ▼                  ▼                  ▼                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Automatic  │    │  Impact     │    │  Resolution │         │
│  │  Resolution │    │  Assessment │    │  Execution  │         │
│  │  (70%)      │    │             │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                            │                     │
│                                            ▼                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              RESOLUTION STRATEGIES                       │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ • Feature flags (runtime selection)                     │    │
│  │ • Branch variants (maintain both)                       │    │
│  │ • Synthesis merge (combine approaches)                  │    │
│  │ • A/B test branch (empirical evaluation)                │    │
│  │ • Temporal separation (staged integration)              │    │
│  │ • Ablation study (isolate contributions)                │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Resolution Strategies

#### Strategy 1: Feature Flag Resolution

```python
# Automatic feature flag generation for incompatible approaches
class FeatureFlagResolution:
    def resolve(self, branch_a: Branch, branch_b: Branch) -> Resolution:
        # Generate feature flags for both approaches
        flags = {
            f"USE_{branch_a.id.upper()}": False,
            f"USE_{branch_b.id.upper()}": False,
        }
        
        # Create configurable implementation
        merged_code = f"""
# Auto-generated feature flag resolution
if config.get("{flags.keys()[0]}"):
    {branch_a.implementation}
elif config.get("{flags.keys()[1]}"):
    {branch_b.implementation}
else:
    # Default: use original
    pass
"""
        return Resolution(
            type="feature-flag",
            code=merged_code,
            flags=flags,
            requires_runtime_selection=True
        )
```

#### Strategy 2: Synthesis Merge

```python
class SynthesisMerge:
    """Merge that combines insights from both approaches."""
    
    def synthesize(self, branch_a: Branch, branch_b: Branch) -> Resolution:
        # Identify complementary aspects
        complementary = self.find_complementary_aspects(branch_a, branch_b)
        
        # Generate synthesized implementation
        synthesis = f"""
# Synthesized approach combining:
# - {branch_a.id}: {branch_a.contribution}
# - {branch_b.id}: {branch_b.contribution}

class SynthesizedApproach:
    def __init__(self):
        self.component_a = {branch_a.component}
        self.component_b = {branch_b.component}
    
    def execute(self, context):
        # Combined execution strategy
        result_a = self.component_a.execute(context)
        result_b = self.component_b.execute(context)
        return self.ensemble_results(result_a, result_b)
"""
        return Resolution(type="synthesis", code=synthesis)
```

#### Strategy 3: A/B Test Branch

```python
class ABTestResolution:
    """Create parallel branches for empirical evaluation."""
    
    def create_test_branches(self, branches: List[Branch]) -> TestConfiguration:
        # Create test branch with both variants
        test_branch = self.create_test_branch(branches)
        
        # Configure A/B testing framework
        config = {
            "branches": [b.id for b in branches],
            "metrics": self.select_evaluation_metrics(branches),
            "duration": "7d",
            "sample_size": self.calculate_sample_size(branches),
            "success_criteria": self.define_success_criteria(branches)
        }
        
        return TestConfiguration(branch=test_branch, config=config)
```

### 4.4 Arbitration Decision Tree

```
Conflict Detected
       │
       ▼
┌───────────────┐
│ Can branches  │──No──▶ Create variant branches
│ coexist?      │        (maintain both)
└───────────────┘
       │ Yes
       ▼
┌───────────────┐
│ Are changes   │──No──▶ Automatic merge
│ in same file? │
└───────────────┘
       │ Yes
       ▼
┌───────────────┐
│ Same function │──No──▶ File-level feature flag
│ or method?    │
└───────────────┘
       │ Yes
       ▼
┌───────────────┐
│ Semantic      │──No──▶ Syntax-level merge
│ conflict?     │
└───────────────┘
       │ Yes
       ▼
┌───────────────┐
│ Research      │──No──▶ Meta-agent arbitration
│ methodology?  │        (technical decision)
└───────────────┘
       │ Yes
       ▼
┌───────────────┐
│ A/B test both │
│ approaches    │
└───────────────┘
```

---

## 5. Git Automation Approach

### 5.1 Automation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SWARM GIT AUTOMATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Event-Driven Architecture                   │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  Webhooks ◄─── Git Events ◄─── Agent Actions           │    │
│  │       │              │              │                    │    │
│  │       ▼              ▼              ▼                    │    │
│  │  ┌───────────────────────────────────────────────┐      │    │
│  │  │         Message Queue (Redis/RabbitMQ)         │      │    │
│  │  └───────────────────────────────────────────────┘      │    │
│  │       │              │              │                    │    │
│  │       ▼              ▼              ▼                    │    │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐               │    │
│  │  │  Merge  │   │ Conflict│   │  Branch │               │    │
│  │  │ Handler │   │ Handler │   │ Manager │               │    │
│  │  └─────────┘   └─────────┘   └─────────┘               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Automation Components                       │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ • swarm-branch: Branch lifecycle management             │    │
│  │ • swarm-merge: Automated merge operations               │    │
│  │ • swarm-conflict: Conflict detection & resolution       │    │
│  │ • swarm-validate: Validation pipeline orchestration     │    │
│  │ • swarm-cleanup: Branch and worktree maintenance        │    │
│  │ • swarm-report: Status and analytics reporting          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Core Automation Scripts

#### Branch Management (`swarm-branch`)

```bash
#!/bin/bash
# swarm-branch: autoconstitution branch management

COMMAND=$1
shift

case $COMMAND in
    create)
        # Create new research branch with proper naming
        AGENT_ID=$1
        DIRECTION=$2
        TIMESTAMP=$(date +%Y%m%d-%H%M%S)
        BRANCH_NAME="research/${AGENT_ID}/${DIRECTION}/${TIMESTAMP}"
        
        git checkout -b "$BRANCH_NAME" develop
        git push -u origin "$BRANCH_NAME"
        
        # Create worktree
        git worktree add "../worktrees/${AGENT_ID}-${DIRECTION}" "$BRANCH_NAME"
        echo "Created branch: $BRANCH_NAME"
        ;;
    
    archive)
        # Archive inactive branch
        BRANCH=$1
        ARCHIVE_DATE=$(date +%Y-%m-%d)
        ARCHIVE_NAME="archive/${ARCHIVE_DATE}/${BRANCH//\//-}"
        
        git branch -m "$BRANCH" "$ARCHIVE_NAME"
        git push origin "$ARCHIVE_NAME"
        git push origin --delete "$BRANCH"
        echo "Archived branch: $BRANCH -> $ARCHIVE_NAME"
        ;;
    
    list)
        # List all research branches with metadata
        echo "=== Active Research Branches ==="
        git branch -r --list "origin/research/*" | while read branch; do
            LAST_COMMIT=$(git log -1 --format="%cr" "$branch")
            AUTHOR=$(git log -1 --format="%an" "$branch")
            echo "$branch | Last: $LAST_COMMIT | Author: $AUTHOR"
        done
        ;;
    
    cleanup)
        # Remove stale branches
        STALE_DAYS=${1:-7}
        git branch -r --list "origin/research/*" | while read branch; do
            LAST_COMMIT_DATE=$(git log -1 --format="%ct" "$branch")
            NOW=$(date +%s)
            AGE=$(( (NOW - LAST_COMMIT_DATE) / 86400 ))
            
            if [ $AGE -gt $STALE_DAYS ]; then
                echo "Archiving stale branch: $branch (${AGE} days old)"
                swarm-branch archive "$branch"
            fi
        done
        ;;
esac
```

#### Automated Merge (`swarm-merge`)

```python
#!/usr/bin/env python3
"""swarm-merge: Automated merge with validation and conflict resolution"""

import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class MergeStrategy(Enum):
    FAST_FORWARD = "ff"
    RECURSIVE = "recursive"
    OCTOPUS = "octopus"
    OURS = "ours"
    THEIRS = "theirs"

@dataclass
class MergeRequest:
    source_branches: List[str]
    target_branch: str
    strategy: MergeStrategy
    validation_required: bool = True

class SwarmMerge:
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.validation_pipeline = ValidationPipeline()
        self.conflict_resolver = ConflictResolver()
    
    def execute_merge(self, request: MergeRequest) -> MergeResult:
        """Execute merge with full automation."""
        
        # Step 1: Pre-merge validation
        if request.validation_required:
            for branch in request.source_branches:
                result = self.validation_pipeline.run(branch)
                if not result.passed:
                    return MergeResult(
                        success=False,
                        error=f"Validation failed for {branch}: {result.errors}"
                    )
        
        # Step 2: Conflict detection
        conflicts = self.detect_conflicts(request.source_branches)
        if conflicts:
            resolution = self.conflict_resolver.resolve(conflicts)
            if not resolution.success:
                return MergeResult(
                    success=False,
                    error=f"Conflict resolution failed: {resolution.error}"
                )
        
        # Step 3: Execute merge
        if len(request.source_branches) > 1:
            return self.octopus_merge(request)
        else:
            return self.single_merge(request)
    
    def octopus_merge(self, request: MergeRequest) -> MergeResult:
        """Execute octopus merge for multiple branches."""
        cmd = [
            "git", "merge", "-s", "octopus",
            *request.source_branches,
            "-m", self.generate_merge_message(request)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return MergeResult(
                success=True,
                commit_hash=self.get_current_commit(),
                message="Octopus merge successful"
            )
        else:
            return MergeResult(
                success=False,
                error=result.stderr
            )
    
    def detect_conflicts(self, branches: List[str]) -> List[Conflict]:
        """Detect potential conflicts between branches."""
        conflicts = []
        
        # Create temporary merge tree
        for i, branch_a in enumerate(branches):
            for branch_b in branches[i+1:]:
                # Check for file-level conflicts
                files_a = self.get_changed_files(branch_a)
                files_b = self.get_changed_files(branch_b)
                
                common_files = set(files_a) & set(files_b)
                if common_files:
                    conflicts.append(Conflict(
                        type=ConflictType.FILE,
                        branches=[branch_a, branch_b],
                        files=list(common_files)
                    ))
        
        return conflicts
    
    def generate_merge_message(self, request: MergeRequest) -> str:
        """Generate descriptive merge commit message."""
        branches_str = "\n".join(f"- {b}" for b in request.source_branches)
        return f"""Batch integration merge

Merged branches:
{branches_str}

Strategy: {request.strategy.value}
Validation: {'passed' if request.validation_required else 'skipped'}
Timestamp: {datetime.now().isoformat()}
"""
```

### 5.3 Git Hooks Integration

```bash
# .git/hooks/pre-commit
#!/bin/bash
# Pre-commit hook for autoconstitution

echo "Running autoconstitution pre-commit checks..."

# Check branch naming convention
BRANCH=$(git branch --show-current)
if [[ ! $BRANCH =~ ^(research|experiment|validation|integration|hotfix|release|main|develop|archive)/ ]]; then
    echo "ERROR: Branch name does not follow convention: $BRANCH"
    echo "Expected: research/, experiment/, validation/, integration/, hotfix/, release/, main, develop, archive/"
    exit 1
fi

# Run automated tests
python -m pytest tests/ -x --tb=short
if [ $? -ne 0 ]; then
    echo "ERROR: Tests failed. Commit aborted."
    exit 1
fi

# Run linting
python -m flake8 src/
if [ $? -ne 0 ]; then
    echo "ERROR: Linting failed. Commit aborted."
    exit 1
fi

echo "Pre-commit checks passed!"
exit 0
```

```bash
# .git/hooks/post-merge
#!/bin/bash
# Post-merge hook for autoconstitution

MERGE_COMMIT=$(git rev-parse HEAD)
PARENT_COUNT=$(git rev-list --parents -n1 $MERGE_COMMIT | wc -w)
PARENT_COUNT=$((PARENT_COUNT - 1))

# Detect octopus merge
if [ $PARENT_COUNT -gt 2 ]; then
    echo "Octopus merge detected with $PARENT_COUNT parents"
    
    # Trigger post-merge validation
    python -m swarm.validate --merge-commit $MERGE_COMMIT
    
    # Update branch metadata
    python -m swarm.metadata --update --commit $MERGE_COMMIT
fi

# Notify agents of merge
python -m swarm.notify --event merge --commit $MERGE_COMMIT
```

### 5.4 CI/CD Pipeline Integration

```yaml
# .github/workflows/swarm-automation.yml
name: autoconstitution Automation

on:
  push:
    branches:
      - 'research/**'
      - 'experiment/**'
      - 'validation/**'
  pull_request:
    branches:
      - develop
      - main

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run validation suite
        run: |
          python -m swarm.validate \
            --branch ${{ github.ref }} \
            --full-suite
      
      - name: Generate validation report
        run: |
          python -m swarm.report \
            --type validation \
            --output validation-report.json
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: validation-report
          path: validation-report.json

  merge-check:
    runs-on: ubuntu-latest
    needs: validate
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v3
      
      - name: Check merge compatibility
        run: |
          python -m swarm.merge --dry-run \
            --source ${{ github.head_ref }} \
            --target ${{ github.base_ref }}
      
      - name: Detect conflicts
        run: |
          python -m swarm.conflict --detect \
            --branches ${{ github.head_ref }},${{ github.base_ref }}

  auto-merge:
    runs-on: ubuntu-latest
    needs: [validate, merge-check]
    if: |
      github.event_name == 'pull_request' &&
      github.event.pull_request.draft == false &&
      contains(github.event.pull_request.labels.*.name, 'auto-merge')
    steps:
      - uses: actions/checkout@v3
      
      - name: Auto-merge validated PR
        run: |
          python -m swarm.merge --auto \
            --pr ${{ github.event.pull_request.number }} \
            --strategy octopus
```

---

## 6. Branch Lifecycle Management

### 6.1 Lifecycle States

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              BRANCH LIFECYCLE STATES                       │
                    └─────────────────────────────────────────────────────────┘

    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ CREATED │───▶│ ACTIVE  │───▶│VALIDATED│───▶│ MERGED  │───▶│ARCHIVED │
    └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
         │              │              │              │              │
         │              │              │              │              │
         ▼              ▼              ▼              ▼              ▼
    • Initialized  • Development  • Tests pass   • In main     • Moved to
    • Worktree     • Commits      • Review done  • Tagged      • archive/
      created      • CI running   • Benchmarks   • Notified    • Retained
    • Agent        • Docs         • Consensus      agents        1 year
      notified       updated        reached

         │              │              │              │              │
         │              │              │              │              │
         ▼              ▼              ▼              ▼              ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │  TTL:   │    │  TTL:   │    │  TTL:   │    │  TTL:   │    │  TTL:   │
    │  30d    │    │  14d    │    │  7d     │    │  ∞      │    │  365d   │
    └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘

    State Transitions:
    ─────────────────
    CREATED → ACTIVE:     First commit pushed
    ACTIVE → VALIDATED:   All validation criteria met
    ACTIVE → STALE:       No activity for 7 days
    VALIDATED → MERGED:   Successfully merged to target
    VALIDATED → CONFLICT: Merge conflict detected
    CONFLICT → ACTIVE:    Conflict resolved
    MERGED → ARCHIVED:    30 days post-merge
    STALE → ARCHIVED:     No activity for 14 additional days
```

### 6.2 Lifecycle Automation

```python
#!/usr/bin/env python3
"""swarm-lifecycle: Automated branch lifecycle management"""

from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional

class BranchState(Enum):
    CREATED = auto()
    ACTIVE = auto()
    VALIDATED = auto()
    CONFLICT = auto()
    MERGED = auto()
    STALE = auto()
    ARCHIVED = auto()
    ABANDONED = auto()

class LifecyclePolicy:
    """Defines lifecycle policies for branches."""
    
    TTL_CONFIG = {
        BranchState.CREATED: timedelta(days=30),
        BranchState.ACTIVE: timedelta(days=14),
        BranchState.VALIDATED: timedelta(days=7),
        BranchState.CONFLICT: timedelta(days=3),
        BranchState.STALE: timedelta(days=14),
        BranchState.MERGED: timedelta(days=30),
        BranchState.ARCHIVED: timedelta(days=365),
    }
    
    AUTO_ACTIONS = {
        BranchState.STALE: "notify_agent",
        BranchState.ABANDONED: "archive_branch",
    }

class BranchLifecycle:
    """Manages branch lifecycle state transitions."""
    
    def __init__(self, repo):
        self.repo = repo
        self.policy = LifecyclePolicy()
    
    def transition(self, branch: str, to_state: BranchState) -> bool:
        """Transition branch to new state."""
        current = self.get_state(branch)
        
        # Validate transition
        if not self.is_valid_transition(current, to_state):
            raise InvalidTransitionError(f"Cannot transition {current} -> {to_state}")
        
        # Execute transition
        self._execute_transition(branch, current, to_state)
        
        # Update metadata
        self._update_metadata(branch, to_state)
        
        return True
    
    def check_expiration(self, branch: str) -> Optional[BranchState]:
        """Check if branch has exceeded TTL for current state."""
        state = self.get_state(branch)
        ttl = self.policy.TTL_CONFIG.get(state)
        
        if not ttl:
            return None
        
        last_activity = self.get_last_activity(branch)
        if datetime.now() - last_activity > ttl:
            return self.get_expiration_action(state)
        
        return None
    
    def get_expiration_action(self, state: BranchState) -> BranchState:
        """Determine next state when TTL expires."""
        expiration_map = {
            BranchState.CREATED: BranchState.ABANDONED,
            BranchState.ACTIVE: BranchState.STALE,
            BranchState.STALE: BranchState.ARCHIVED,
            BranchState.VALIDATED: BranchState.STALE,
            BranchState.CONFLICT: BranchState.STALE,
            BranchState.MERGED: BranchState.ARCHIVED,
        }
        return expiration_map.get(state, BranchState.ARCHIVED)
    
    def _execute_transition(self, branch: str, from_state: BranchState, to_state: BranchState):
        """Execute state transition actions."""
        transition_handlers = {
            (BranchState.ACTIVE, BranchState.VALIDATED): self._on_validated,
            (BranchState.VALIDATED, BranchState.MERGED): self._on_merged,
            (BranchState.ANY, BranchState.ARCHIVED): self._on_archived,
            (BranchState.ACTIVE, BranchState.STALE): self._on_stale,
        }
        
        handler = transition_handlers.get((from_state, to_state))
        if handler:
            handler(branch)
    
    def _on_validated(self, branch: str):
        """Handle transition to validated state."""
        # Add to merge queue
        self.add_to_merge_queue(branch)
        # Notify agents
        self.notify_agents("branch_validated", branch)
    
    def _on_merged(self, branch: str):
        """Handle transition to merged state."""
        # Tag the merge
        self.tag_merge(branch)
        # Schedule archival
        self.schedule_archival(branch, days=30)
        # Update metrics
        self.update_merge_metrics(branch)
    
    def _on_archived(self, branch: str):
        """Handle transition to archived state."""
        # Rename to archive namespace
        self.archive_branch(branch)
        # Remove worktree
        self.remove_worktree(branch)
        # Cleanup remote
        self.cleanup_remote(branch)
    
    def _on_stale(self, branch: str):
        """Handle transition to stale state."""
        # Notify agent
        self.notify_agents("branch_stale", branch)
        # Add stale label
        self.add_label(branch, "stale")
        # Schedule reminder
        self.schedule_reminder(branch, days=3)
```

### 6.3 Cleanup and Maintenance

```bash
#!/bin/bash
# swarm-cleanup: Automated cleanup script

# Configuration
STALE_DAYS=7
ARCHIVE_AFTER=30
MAX_BRANCHES=150

# Function: Find stale branches
find_stale_branches() {
    git branch -r --list "origin/research/*" | while read branch; do
        branch=${branch#origin/}
        last_commit=$(git log -1 --format="%ct" "origin/$branch")
        now=$(date +%s)
        age=$(( (now - last_commit) / 86400 ))
        
        if [ $age -gt $STALE_DAYS ]; then
            echo "$branch $age"
        fi
    done
}

# Function: Archive branch
archive_branch() {
    local branch=$1
    local archive_date=$(date +%Y-%m-%d)
    local archive_name="archive/${archive_date}/${branch//\//-}"
    
    echo "Archiving: $branch -> $archive_name"
    
    # Create archive branch
    git branch -r "origin/$branch" "$archive_name"
    git push origin "$archive_name"
    
    # Delete original
    git push origin --delete "$branch"
    
    # Remove worktree if exists
    worktree_path="../worktrees/$(echo $branch | tr '/' '-')"
    if [ -d "$worktree_path" ]; then
        git worktree remove "$worktree_path" --force
    fi
}

# Function: Cleanup old archives
cleanup_old_archives() {
    local cutoff_date=$(date -d "-${ARCHIVE_AFTER} days" +%Y-%m-%d)
    
    git branch -r --list "origin/archive/*" | while read branch; do
        branch_date=$(echo $branch | grep -oP 'archive/\K[0-9]{4}-[0-9]{2}-[0-9]{2}')
        if [[ "$branch_date" < "$cutoff_date" ]]; then
            echo "Removing old archive: $branch"
            git push origin --delete "$branch"
        fi
    done
}

# Function: Prune worktrees
prune_worktrees() {
    echo "Pruning stale worktrees..."
    git worktree prune --verbose
    
    # Remove orphaned worktrees
    git worktree list --porcelain | awk '/^worktree /{print $2}' | while read path; do
        if [ ! -d "$path/.git" ] && [ ! -f "$path/.git" ]; then
            echo "Removing orphaned worktree: $path"
            rm -rf "$path"
        fi
    done
}

# Main execution
echo "=== autoconstitution Cleanup ==="
echo "Starting cleanup at $(date)"

# Find and process stale branches
echo "Checking for stale branches..."
find_stale_branches | while read branch age; do
    if [ $age -gt $ARCHIVE_AFTER ]; then
        archive_branch "$branch"
    else
        echo "Marking stale: $branch (${age} days)"
        # Add stale label via API or git notes
        git notes --ref=stale add -m "Stale: ${age} days inactive" "origin/$branch"
    fi
done

# Cleanup old archives
cleanup_old_archives

# Prune worktrees
prune_worktrees

# Report statistics
echo ""
echo "=== Cleanup Statistics ==="
echo "Active research branches: $(git branch -r --list 'origin/research/*' | wc -l)"
echo "Archived branches: $(git branch -r --list 'origin/archive/*' | wc -l)"
echo "Active worktrees: $(git worktree list | wc -l)"
echo "Cleanup completed at $(date)"
```

### 6.4 Monitoring and Metrics

```python
#!/usr/bin/env python3
"""swarm-metrics: Branch and merge metrics collection"""

import json
from datetime import datetime
from collections import defaultdict

class SwarmMetrics:
    """Collect and report autoconstitution metrics."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.metrics = defaultdict(list)
    
    def collect_branch_metrics(self) -> dict:
        """Collect comprehensive branch metrics."""
        return {
            "timestamp": datetime.now().isoformat(),
            "branches": {
                "total": self.count_branches(),
                "by_state": self.count_by_state(),
                "by_agent": self.count_by_agent(),
                "by_category": self.count_by_category(),
                "stale": self.count_stale_branches(),
                "archived": self.count_archived_branches(),
            },
            "merges": {
                "total_today": self.count_merges_today(),
                "octopus_merges": self.count_octopus_merges(),
                "conflict_rate": self.calculate_conflict_rate(),
                "avg_merge_time": self.calculate_avg_merge_time(),
            },
            "worktrees": {
                "active": self.count_active_worktrees(),
                "disk_usage": self.calculate_worktree_disk_usage(),
            },
            "agents": {
                "active": self.count_active_agents(),
                "contributions": self.get_agent_contributions(),
            }
        }
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate JSON metrics report."""
        metrics = self.collect_branch_metrics()
        report = json.dumps(metrics, indent=2)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report
    
    def count_by_state(self) -> dict:
        """Count branches by lifecycle state."""
        states = defaultdict(int)
        for branch in self.get_all_branches():
            state = self.get_branch_state(branch)
            states[state] += 1
        return dict(states)
    
    def count_by_agent(self) -> dict:
        """Count branches by agent."""
        agents = defaultdict(int)
        for branch in self.get_research_branches():
            agent = self.extract_agent_id(branch)
            agents[agent] += 1
        return dict(agents)
    
    def calculate_conflict_rate(self) -> float:
        """Calculate merge conflict rate."""
        total_merges = self.count_total_merges()
        conflicted_merges = self.count_conflicted_merges()
        
        if total_merges == 0:
            return 0.0
        
        return conflicted_merges / total_merges
```

---

## 7. Meta-Agent Arbitration Protocol

### 7.1 Arbitration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 META-AGENT ARBITRATION SYSTEM                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Arbitration Triggers                    │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ • Merge conflict detected (automatic)                   │    │
│  • Validation disagreement (threshold: 3 agents)           │    │
│  │ • Strategic direction conflict (manual trigger)         │    │
│  • Resource contention (automatic)                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Arbitration Committee                       │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │                                                          │    │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │    │
│  │   │ Senior  │  │ Senior  │  │ Senior  │  │ Rotating│   │    │
│  │   │ Agent 1 │  │ Agent 2 │  │ Agent 3 │  │ Agent   │   │    │
│  │   │ (fixed) │  │ (fixed) │  │ (fixed) │  │ (voted) │   │    │
│  │   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │    │
│  │        └─────────────┴─────────────┴─────────────┘      │    │
│  │                      │                                   │    │
│  │                      ▼                                   │    │
│  │              ┌─────────────┐                             │    │
│  │              │   Voting    │                             │    │
│  │              │   Engine    │                             │    │
│  │              └─────────────┘                             │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Resolution Actions                          │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ • Select winning approach                               │    │
│  │ • Mandate synthesis merge                               │    │
│  │ • Order A/B test                                        │    │
│  │ • Define feature flag strategy                          │    │
│  │ • Escalate to human oversight                           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Voting Protocol

```python
class ArbitrationVote:
    """Voting system for meta-agent arbitration."""
    
    VOTE_WEIGHTS = {
        "senior_agent": 1.0,
        "rotating_agent": 0.8,
        "contributing_agent": 0.5,
        "observer": 0.2,
    }
    
    THRESHOLDS = {
        "simple_majority": 0.51,
        "qualified_majority": 0.66,
        "consensus": 0.75,
        "unanimous": 1.0,
    }
    
    def __init__(self, issue: Conflict, committee: List[Agent]):
        self.issue = issue
        self.committee = committee
        self.votes = {}
    
    def cast_vote(self, agent: Agent, option: str, rationale: str):
        """Record a vote from an agent."""
        weight = self.VOTE_WEIGHTS.get(agent.role, 0.2)
        
        self.votes[agent.id] = {
            "option": option,
            "rationale": rationale,
            "weight": weight,
            "timestamp": datetime.now().isoformat(),
        }
    
    def tally_votes(self) -> VoteResult:
        """Tally votes and determine outcome."""
        # Group votes by option
        option_weights = defaultdict(float)
        option_rationales = defaultdict(list)
        
        for agent_id, vote in self.votes.items():
            option_weights[vote["option"]] += vote["weight"]
            option_rationales[vote["option"]].append(vote["rationale"])
        
        # Calculate total weight
        total_weight = sum(vote["weight"] for vote in self.votes.values())
        
        # Find winning option
        winning_option = max(option_weights, key=option_weights.get)
        winning_weight = option_weights[winning_option]
        winning_percentage = winning_weight / total_weight
        
        # Determine if threshold met
        threshold = self.THRESHOLDS.get(self.issue.required_threshold, 0.51)
        threshold_met = winning_percentage >= threshold
        
        return VoteResult(
            winning_option=winning_option,
            winning_percentage=winning_percentage,
            threshold=threshold,
            threshold_met=threshold_met,
            vote_breakdown=dict(option_weights),
            rationales=dict(option_rationales),
        )
```

---

## 8. Implementation Checklist

### Phase 1: Foundation (Week 1)
- [ ] Set up bare repository with worktree support
- [ ] Implement branch naming conventions
- [ ] Create base automation scripts (swarm-branch, swarm-merge)
- [ ] Configure Git hooks
- [ ] Set up CI/CD pipeline

### Phase 2: Automation (Week 2)
- [ ] Implement validation pipeline
- [ ] Create conflict detection system
- [ ] Build octopus merge automation
- [ ] Implement lifecycle management
- [ ] Set up monitoring and metrics

### Phase 3: Arbitration (Week 3)
- [ ] Implement meta-agent arbitration system
- [ ] Create voting protocol
- [ ] Build resolution strategy library
- [ ] Implement A/B test framework
- [ ] Set up notification system

### Phase 4: Optimization (Week 4)
- [ ] Performance tuning
- [ ] Scale testing (100+ branches)
- [ ] Documentation
- [ ] Training materials
- [ ] Monitoring dashboards

---

## 9. Quick Reference

### Common Commands

```bash
# Create new research branch
swarm-branch create <agent-id> <direction>

# Check branch status
swarm-branch list --active

# Request merge
swarm-merge request <branch> --target develop

# Check for conflicts
swarm-conflict detect <branch-a> <branch-b>

# Archive old branches
swarm-cleanup --stale-days 7

# View metrics
swarm-metrics report --format json

# Trigger arbitration
swarm-arbitrate --issue <issue-id> --committee auto
```

### Configuration File

```yaml
# .swarm/config.yaml
branch_management:
  naming_convention: "research/{agent}/{direction}/{timestamp}"
  max_branches: 150
  auto_archive_after_days: 30
  
merge_protocol:
  validation_required: true
  auto_merge_threshold: 0.75
  octopus_merge_max: 10
  
conflict_resolution:
  auto_resolve: true
  arbitration_threshold: 0.66
  strategies:
    - feature_flag
    - synthesis
    - ab_test
    
worktrees:
  max_concurrent: 100
  auto_cleanup: true
  disk_limit_gb: 10
  
meta_agent:
  committee_size: 5
  voting_threshold: 0.66
  senior_agents:
    - agent-alpha
    - agent-beta
    - agent-gamma
```

---

## Appendix A: Git Configuration

```bash
# Recommended Git configuration for autoconstitution

# Enable rerere for automatic conflict resolution
 git config --global rerere.enabled true

# Configure merge strategies
git config --global merge.ff false
git config --global pull.rebase true

# Configure octopus merge
git config --global merge.octopus.name "Swarm Octopus Merge"
git config --global merge.octopus.driver "git merge-file %O %A %B"

# Configure worktree
git config --global worktree.guessRemote true

# Configure hooks path
git config core.hooksPath .githooks
```

---

*Document Version: 1.0*
*Last Updated: 2025-01-15*
*Author: autoconstitution Architecture Team*
