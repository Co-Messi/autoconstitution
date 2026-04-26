"""
Agents module for autoconstitution.

This module contains specialized agent implementations for the swarm
research framework.
"""

from autoconstitution.agents.researcher import (
    # Main agent
    ResearcherAgent,
    BaseAgent,
    create_researcher_agent,
    # Data classes
    Hypothesis,
    Experiment,
    ExperimentDesign,
    ExperimentResult,
    CodeChange,
    CrossPollinationFinding,
    Metric,
    # Enums
    HypothesisStatus,
    ExperimentStatus,
    ResultInterpretation,
    ChangePriority,
    ChangeType,
)

from autoconstitution.agents.synthesiser import (
    # Main agent
    SynthesiserAgent,
    create_synthesiser_agent,
    # Data classes
    ValidatedImprovement,
    IdentifiedPattern,
    Synergy,
    CompositeImprovement,
    SynthesisResult,
    SynthesiserContext,
    # Enums
    SynthesisStatus,
    PatternType,
    SynergyType,
    CompositePriority,
    # Protocols
    ImprovementProvider,
    # Exceptions
    SynthesisError,
    SynthesisExecutionError,
)

__all__ = [
    # Main agents
    "ResearcherAgent",
    "SynthesiserAgent",
    "BaseAgent",
    "create_researcher_agent",
    "create_synthesiser_agent",
    # Researcher data classes
    "Hypothesis",
    "Experiment",
    "ExperimentDesign",
    "ExperimentResult",
    "CodeChange",
    "CrossPollinationFinding",
    "Metric",
    # Synthesiser data classes
    "ValidatedImprovement",
    "IdentifiedPattern",
    "Synergy",
    "CompositeImprovement",
    "SynthesisResult",
    "SynthesiserContext",
    # Researcher enums
    "HypothesisStatus",
    "ExperimentStatus",
    "ResultInterpretation",
    "ChangePriority",
    "ChangeType",
    # Synthesiser enums
    "SynthesisStatus",
    "PatternType",
    "SynergyType",
    "CompositePriority",
    # Protocols
    "ImprovementProvider",
    # Exceptions
    "SynthesisError",
    "SynthesisExecutionError",
]
