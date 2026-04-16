"""
Basic usage example for SwarmOrchestrator.

Demonstrates:
- Branch creation
- Task dependency chains
- Parallel execution
- Metrics collection
"""

import asyncio
import random
from datetime import datetime

from autoconstitution import (
    SwarmOrchestrator,
    TaskDependency,
    BranchPriority,
    task,
    retryable,
)


# Simulated research tasks
async def search_papers(query: str, count: int = 5) -> list[dict]:
    """Simulate searching for academic papers."""
    await asyncio.sleep(0.2)  # Simulate network delay
    papers = []
    for i in range(count):
        papers.append({
            "id": f"paper_{i}",
            "title": f"{query.title()} Research Paper {i}",
            "year": 2020 + i,
            "citations": random.randint(10, 1000),
        })
    return papers


async def download_paper(paper_id: str) -> bytes:
    """Simulate downloading a paper."""
    await asyncio.sleep(0.1)
    return f"Content of {paper_id}".encode()


async def extract_keywords(paper_content: bytes) -> list[str]:
    """Simulate keyword extraction."""
    await asyncio.sleep(0.15)
    return ["machine learning", "neural networks", "deep learning"]


async def analyze_sentiment(text: str) -> dict:
    """Simulate sentiment analysis."""
    await asyncio.sleep(0.1)
    return {
        "positive": random.uniform(0, 1),
        "negative": random.uniform(0, 1),
        "neutral": random.uniform(0, 1),
    }


async def generate_summary(analyses: list[dict]) -> str:
    """Simulate summary generation."""
    await asyncio.sleep(0.2)
    return f"Summary of {len(analyses)} analyses"


@retryable(max_retries=2, backoff_sec=0.5)
async def fragile_network_call() -> str:
    """Simulate a potentially failing network call."""
    if random.random() < 0.3:  # 30% failure rate
        raise ConnectionError("Network error")
    return "Success"


async def main():
    print("=" * 60)
    print("SwarmOrchestrator Basic Usage Example")
    print("=" * 60)
    
    # Create orchestrator with custom settings
    async with SwarmOrchestrator(
        max_concurrent_tasks=10,
        task_timeout_sec=30.0,
        enable_auto_scaling=True,
        enable_monitoring=True,
    ) as orchestrator:
        
        # Create research branches
        print("\n1. Creating research branches...")
        
        literature_branch = await orchestrator.create_branch(
            name="Literature Review",
            description="Comprehensive literature review on ML",
            priority=BranchPriority.HIGH,
        )
        print(f"   Created: {literature_branch.name} ({literature_branch.branch_id})")
        
        analysis_branch = await orchestrator.create_branch(
            name="Sentiment Analysis",
            description="Analyze sentiment in research papers",
            priority=BranchPriority.NORMAL,
        )
        print(f"   Created: {analysis_branch.name} ({analysis_branch.branch_id})")
        
        # Add tasks to literature branch
        print("\n2. Adding tasks to Literature Review branch...")
        
        search_task = await orchestrator.add_task(
            branch_id=literature_branch.branch_id,
            name="Search ML Papers",
            coro=search_papers,
            args=("machine learning",),
            kwargs={"count": 5},
            priority=0,
        )
        print(f"   Added: {search_task.name} ({search_task.task_id})")
        
        # Add download tasks that depend on search
        download_tasks = []
        for i in range(3):
            task_node = await orchestrator.add_task(
                branch_id=literature_branch.branch_id,
                name=f"Download Paper {i}",
                coro=download_paper,
                args=(f"paper_{i}",),
                dependencies={TaskDependency(search_task.task_id)},
                priority=1,
            )
            download_tasks.append(task_node)
            print(f"   Added: {task_node.name} ({task_node.task_id})")
        
        # Add keyword extraction tasks
        keyword_tasks = []
        for i, dl_task in enumerate(download_tasks):
            task_node = await orchestrator.add_task(
                branch_id=literature_branch.branch_id,
                name=f"Extract Keywords {i}",
                coro=extract_keywords,
                args=(b"paper content",),  # Would use dl_task.result
                dependencies={TaskDependency(dl_task.task_id)},
                priority=2,
            )
            keyword_tasks.append(task_node)
        
        # Add tasks to analysis branch
        print("\n3. Adding tasks to Sentiment Analysis branch...")
        
        sentiment_tasks = []
        for i in range(3):
            task_node = await orchestrator.add_task(
                branch_id=analysis_branch.branch_id,
                name=f"Analyze Sentiment {i}",
                coro=analyze_sentiment,
                args=(f"Sample text {i}",),
                priority=0,
            )
            sentiment_tasks.append(task_node)
            print(f"   Added: {task_node.name} ({task_node.task_id})")
        
        # Add summary task with dependencies
        summary_task = await orchestrator.add_task(
            branch_id=analysis_branch.branch_id,
            name="Generate Summary",
            coro=generate_summary,
            args=([{"dummy": "data"}],),  # Would collect all sentiment results
            dependencies={TaskDependency(t.task_id) for t in sentiment_tasks},
            priority=1,
        )
        print(f"   Added: {summary_task.name} ({summary_task.task_id})")
        
        # Execute branches
        print("\n4. Executing Literature Review branch...")
        lit_results = await orchestrator.execute_branch(literature_branch.branch_id)
        print(f"   Completed {len(lit_results)} tasks")
        
        print("\n5. Executing Sentiment Analysis branch...")
        analysis_results = await orchestrator.execute_branch(analysis_branch.branch_id)
        print(f"   Completed {len(analysis_results)} tasks")
        
        # Get metrics
        print("\n6. Collecting metrics...")
        metrics = await orchestrator.get_metrics()
        
        print(f"\n   DAG Stats:")
        print(f"     Total tasks: {metrics['dag']['total_tasks']}")
        print(f"     Status breakdown: {metrics['dag']['status_breakdown']}")
        
        print(f"\n   Agent Stats:")
        print(f"     Total agents: {metrics['agents']['total_agents']}")
        print(f"     Status breakdown: {metrics['agents']['status_breakdown']}")
        
        if 'health' in metrics:
            print(f"\n   Health Report:")
            print(f"     Overall success rate: {metrics['health']['overall_success_rate']:.2%}")
            print(f"     Active branches: {metrics['health']['active_branches']}")
            print(f"     Active agents: {metrics['health']['active_agents']}")
        
        # Print branch details
        print("\n7. Branch Details:")
        for branch_id, branch_data in metrics['branches'].items():
            print(f"\n   {branch_data['name']}:")
            print(f"     Task count: {branch_data['task_count']}")
            print(f"     Agent count: {branch_data['agent_count']}")
            if 'metrics' in branch_data:
                m = branch_data['metrics']
                print(f"     Success rate: {m.get('success_rate', 0):.2%}")
                print(f"     Avg task duration: {m.get('avg_task_duration_ms', 0):.1f}ms")
        
        # Demonstrate dynamic reallocation
        print("\n8. Analyzing and reallocating...")
        reallocation_report = await orchestrator.analyze_and_reallocate()
        print(f"   Bottlenecks found: {len(reallocation_report['bottlenecks'])}")
        print(f"   Agents spawned: {len(reallocation_report['spawns'])}")
        print(f"   Agents terminated: {len(reallocation_report['terminations'])}")
        print(f"   Reallocations: {len(reallocation_report['reallocations'])}")
        
        # Print final report
        print("\n9. Final Report:")
        report = orchestrator.generate_report()
        print(f"   Running tasks: {report['running_tasks']}")
        print(f"   Total branches: {len(report['branches'])}")
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
