"""
autoconstitution BranchManager - Usage Examples

This file demonstrates how to use the BranchManager for multi-agent
Git workflows in autoconstitution.
"""

import asyncio
from pathlib import Path

from branch_manager import (
    BranchManager,
    ConflictResolutionStrategy,
    GitConfig,
    create_branch_manager,
    TestValidationHook,
    LintValidationHook,
)


async def basic_workflow_example() -> None:
    """
    Basic workflow: Create branch, make changes, commit, and merge.
    """
    # Initialize the branch manager
    manager = await create_branch_manager(
        repo_path="/path/to/your/repo",
        user_name="Research Agent Alpha",
        user_email="alpha@autoconstitution.ai",
    )
    
    # Create a branch for this agent
    branch_name = await manager.create_agent_branch(
        agent_id="agent_alpha_001",
        base_branch="main",
    )
    print(f"Created branch: {branch_name}")
    
    # After making changes to files...
    
    # Stage and commit changes
    commit = await manager.commit_changes(
        branch=branch_name,
        message="Add new research findings",
        files=["research/data.json", "analysis/results.py"],
    )
    print(f"Created commit: {commit.short_hash}")
    
    # Create a pull request
    pr = await manager.create_pull_request(
        source_branch=branch_name,
        title="Research findings from Agent Alpha",
        description="Contains new data analysis and results",
        target_branch="main",
        labels=["research", "automated"],
    )
    print(f"Created PR: {pr.id}")
    
    # Merge the PR
    result = await manager.merge_pr(pr.id)
    print(f"Merge successful: {result.success}")


async def agent_session_example() -> None:
    """
    Using agent session context manager for automatic cleanup.
    """
    manager = await create_branch_manager(
        repo_path="/path/to/your/repo",
    )
    
    # Use context manager - branch is created on enter
    async with manager.agent_session("agent_beta") as branch:
        print(f"Working on branch: {branch}")
        
        # Make changes and commit
        await manager.commit_changes(
            branch=branch,
            message="Agent Beta improvements",
            stage_all=True,
        )
        
        # Create PR with auto-merge
        pr = await manager.create_pull_request(
            source_branch=branch,
            title="Agent Beta contribution",
            auto_merge=True,
        )
        
    # Session cleanup happens automatically


async def parallel_agents_example() -> None:
    """
    Multiple agents working in parallel with conflict resolution.
    """
    manager = await create_branch_manager(
        repo_path="/path/to/your/repo",
    )
    
    # Create branches for multiple agents
    agent_ids = ["agent_1", "agent_2", "agent_3"]
    branches = []
    
    for agent_id in agent_ids:
        branch = await manager.create_agent_branch(agent_id)
        branches.append(branch)
    
    # Agents work in parallel (simulated)
    async def agent_work(agent_id: str, branch: str) -> None:
        # Each agent makes their changes
        await manager.commit_changes(
            branch=branch,
            message=f"Contribution from {agent_id}",
            stage_all=True,
        )
    
    # Run all agents concurrently
    await asyncio.gather(*[
        agent_work(aid, branch)
        for aid, branch in zip(agent_ids, branches)
    ])
    
    # Merge with smart conflict resolution
    for branch in branches:
        try:
            result = await manager.merge_branch(
                source_branch=branch,
                target_branch="main",
                strategy=ConflictResolutionStrategy.SMART,
            )
            print(f"Merged {branch}: {result.success}")
        except Exception as e:
            print(f"Failed to merge {branch}: {e}")


async def validation_hooks_example() -> None:
    """
    Using validation hooks for quality control.
    """
    manager = await create_branch_manager(
        repo_path="/path/to/your/repo",
    )
    
    # Register validation hooks
    manager.register_validation_hook(TestValidationHook())
    manager.register_validation_hook(LintValidationHook(
        repo_path=Path("/path/to/your/repo")
    ))
    
    # Create a custom validation hook
    from branch_manager import ValidationHook
    from typing import List
    
    class DocumentationValidationHook:
        """Ensures changes include documentation updates."""
        
        async def validate(
            self,
            branch: str,
            target: str,
            changes: List[str],
        ) -> tuple[bool, str]:
            has_docs = any("docs/" in f or f.endswith(".md") for f in changes)
            has_code = any(f.endswith(".py") for f in changes)
            
            if has_code and not has_docs:
                return False, "Code changes should include documentation"
            
            return True, "Documentation check passed"
    
    manager.register_validation_hook(DocumentationValidationHook())
    
    # Now PRs will be validated before merging
    branch = await manager.create_agent_branch("agent_with_validation")
    
    # Create PR - validation will run
    pr = await manager.create_pull_request(
        source_branch=branch,
        title="Feature with validation",
        auto_merge=True,  # Will only merge if validation passes
    )


async def conflict_resolution_example() -> None:
    """
    Handling merge conflicts with different strategies.
    """
    manager = await create_branch_manager(
        repo_path="/path/to/your/repo",
    )
    
    # Create two branches that might conflict
    branch_a = await manager.create_agent_branch("agent_a")
    branch_b = await manager.create_agent_branch("agent_b")
    
    # Merge first branch
    result_a = await manager.merge_branch(
        source_branch=branch_a,
        strategy=ConflictResolutionStrategy.OURS,
    )
    
    # Try to merge second branch
    try:
        result_b = await manager.merge_branch(
            source_branch=branch_b,
            strategy=ConflictResolutionStrategy.SMART,  # Try smart resolution
        )
    except Exception as e:
        print(f"Conflict detected: {e}")
        # Handle conflict manually or abort


async def cleanup_example() -> None:
    """
    Cleaning up stale branches.
    """
    manager = await create_branch_manager(
        repo_path="/path/to/your/repo",
    )
    
    # List all agent branches
    branches = await manager.get_agent_branches()
    print(f"Active agent branches: {len(branches)}")
    
    # Find stale branches (dry run)
    stale = await manager.cleanup_stale_branches(
        max_age_hours=48,
        dry_run=True,
    )
    print(f"Would delete {len(stale)} stale branches")
    
    # Actually delete stale branches
    deleted = await manager.cleanup_stale_branches(
        max_age_hours=48,
        dry_run=False,
    )
    print(f"Deleted branches: {deleted}")


async def advanced_merge_example() -> None:
    """
    Advanced merge operations.
    """
    manager = await create_branch_manager(
        repo_path="/path/to/your/repo",
    )
    
    # Squash merge
    result = await manager.merge_branch(
        source_branch="feature-branch",
        target_branch="main",
        squash=True,
        message="Squashed feature implementation",
    )
    
    # Cherry-pick specific commits
    commit = await manager.cherry_pick(
        commit_hash="abc123",
        branch="main",
    )
    print(f"Cherry-picked: {commit.short_hash}")
    
    # Revert changes
    reverted = await manager.revert_changes(
        commit_hash="def456",
        branch="main",
    )
    if reverted:
        print(f"Reverted in: {reverted.short_hash}")


async def stash_example() -> None:
    """
    Using stash for temporary changes.
    """
    manager = await create_branch_manager(
        repo_path="/path/to/your/repo",
    )
    
    # Stash current changes
    stash_ref = await manager.stash_changes(
        message="WIP: experimental changes",
        include_untracked=True,
    )
    print(f"Stashed: {stash_ref}")
    
    # Do something else...
    
    # Pop the stash
    await manager.pop_stash(stash_ref)


async def diff_and_compare_example() -> None:
    """
    Comparing branches and viewing diffs.
    """
    manager = await create_branch_manager(
        repo_path="/path/to/your/repo",
    )
    
    # Get diff between branches
    diff = await manager.get_diff(
        branch_a="main",
        branch_b="feature-branch",
    )
    print(f"Diff size: {len(diff)} characters")
    
    # Get statistics only
    stats = await manager.get_diff(
        branch_a="main",
        branch_b="feature-branch",
        stat_only=True,
    )
    print(f"Changed files:\n{stats}")
    
    # Get branch info
    info = await manager.get_branch_info("feature-branch")
    print(f"Branch: {info.name}")
    print(f"Ahead: {info.ahead_count}, Behind: {info.behind_count}")


async def notification_example() -> None:
    """
    Setting up notifications for branch events.
    """
    from branch_manager import NotificationHook
    
    class SlackNotificationHook:
        """Example notification hook that logs to console (replace with Slack)."""
        
        async def notify(self, event: str, data: dict) -> None:
            print(f"[NOTIFICATION] Event: {event}")
            print(f"  Data: {data}")
    
    manager = await create_branch_manager(
        repo_path="/path/to/your/repo",
    )
    
    # Register notification hook
    manager.register_notification_hook(SlackNotificationHook())
    
    # Now all events will trigger notifications
    branch = await manager.create_agent_branch("notified_agent")
    # Will print: [NOTIFICATION] Event: branch_created


# Run examples
if __name__ == "__main__":
    print("autoconstitution BranchManager Examples")
    print("=" * 50)
    
    # Uncomment the example you want to run:
    # asyncio.run(basic_workflow_example())
    # asyncio.run(agent_session_example())
    # asyncio.run(parallel_agents_example())
    # asyncio.run(validation_hooks_example())
    # asyncio.run(conflict_resolution_example())
    # asyncio.run(cleanup_example())
    # asyncio.run(advanced_merge_example())
    # asyncio.run(stash_example())
    # asyncio.run(diff_and_compare_example())
    # asyncio.run(notification_example())
    
    print("\nExamples defined. Uncomment the one you want to run.")
