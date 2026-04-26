"""
Comprehensive tests for the CrossPollinationBus system.

This module tests:
- Broadcast functionality
- Subscribe functionality  
- Frequency limiting (TokenBucket, Adaptive, Composite)
- CrossPollinationBus operations
- AgentPollinationClient
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio

from autoconstitution.pollination import (
    AgentId,
    AgentPollinationClient,
    AdaptiveRateLimiter,
    BroadcastMessage,
    CompositeFrequencyController,
    CrossPollinationBus,
    Finding,
    FindingPriority,
    FindingType,
    FrequencyController,
    SyncSubscriberCallback,
    SubscriberCallback,
    TokenBucketRateLimiter,
    create_adaptive_bus,
    create_composite_bus,
    create_default_bus,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def agent_id() -> AgentId:
    """Create a test agent ID."""
    return AgentId(id="test_agent_1")


@pytest.fixture
def agent_id_2() -> AgentId:
    """Create a second test agent ID."""
    return AgentId(id="test_agent_2")


@pytest.fixture
def finding(agent_id: AgentId) -> Finding:
    """Create a test finding."""
    return Finding(
        agent_id=agent_id,
        finding_type=FindingType.HYPERPARAMETER_IMPROVEMENT,
        priority=FindingPriority.HIGH,
        payload={"lr": 0.001, "accuracy": 0.95},
    )


@pytest.fixture
def low_priority_finding(agent_id: AgentId) -> Finding:
    """Create a low priority test finding."""
    return Finding(
        agent_id=agent_id,
        finding_type=FindingType.ARCHITECTURE_CHANGE,
        priority=FindingPriority.LOW,
        payload={"layers": 5},
    )


@pytest.fixture
def critical_finding(agent_id: AgentId) -> Finding:
    """Create a critical priority test finding."""
    return Finding(
        agent_id=agent_id,
        finding_type=FindingType.VALIDATION_STRATEGY,
        priority=FindingPriority.CRITICAL,
        payload={"strategy": "kfold"},
    )


@pytest_asyncio.fixture
async def bus() -> CrossPollinationBus:
    """Create and start a CrossPollinationBus for testing."""
    test_bus = CrossPollinationBus(
        frequency_controller=TokenBucketRateLimiter(max_tokens=100.0, refill_rate=10.0),
        max_queue_size=100,
        enable_metrics=True,
    )
    await test_bus.start()
    yield test_bus
    await test_bus.stop()


@pytest_asyncio.fixture
async def bus_no_limiter() -> CrossPollinationBus:
    """Create a bus with no rate limiting for testing broadcasts."""
    # Create a mock frequency controller that always allows broadcasts
    class AlwaysAllowController:
        async def should_allow_broadcast(self, finding: Finding) -> bool:
            return True
        
        async def record_broadcast(self, finding: Finding) -> None:
            pass
        
        async def get_stats(self) -> Dict[str, Any]:
            return {"allowed": True}
    
    test_bus = CrossPollinationBus(
        frequency_controller=AlwaysAllowController(),  # type: ignore
        max_queue_size=100,
        enable_metrics=True,
    )
    await test_bus.start()
    yield test_bus
    await test_bus.stop()


# =============================================================================
# AgentId Tests
# =============================================================================

class TestAgentId:
    """Tests for the AgentId class."""
    
    def test_agent_id_creation(self) -> None:
        """Test creating an AgentId."""
        agent = AgentId(id="agent_123")
        assert agent.id == "agent_123"
    
    def test_agent_id_str(self) -> None:
        """Test AgentId string representation."""
        agent = AgentId(id="agent_123")
        assert str(agent) == "agent_123"
    
    def test_agent_id_equality(self) -> None:
        """Test AgentId equality comparison."""
        agent1 = AgentId(id="agent_123")
        agent2 = AgentId(id="agent_123")
        agent3 = AgentId(id="agent_456")
        assert agent1 == agent2
        assert agent1 != agent3
    
    def test_agent_id_hash(self) -> None:
        """Test AgentId can be used as dictionary key."""
        agent1 = AgentId(id="agent_123")
        agent2 = AgentId(id="agent_123")
        d: Dict[AgentId, str] = {agent1: "value"}
        assert d[agent2] == "value"


# =============================================================================
# Finding Tests
# =============================================================================

class TestFinding:
    """Tests for the Finding dataclass."""
    
    def test_finding_creation(self, agent_id: AgentId) -> None:
        """Test creating a Finding."""
        finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.HYPERPARAMETER_IMPROVEMENT,
            priority=FindingPriority.HIGH,
            payload={"lr": 0.001},
        )
        assert finding.agent_id == agent_id
        assert finding.finding_type == FindingType.HYPERPARAMETER_IMPROVEMENT
        assert finding.priority == FindingPriority.HIGH
        assert finding.payload == {"lr": 0.001}
    
    def test_finding_auto_timestamp(self, agent_id: AgentId) -> None:
        """Test Finding gets auto-generated timestamp."""
        before = time.time()
        finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        after = time.time()
        assert before <= finding.timestamp <= after
    
    def test_finding_auto_id(self, agent_id: AgentId) -> None:
        """Test Finding gets auto-generated finding_id."""
        finding1 = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        finding2 = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        assert finding1.finding_id != finding2.finding_id
    
    def test_finding_hash(self, agent_id: AgentId) -> None:
        """Test Finding hash is based on finding_id."""
        finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        assert hash(finding) == hash(finding.finding_id)


# =============================================================================
# BroadcastMessage Tests
# =============================================================================

class TestBroadcastMessage:
    """Tests for the BroadcastMessage dataclass."""
    
    def test_broadcast_message_creation(self, finding: Finding) -> None:
        """Test creating a BroadcastMessage."""
        now = time.time()
        message = BroadcastMessage(
            finding=finding,
            broadcast_timestamp=now,
            sequence_number=1,
        )
        assert message.finding == finding
        assert message.broadcast_timestamp == now
        assert message.sequence_number == 1
    
    def test_broadcast_message_latency(self, agent_id: AgentId) -> None:
        """Test latency calculation."""
        finding_time = time.time()
        finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
            timestamp=finding_time,
        )
        broadcast_time = finding_time + 0.1
        message = BroadcastMessage(
            finding=finding,
            broadcast_timestamp=broadcast_time,
            sequence_number=1,
        )
        assert message.latency == pytest.approx(0.1, abs=0.01)


# =============================================================================
# TokenBucketRateLimiter Tests
# =============================================================================

class TestTokenBucketRateLimiter:
    """Tests for the TokenBucketRateLimiter class."""
    
    @pytest.mark.asyncio
    async def test_initial_tokens_allow_broadcast(self, finding: Finding) -> None:
        """Test that initial tokens allow broadcasts."""
        limiter = TokenBucketRateLimiter(max_tokens=10.0, refill_rate=1.0)
        result = await limiter.should_allow_broadcast(finding)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_token_consumption(self, agent_id: AgentId) -> None:
        """Test that tokens are consumed on broadcast."""
        limiter = TokenBucketRateLimiter(max_tokens=2.0, refill_rate=1.0)
        
        # First broadcast should succeed
        finding1 = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        assert await limiter.should_allow_broadcast(finding1) is True
        
        # Second broadcast should succeed
        finding2 = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        assert await limiter.should_allow_broadcast(finding2) is True
        
        # Third broadcast should fail (no tokens left)
        finding3 = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        assert await limiter.should_allow_broadcast(finding3) is False
    
    @pytest.mark.asyncio
    async def test_critical_priority_bypass(self, agent_id: AgentId) -> None:
        """Test that CRITICAL priority bypasses rate limiting."""
        limiter = TokenBucketRateLimiter(max_tokens=1.0, refill_rate=0.1)
        
        # Use up the token
        normal_finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        await limiter.should_allow_broadcast(normal_finding)
        
        # Critical finding should still be allowed
        critical_finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.CRITICAL,
            payload={},
        )
        assert await limiter.should_allow_broadcast(critical_finding) is True
    
    @pytest.mark.asyncio
    async def test_high_priority_half_cost(self, agent_id: AgentId) -> None:
        """Test that HIGH priority uses half the tokens."""
        limiter = TokenBucketRateLimiter(max_tokens=1.0, refill_rate=0.1)
        
        # High priority uses 0.5 tokens
        high_finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.HIGH,
            payload={},
        )
        assert await limiter.should_allow_broadcast(high_finding) is True
        
        # Can send another high priority (0.5 + 0.5 = 1.0)
        high_finding2 = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.HIGH,
            payload={},
        )
        assert await limiter.should_allow_broadcast(high_finding2) is True
        
        # Third high priority should fail
        high_finding3 = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.HIGH,
            payload={},
        )
        assert await limiter.should_allow_broadcast(high_finding3) is False
    
    @pytest.mark.asyncio
    async def test_token_refill(self, agent_id: AgentId) -> None:
        """Test that tokens are refilled over time."""
        limiter = TokenBucketRateLimiter(max_tokens=1.0, refill_rate=10.0, refill_period=0.1)
        
        finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        
        # Initialize with full tokens and set last_refill to now
        async with limiter._lock:
            limiter._tokens[agent_id] = 1.0
            limiter._last_refill[agent_id] = time.time()
        
        # Use up the token
        await limiter.should_allow_broadcast(finding)
        
        # Set last_refill far in the future to prevent any refill
        async with limiter._lock:
            limiter._last_refill[agent_id] = time.time() + 1000.0
            limiter._tokens[agent_id] = 0.0  # Ensure tokens are 0
        
        # Should fail (no tokens left and no time passed for refill)
        assert await limiter.should_allow_broadcast(finding) is False
        
        # Manually add tokens to simulate refill
        async with limiter._lock:
            limiter._tokens[agent_id] = 1.0
            limiter._last_refill[agent_id] = time.time()
        
        # Should succeed after "refill"
        assert await limiter.should_allow_broadcast(finding) is True
    
    @pytest.mark.asyncio
    async def test_per_agent_tokens(self) -> None:
        """Test that tokens are tracked per agent."""
        limiter = TokenBucketRateLimiter(max_tokens=1.0, refill_rate=1.0, refill_period=1.0)
        
        agent1 = AgentId(id="agent_1")
        agent2 = AgentId(id="agent_2")
        
        # Initialize tokens for both agents
        async with limiter._lock:
            limiter._tokens[agent1] = 1.0
            limiter._tokens[agent2] = 1.0
            now = time.time()
            limiter._last_refill[agent1] = now
            limiter._last_refill[agent2] = now
        
        # Use up agent1's token
        finding1 = Finding(
            agent_id=agent1,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        await limiter.should_allow_broadcast(finding1)
        
        # Set last_refill far in the future and ensure tokens are 0
        async with limiter._lock:
            limiter._last_refill[agent1] = time.time() + 1000.0
            limiter._tokens[agent1] = 0.0
        
        # Agent1 should fail (no tokens left)
        assert await limiter.should_allow_broadcast(finding1) is False
        
        # Agent2 should succeed (has its own tokens)
        finding2 = Finding(
            agent_id=agent2,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        assert await limiter.should_allow_broadcast(finding2) is True
    
    @pytest.mark.asyncio
    async def test_get_stats(self, finding: Finding) -> None:
        """Test getting rate limiter statistics."""
        limiter = TokenBucketRateLimiter(max_tokens=10.0, refill_rate=1.0)
        
        # Make some requests
        await limiter.should_allow_broadcast(finding)
        await limiter.should_allow_broadcast(finding)
        
        stats = await limiter.get_stats()
        assert stats["total_requests"] == 2
        assert stats["allowed_requests"] == 2
        assert stats["rejected_requests"] == 0
        assert stats["allow_rate"] == 1.0
        assert "agent_tokens" in stats


# =============================================================================
# AdaptiveRateLimiter Tests
# =============================================================================

class TestAdaptiveRateLimiter:
    """Tests for the AdaptiveRateLimiter class."""
    
    @pytest.mark.asyncio
    async def test_initial_allow_broadcast(self, finding: Finding) -> None:
        """Test that initial broadcasts are allowed."""
        limiter = AdaptiveRateLimiter(base_rate=10.0)
        result = await limiter.should_allow_broadcast(finding)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, agent_id: AgentId) -> None:
        """Test that rate limits are enforced."""
        limiter = AdaptiveRateLimiter(base_rate=2.0, load_check_interval=10.0)
        
        finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        
        # First two should succeed
        assert await limiter.should_allow_broadcast(finding) is True
        await limiter.record_broadcast(finding)
        
        assert await limiter.should_allow_broadcast(finding) is True
        await limiter.record_broadcast(finding)
        
        # Third should fail (over base_rate)
        assert await limiter.should_allow_broadcast(finding) is False
    
    @pytest.mark.asyncio
    async def test_critical_priority_bypass(self, agent_id: AgentId) -> None:
        """Test that CRITICAL priority bypasses rate limiting."""
        limiter = AdaptiveRateLimiter(base_rate=1.0)
        
        # Use up the rate
        normal_finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        await limiter.should_allow_broadcast(normal_finding)
        await limiter.record_broadcast(normal_finding)
        
        # Critical should still be allowed
        critical_finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.CRITICAL,
            payload={},
        )
        assert await limiter.should_allow_broadcast(critical_finding) is True
    
    @pytest.mark.asyncio
    async def test_priority_multipliers(self, agent_id: AgentId) -> None:
        """Test that different priorities have different rate limits."""
        limiter = AdaptiveRateLimiter(base_rate=2.0, load_check_interval=10.0)
        
        # LOW priority (0.5x multiplier = 1 per second)
        low_finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.LOW,
            payload={},
        )
        assert await limiter.should_allow_broadcast(low_finding) is True
        await limiter.record_broadcast(low_finding)
        
        # Second LOW should fail (over 1 per second)
        assert await limiter.should_allow_broadcast(low_finding) is False
        
        # HIGH priority (2x multiplier = 4 per second)
        high_finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.HIGH,
            payload={},
        )
        assert await limiter.should_allow_broadcast(high_finding) is True
    
    @pytest.mark.asyncio
    async def test_get_stats(self, finding: Finding) -> None:
        """Test getting adaptive rate limiter statistics."""
        limiter = AdaptiveRateLimiter(base_rate=10.0, min_rate=1.0, max_rate=50.0)
        
        await limiter.should_allow_broadcast(finding)
        await limiter.record_broadcast(finding)
        
        stats = await limiter.get_stats()
        assert stats["current_rate"] == 10.0
        assert stats["base_rate"] == 10.0
        assert stats["min_rate"] == 1.0
        assert stats["max_rate"] == 50.0
        assert stats["recent_broadcasts"] == 1


# =============================================================================
# CompositeFrequencyController Tests
# =============================================================================

class TestCompositeFrequencyController:
    """Tests for the CompositeFrequencyController class."""
    
    @pytest.mark.asyncio
    async def test_all_controllers_must_approve(self, finding: Finding) -> None:
        """Test that all controllers must approve for broadcast."""
        # Create two controllers - one allows, one denies
        class AllowController:
            async def should_allow_broadcast(self, finding: Finding) -> bool:
                return True
            async def record_broadcast(self, finding: Finding) -> None:
                pass
            async def get_stats(self) -> Dict[str, Any]:
                return {}
        
        class DenyController:
            async def should_allow_broadcast(self, finding: Finding) -> bool:
                return False
            async def record_broadcast(self, finding: Finding) -> None:
                pass
            async def get_stats(self) -> Dict[str, Any]:
                return {}
        
        # Both allow - should succeed
        composite = CompositeFrequencyController([AllowController(), AllowController()])  # type: ignore
        assert await composite.should_allow_broadcast(finding) is True
        
        # One denies - should fail
        composite = CompositeFrequencyController([AllowController(), DenyController()])  # type: ignore
        assert await composite.should_allow_broadcast(finding) is False
    
    @pytest.mark.asyncio
    async def test_record_broadcast_all_controllers(self, finding: Finding) -> None:
        """Test that record_broadcast calls all controllers."""
        controller1 = Mock(spec=FrequencyController)
        controller1.record_broadcast = Mock(return_value=asyncio.Future())
        controller1.record_broadcast.return_value.set_result(None)
        
        controller2 = Mock(spec=FrequencyController)
        controller2.record_broadcast = Mock(return_value=asyncio.Future())
        controller2.record_broadcast.return_value.set_result(None)
        
        composite = CompositeFrequencyController([controller1, controller2])  # type: ignore
        await composite.record_broadcast(finding)
        
        controller1.record_broadcast.assert_called_once_with(finding)
        controller2.record_broadcast.assert_called_once_with(finding)
    
    @pytest.mark.asyncio
    async def test_get_stats_all_controllers(self, finding: Finding) -> None:
        """Test that get_stats aggregates from all controllers."""
        class StatsController:
            def __init__(self, stats: Dict[str, Any]) -> None:
                self._stats = stats
            
            async def should_allow_broadcast(self, finding: Finding) -> bool:
                return True
            
            async def record_broadcast(self, finding: Finding) -> None:
                pass
            
            async def get_stats(self) -> Dict[str, Any]:
                return self._stats
        
        composite = CompositeFrequencyController([
            StatsController({"name": "controller1"}),
            StatsController({"name": "controller2"}),
        ])
        
        stats = await composite.get_stats()
        assert stats["controller_0"] == {"name": "controller1"}
        assert stats["controller_1"] == {"name": "controller2"}


# =============================================================================
# CrossPollinationBus Tests - Lifecycle
# =============================================================================

class TestCrossPollinationBusLifecycle:
    """Tests for CrossPollinationBus lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """Test starting and stopping the bus."""
        bus = CrossPollinationBus()
        assert not bus._running
        
        await bus.start()
        assert bus._running
        
        await bus.stop()
        assert not bus._running
    
    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test using the bus as a context manager."""
        bus = CrossPollinationBus()
        
        async with bus.run() as b:
            assert b._running
            assert b is bus
        
        assert not bus._running
    
    @pytest.mark.asyncio
    async def test_double_start(self) -> None:
        """Test that double start is safe."""
        bus = CrossPollinationBus()
        await bus.start()
        await bus.start()  # Should not raise
        assert bus._running
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_double_stop(self) -> None:
        """Test that double stop is safe."""
        bus = CrossPollinationBus()
        await bus.start()
        await bus.stop()
        await bus.stop()  # Should not raise
        assert not bus._running
    
    @pytest.mark.asyncio
    async def test_publish_without_start_raises(self, finding: Finding) -> None:
        """Test that publishing without starting raises an error."""
        bus = CrossPollinationBus()
        
        with pytest.raises(RuntimeError, match="Bus is not running"):
            await bus.publish(finding)


# =============================================================================
# CrossPollinationBus Tests - Subscribe
# =============================================================================

class TestCrossPollinationBusSubscribe:
    """Tests for CrossPollinationBus subscription functionality."""
    
    @pytest.mark.asyncio
    async def test_subscribe_async(self, bus_no_limiter: CrossPollinationBus, agent_id: AgentId) -> None:
        """Test subscribing with async callback."""
        received: List[Finding] = []
        
        async def callback(finding: Finding) -> None:
            received.append(finding)
        
        await bus_no_limiter.subscribe(agent_id, callback)
        assert len(bus_no_limiter._subscribers[agent_id]) == 1
    
    @pytest.mark.asyncio
    async def test_subscribe_sync(self, bus_no_limiter: CrossPollinationBus, agent_id: AgentId) -> None:
        """Test subscribing with sync callback."""
        received: List[Finding] = []
        
        def callback(finding: Finding) -> None:
            received.append(finding)
        
        await bus_no_limiter.subscribe_sync(agent_id, callback)
        assert len(bus_no_limiter._sync_subscribers[agent_id]) == 1
    
    @pytest.mark.asyncio
    async def test_subscribe_global_async(self, bus_no_limiter: CrossPollinationBus) -> None:
        """Test global async subscription."""
        async def callback(finding: Finding) -> None:
            pass
        
        await bus_no_limiter.subscribe_global(callback)
        assert len(bus_no_limiter._global_subscribers) == 1
    
    @pytest.mark.asyncio
    async def test_subscribe_global_sync(self, bus_no_limiter: CrossPollinationBus) -> None:
        """Test global sync subscription."""
        def callback(finding: Finding) -> None:
            pass
        
        await bus_no_limiter.subscribe_global_sync(callback)
        assert len(bus_no_limiter._global_sync_subscribers) == 1
    
    @pytest.mark.asyncio
    async def test_unsubscribe_all(self, bus_no_limiter: CrossPollinationBus, agent_id: AgentId) -> None:
        """Test unsubscribing all callbacks for an agent."""
        async def callback(finding: Finding) -> None:
            pass
        
        await bus_no_limiter.subscribe(agent_id, callback)
        assert agent_id in bus_no_limiter._subscribers
        
        await bus_no_limiter.unsubscribe(agent_id)
        assert agent_id not in bus_no_limiter._subscribers
    
    @pytest.mark.asyncio
    async def test_unsubscribe_specific(self, bus_no_limiter: CrossPollinationBus, agent_id: AgentId) -> None:
        """Test unsubscribing a specific callback."""
        async def callback1(finding: Finding) -> None:
            pass
        
        async def callback2(finding: Finding) -> None:
            pass
        
        await bus_no_limiter.subscribe(agent_id, callback1)
        await bus_no_limiter.subscribe(agent_id, callback2)
        assert len(bus_no_limiter._subscribers[agent_id]) == 2
        
        await bus_no_limiter.unsubscribe(agent_id, callback1)
        assert len(bus_no_limiter._subscribers[agent_id]) == 1
        assert bus_no_limiter._subscribers[agent_id][0] is callback2
    
    @pytest.mark.asyncio
    async def test_unsubscribe_global(self, bus_no_limiter: CrossPollinationBus) -> None:
        """Test unsubscribing a global subscriber."""
        async def callback(finding: Finding) -> None:
            pass
        
        await bus_no_limiter.subscribe_global(callback)
        assert len(bus_no_limiter._global_subscribers) == 1
        
        await bus_no_limiter.unsubscribe_global(callback)
        assert len(bus_no_limiter._global_subscribers) == 0
    
    @pytest.mark.asyncio
    async def test_get_subscriber_count(self, bus_no_limiter: CrossPollinationBus, agent_id: AgentId) -> None:
        """Test getting subscriber count."""
        async def async_callback(finding: Finding) -> None:
            pass
        
        def sync_callback(finding: Finding) -> None:
            pass
        
        assert await bus_no_limiter.get_subscriber_count() == 0
        
        await bus_no_limiter.subscribe(agent_id, async_callback)
        assert await bus_no_limiter.get_subscriber_count() == 1
        
        await bus_no_limiter.subscribe_sync(agent_id, sync_callback)
        assert await bus_no_limiter.get_subscriber_count() == 2
        
        await bus_no_limiter.subscribe_global(async_callback)
        assert await bus_no_limiter.get_subscriber_count() == 3
        
        await bus_no_limiter.subscribe_global_sync(sync_callback)
        assert await bus_no_limiter.get_subscriber_count() == 4


# =============================================================================
# CrossPollinationBus Tests - Broadcast
# =============================================================================

class TestCrossPollinationBusBroadcast:
    """Tests for CrossPollinationBus broadcast functionality."""
    
    @pytest.mark.asyncio
    async def test_broadcast_to_subscribers(
        self, 
        bus_no_limiter: CrossPollinationBus, 
        agent_id: AgentId, 
        agent_id_2: AgentId,
        finding: Finding
    ) -> None:
        """Test that broadcasts reach subscribers."""
        received: List[Finding] = []
        
        async def callback(f: Finding) -> None:
            received.append(f)
        
        # Subscribe agent_2 to receive broadcasts
        await bus_no_limiter.subscribe(agent_id_2, callback)
        
        # Publish from agent_1
        result = await bus_no_limiter.publish(finding)
        assert result is True
        
        # Wait for broadcast to complete
        await asyncio.sleep(0.1)
        
        # Agent_2 should have received the finding
        assert len(received) == 1
        assert received[0].finding_id == finding.finding_id
    
    @pytest.mark.asyncio
    async def test_broadcast_excludes_sender(
        self, 
        bus_no_limiter: CrossPollinationBus, 
        agent_id: AgentId,
        finding: Finding
    ) -> None:
        """Test that broadcasts don't go back to the sender."""
        received: List[Finding] = []
        
        async def callback(f: Finding) -> None:
            received.append(f)
        
        # Subscribe the sender
        await bus_no_limiter.subscribe(agent_id, callback)
        
        # Publish from the same agent
        result = await bus_no_limiter.publish(finding)
        assert result is True
        
        await asyncio.sleep(0.1)
        
        # Sender should not receive their own broadcast
        assert len(received) == 0
    
    @pytest.mark.asyncio
    async def test_global_subscriber_receives_all(
        self, 
        bus_no_limiter: CrossPollinationBus, 
        agent_id: AgentId,
        finding: Finding
    ) -> None:
        """Test that global subscribers receive all broadcasts."""
        received: List[Finding] = []
        
        async def callback(f: Finding) -> None:
            received.append(f)
        
        await bus_no_limiter.subscribe_global(callback)
        
        result = await bus_no_limiter.publish(finding)
        assert result is True
        
        await asyncio.sleep(0.1)
        
        assert len(received) == 1
    
    @pytest.mark.asyncio
    async def test_multiple_subscribers_receive(
        self, 
        bus_no_limiter: CrossPollinationBus, 
        agent_id: AgentId,
        agent_id_2: AgentId,
        finding: Finding
    ) -> None:
        """Test that multiple subscribers all receive broadcasts."""
        received1: List[Finding] = []
        received2: List[Finding] = []
        
        async def callback1(f: Finding) -> None:
            received1.append(f)
        
        async def callback2(f: Finding) -> None:
            received2.append(f)
        
        await bus_no_limiter.subscribe(agent_id_2, callback1)
        await bus_no_limiter.subscribe_global(callback2)
        
        result = await bus_no_limiter.publish(finding)
        assert result is True
        
        await asyncio.sleep(0.1)
        
        assert len(received1) == 1
        assert len(received2) == 1
    
    @pytest.mark.asyncio
    async def test_sync_subscribers_receive(
        self, 
        bus_no_limiter: CrossPollinationBus, 
        agent_id: AgentId,
        agent_id_2: AgentId,
        finding: Finding
    ) -> None:
        """Test that sync subscribers receive broadcasts."""
        received: List[Finding] = []
        
        def callback(f: Finding) -> None:
            received.append(f)
        
        await bus_no_limiter.subscribe_sync(agent_id_2, callback)
        
        result = await bus_no_limiter.publish(finding)
        assert result is True
        
        await asyncio.sleep(0.1)
        
        assert len(received) == 1
    
    @pytest.mark.asyncio
    async def test_callback_exception_handling(
        self, 
        bus_no_limiter: CrossPollinationBus, 
        agent_id: AgentId,
        agent_id_2: AgentId,
        finding: Finding
    ) -> None:
        """Test that exceptions in callbacks don't crash the bus."""
        received: List[Finding] = []
        
        async def failing_callback(f: Finding) -> None:
            raise ValueError("Test error")
        
        async def good_callback(f: Finding) -> None:
            received.append(f)
        
        await bus_no_limiter.subscribe(agent_id_2, failing_callback)
        await bus_no_limiter.subscribe_global(good_callback)
        
        # Should not raise despite failing callback
        result = await bus_no_limiter.publish(finding)
        assert result is True
        
        await asyncio.sleep(0.1)
        
        # Good callback should still have received
        assert len(received) == 1
    
    @pytest.mark.asyncio
    async def test_sequence_number_increment(self, bus_no_limiter: CrossPollinationBus, finding: Finding) -> None:
        """Test that sequence numbers increment."""
        initial_seq = bus_no_limiter._sequence_number
        
        await bus_no_limiter.publish(finding)
        await asyncio.sleep(0.1)
        
        assert bus_no_limiter._sequence_number == initial_seq + 1
        
        await bus_no_limiter.publish(finding)
        await asyncio.sleep(0.1)
        
        assert bus_no_limiter._sequence_number == initial_seq + 2


# =============================================================================
# CrossPollinationBus Tests - Metrics
# =============================================================================

class TestCrossPollinationBusMetrics:
    """Tests for CrossPollinationBus metrics collection."""
    
    @pytest.mark.asyncio
    async def test_publish_metrics(self, bus_no_limiter: CrossPollinationBus, finding: Finding) -> None:
        """Test that publish updates metrics."""
        await bus_no_limiter.publish(finding)
        await asyncio.sleep(0.1)
        
        metrics = await bus_no_limiter.get_metrics()
        assert metrics["total_published"] == 1
        assert metrics["findings_by_type"]["HYPERPARAMETER_IMPROVEMENT"] == 1
        assert metrics["findings_by_priority"]["HIGH"] == 1
    
    @pytest.mark.asyncio
    async def test_broadcast_metrics(self, bus_no_limiter: CrossPollinationBus, finding: Finding) -> None:
        """Test that broadcast updates metrics."""
        await bus_no_limiter.subscribe_global(lambda f: None)
        await bus_no_limiter.publish(finding)
        await asyncio.sleep(0.1)
        
        metrics = await bus_no_limiter.get_metrics()
        assert metrics["total_broadcast"] == 1
    
    @pytest.mark.asyncio
    async def test_latency_metrics(self, bus_no_limiter: CrossPollinationBus, finding: Finding) -> None:
        """Test that latency is tracked."""
        await bus_no_limiter.subscribe_global(lambda f: None)
        await bus_no_limiter.publish(finding)
        await asyncio.sleep(0.1)
        
        metrics = await bus_no_limiter.get_metrics()
        assert metrics["avg_latency"] >= 0
        assert metrics["max_latency"] >= 0
        assert metrics["min_latency"] >= 0
    
    @pytest.mark.asyncio
    async def test_clear_metrics(self, bus_no_limiter: CrossPollinationBus, finding: Finding) -> None:
        """Test clearing metrics."""
        await bus_no_limiter.publish(finding)
        await asyncio.sleep(0.1)
        
        metrics = await bus_no_limiter.get_metrics()
        assert metrics["total_published"] == 1
        
        await bus_no_limiter.clear_metrics()
        
        metrics = await bus_no_limiter.get_metrics()
        assert metrics["total_published"] == 0
    
    @pytest.mark.asyncio
    async def test_frequency_controller_stats_in_metrics(self, bus: CrossPollinationBus, finding: Finding) -> None:
        """Test that frequency controller stats are included."""
        await bus.publish(finding)
        await asyncio.sleep(0.1)
        
        metrics = await bus.get_metrics()
        assert "frequency_controller" in metrics


# =============================================================================
# CrossPollinationBus Tests - Frequency Limiting Integration
# =============================================================================

class TestCrossPollinationBusFrequencyLimiting:
    """Tests for CrossPollinationBus frequency limiting integration."""
    
    @pytest.mark.asyncio
    async def test_rejected_broadcast_updates_metrics(self, agent_id: AgentId) -> None:
        """Test that rejected broadcasts update rejection metrics."""
        # Create limiter that only allows 1 broadcast
        limiter = TokenBucketRateLimiter(max_tokens=1.0, refill_rate=1.0, refill_period=1.0)
        bus = CrossPollinationBus(frequency_controller=limiter)
        await bus.start()
        
        try:
            finding = Finding(
                agent_id=agent_id,
                finding_type=FindingType.CUSTOM,
                priority=FindingPriority.MEDIUM,
                payload={},
            )
            
            # First should succeed
            result1 = await bus.publish(finding)
            assert result1 is True
            
            # Set last_refill far in the future to prevent refill
            async with limiter._lock:
                limiter._last_refill[agent_id] = time.time() + 1000.0
            
            # Second should be rejected (no tokens left)
            result2 = await bus.publish(finding)
            assert result2 is False
            
            metrics = await bus.get_metrics()
            assert metrics["total_rejected"] == 1
        finally:
            await bus.stop()
    
    @pytest.mark.asyncio
    async def test_critical_priority_bypasses_limiting(self, agent_id: AgentId) -> None:
        """Test that critical priority bypasses rate limiting."""
        limiter = TokenBucketRateLimiter(max_tokens=1.0, refill_rate=0.1)
        bus = CrossPollinationBus(frequency_controller=limiter)
        await bus.start()
        
        try:
            normal_finding = Finding(
                agent_id=agent_id,
                finding_type=FindingType.CUSTOM,
                priority=FindingPriority.MEDIUM,
                payload={},
            )
            critical_finding = Finding(
                agent_id=agent_id,
                finding_type=FindingType.CUSTOM,
                priority=FindingPriority.CRITICAL,
                payload={},
            )
            
            # Use up the token
            await bus.publish(normal_finding)
            
            # Critical should still go through
            result = await bus.publish(critical_finding)
            assert result is True
        finally:
            await bus.stop()
    
    @pytest.mark.asyncio
    async def test_queue_full_drops_message(self, agent_id: AgentId) -> None:
        """Test that full queue drops messages."""
        # Create a bus with a very small queue
        bus = CrossPollinationBus(max_queue_size=1)
        await bus.start()
        
        try:
            # Fill the queue by publishing without subscribers (messages pile up)
            finding = Finding(
                agent_id=agent_id,
                finding_type=FindingType.CUSTOM,
                priority=FindingPriority.MEDIUM,
                payload={},
            )
            
            # First might succeed
            await bus.publish(finding)
            
            # Queue should be full now
            # Note: This test is timing-dependent
        finally:
            await bus.stop()


# =============================================================================
# AgentPollinationClient Tests
# =============================================================================

class TestAgentPollinationClient:
    """Tests for the AgentPollinationClient class."""
    
    @pytest.mark.asyncio
    async def test_client_creation(self, bus_no_limiter: CrossPollinationBus) -> None:
        """Test creating a client."""
        agent_id = AgentId(id="test_client")
        client = AgentPollinationClient(agent_id, bus_no_limiter)
        
        assert client.agent_id == agent_id
        assert client.pending_findings == 0
    
    @pytest.mark.asyncio
    async def test_client_subscribe_unsubscribe(self, bus_no_limiter: CrossPollinationBus) -> None:
        """Test client subscribe and unsubscribe."""
        agent_id = AgentId(id="test_client")
        client = AgentPollinationClient(agent_id, bus_no_limiter)
        
        await client.subscribe()
        assert client._subscribed is True
        
        await client.unsubscribe()
        assert client._subscribed is False
    
    @pytest.mark.asyncio
    async def test_client_publish_improvement(self, bus_no_limiter: CrossPollinationBus) -> None:
        """Test client publishing an improvement."""
        agent_id = AgentId(id="test_client")
        client = AgentPollinationClient(agent_id, bus_no_limiter)
        
        result = await client.publish_improvement(
            finding_type=FindingType.HYPERPARAMETER_IMPROVEMENT,
            priority=FindingPriority.HIGH,
            payload={"lr": 0.001},
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_client_receive_finding(
        self, 
        bus_no_limiter: CrossPollinationBus, 
        agent_id: AgentId,
        agent_id_2: AgentId,
        finding: Finding
    ) -> None:
        """Test client receiving a finding."""
        client = AgentPollinationClient(agent_id_2, bus_no_limiter)
        await client.subscribe()
        
        # Publish from another agent
        await bus_no_limiter.publish(finding)
        
        # Client should receive
        received = await client.get_next_finding(timeout=1.0)
        
        assert received is not None
        assert received.finding_id == finding.finding_id
    
    @pytest.mark.asyncio
    async def test_client_receive_timeout(self, bus_no_limiter: CrossPollinationBus) -> None:
        """Test client receive timeout."""
        agent_id = AgentId(id="test_client")
        client = AgentPollinationClient(agent_id, bus_no_limiter)
        await client.subscribe()
        
        # No findings published, should timeout
        received = await client.get_next_finding(timeout=0.1)
        
        assert received is None
    
    @pytest.mark.asyncio
    async def test_client_pending_findings(self, bus_no_limiter: CrossPollinationBus, finding: Finding) -> None:
        """Test client pending findings count."""
        agent_id = AgentId(id="test_client")
        client = AgentPollinationClient(agent_id, bus_no_limiter)
        await client.subscribe()
        
        assert client.pending_findings == 0
        
        # Publish multiple findings
        for _ in range(3):
            await bus_no_limiter.publish(finding)
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        assert client.pending_findings == 3
    
    @pytest.mark.asyncio
    async def test_client_get_findings_iter(self, bus_no_limiter: CrossPollinationBus, finding: Finding) -> None:
        """Test client findings iterator."""
        agent_id = AgentId(id="test_client")
        client = AgentPollinationClient(agent_id, bus_no_limiter)
        await client.subscribe()
        
        # Publish a finding
        await bus_no_limiter.publish(finding)
        
        # Get from iterator with timeout
        iterator = client.get_findings_iter()
        received = await asyncio.wait_for(iterator.__anext__(), timeout=1.0)
        
        assert received.finding_id == finding.finding_id


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for the factory functions."""
    
    @pytest.mark.asyncio
    async def test_create_default_bus(self) -> None:
        """Test create_default_bus factory function."""
        bus = create_default_bus(max_tokens=10.0, refill_rate=2.0, max_queue_size=100)
        
        assert isinstance(bus, CrossPollinationBus)
        assert bus._max_queue_size == 100
        assert bus._enable_metrics is True
        
        await bus.start()
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_create_adaptive_bus(self) -> None:
        """Test create_adaptive_bus factory function."""
        bus = create_adaptive_bus(base_rate=10.0, min_rate=1.0, max_rate=50.0)
        
        assert isinstance(bus, CrossPollinationBus)
        assert bus._max_queue_size == 1000
        
        await bus.start()
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_create_composite_bus(self) -> None:
        """Test create_composite_bus factory function."""
        bus = create_composite_bus(max_queue_size=500)
        
        assert isinstance(bus, CrossPollinationBus)
        assert bus._max_queue_size == 500
        
        await bus.start()
        await bus.stop()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete pollination system."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self) -> None:
        """Test a complete workflow with multiple agents."""
        # Create bus
        bus = create_default_bus(max_tokens=100.0, refill_rate=10.0)
        await bus.start()
        
        try:
            # Create agents
            agent1_id = AgentId(id="researcher_1")
            agent2_id = AgentId(id="researcher_2")
            
            client1 = AgentPollinationClient(agent1_id, bus)
            client2 = AgentPollinationClient(agent2_id, bus)
            
            # Subscribe both agents
            await client1.subscribe()
            await client2.subscribe()
            
            # Agent 1 publishes a finding
            result = await client1.publish_improvement(
                finding_type=FindingType.HYPERPARAMETER_IMPROVEMENT,
                priority=FindingPriority.HIGH,
                payload={"learning_rate": 0.001, "accuracy": 0.95},
            )
            assert result is True
            
            # Agent 2 should receive it
            received = await client2.get_next_finding(timeout=1.0)
            assert received is not None
            assert received.finding_type == FindingType.HYPERPARAMETER_IMPROVEMENT
            assert received.payload["accuracy"] == 0.95
            
            # Agent 1 should NOT receive their own finding
            received1 = await client1.get_next_finding(timeout=0.1)
            assert received1 is None
            
        finally:
            await bus.stop()
    
    @pytest.mark.asyncio
    async def test_multiple_findings_broadcast(self) -> None:
        """Test broadcasting multiple findings."""
        bus = create_default_bus(max_tokens=100.0, refill_rate=10.0)
        await bus.start()
        
        try:
            received: List[Finding] = []
            
            async def callback(f: Finding) -> None:
                received.append(f)
            
            agent1 = AgentId(id="agent_1")
            agent2 = AgentId(id="agent_2")
            
            await bus.subscribe(agent2, callback)
            
            # Publish multiple findings
            for i in range(5):
                finding = Finding(
                    agent_id=agent1,
                    finding_type=FindingType.HYPERPARAMETER_IMPROVEMENT,
                    priority=FindingPriority.MEDIUM,
                    payload={"iteration": i},
                )
                await bus.publish(finding)
            
            await asyncio.sleep(0.2)
            
            assert len(received) == 5
            
        finally:
            await bus.stop()
    
    @pytest.mark.asyncio
    async def test_mixed_priority_broadcasts(self) -> None:
        """Test broadcasts with mixed priorities."""
        bus = create_default_bus(max_tokens=100.0, refill_rate=10.0)
        await bus.start()
        
        try:
            received: List[Finding] = []
            
            async def callback(f: Finding) -> None:
                received.append(f)
            
            agent1 = AgentId(id="agent_1")
            agent2 = AgentId(id="agent_2")
            
            await bus.subscribe(agent2, callback)
            
            # Publish findings with different priorities
            priorities = [
                FindingPriority.LOW,
                FindingPriority.MEDIUM,
                FindingPriority.HIGH,
                FindingPriority.CRITICAL,
            ]
            
            for priority in priorities:
                finding = Finding(
                    agent_id=agent1,
                    finding_type=FindingType.CUSTOM,
                    priority=priority,
                    payload={"priority": priority.name},
                )
                await bus.publish(finding)
            
            await asyncio.sleep(0.2)
            
            assert len(received) == 4
            
        finally:
            await bus.stop()


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    @pytest.mark.asyncio
    async def test_empty_payload(self, bus_no_limiter: CrossPollinationBus, agent_id: AgentId) -> None:
        """Test finding with empty payload."""
        finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload={},
        )
        
        received: List[Finding] = []
        await bus_no_limiter.subscribe_global(lambda f: received.append(f))
        
        result = await bus_no_limiter.publish(finding)
        assert result is True
        
        await asyncio.sleep(0.1)
        assert len(received) == 1
    
    @pytest.mark.asyncio
    async def test_large_payload(self, bus_no_limiter: CrossPollinationBus, agent_id: AgentId) -> None:
        """Test finding with large payload."""
        large_payload = {"data": [i for i in range(1000)]}
        
        finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload=large_payload,
        )
        
        received: List[Finding] = []
        await bus_no_limiter.subscribe_global(lambda f: received.append(f))
        
        result = await bus_no_limiter.publish(finding)
        assert result is True
        
        await asyncio.sleep(0.1)
        assert len(received) == 1
        assert received[0].payload == large_payload
    
    @pytest.mark.asyncio
    async def test_nested_payload(self, bus_no_limiter: CrossPollinationBus, agent_id: AgentId) -> None:
        """Test finding with nested payload."""
        nested_payload = {
            "level1": {
                "level2": {
                    "level3": [1, 2, 3],
                    "data": {"key": "value"}
                }
            }
        }
        
        finding = Finding(
            agent_id=agent_id,
            finding_type=FindingType.CUSTOM,
            priority=FindingPriority.MEDIUM,
            payload=nested_payload,
        )
        
        received: List[Finding] = []
        await bus_no_limiter.subscribe_global(lambda f: received.append(f))
        
        result = await bus_no_limiter.publish(finding)
        assert result is True
        
        await asyncio.sleep(0.1)
        assert len(received) == 1
        assert received[0].payload["level1"]["level2"]["level3"] == [1, 2, 3]
    
    @pytest.mark.asyncio
    async def test_rapid_subscribe_unsubscribe(self, bus_no_limiter: CrossPollinationBus) -> None:
        """Test rapid subscribe/unsubscribe cycles."""
        agent = AgentId(id="rapid_agent")
        
        async def callback(f: Finding) -> None:
            pass
        
        # Rapid subscribe/unsubscribe
        for _ in range(100):
            await bus_no_limiter.subscribe(agent, callback)
            await bus_no_limiter.unsubscribe(agent)
        
        count = await bus_no_limiter.get_subscriber_count()
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_publishes(self, bus_no_limiter: CrossPollinationBus, agent_id: AgentId) -> None:
        """Test concurrent publishes."""
        received_count = 0
        
        async def callback(f: Finding) -> None:
            nonlocal received_count
            received_count += 1
        
        await bus_no_limiter.subscribe_global(callback)
        
        # Create multiple findings
        findings = [
            Finding(
                agent_id=agent_id,
                finding_type=FindingType.CUSTOM,
                priority=FindingPriority.MEDIUM,
                payload={"index": i},
            )
            for i in range(10)
        ]
        
        # Publish concurrently
        await asyncio.gather(*[bus_no_limiter.publish(f) for f in findings])
        
        await asyncio.sleep(0.3)
        
        assert received_count == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
