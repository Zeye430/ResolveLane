import pytest

from resolvelane.resilience import BreakerState, CircuitBreaker


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


def breaker(clock, threshold=3, cooldown=30):
    return CircuitBreaker("test", failure_threshold=threshold, cooldown_s=cooldown, clock=clock)


def fail(b, times=1):
    for _ in range(times):
        b.failed(b.acquire())


def test_stays_closed_until_enough_consecutive_failures(clock):
    b = breaker(clock)
    fail(b, 2)
    b.succeeded(b.acquire())  # a success resets the count
    fail(b, 2)
    assert b.state is BreakerState.CLOSED
    fail(b)
    assert b.state is BreakerState.OPEN
    assert b.acquire() is None


def test_after_the_cooldown_only_one_probe_is_let_through(clock):
    b = breaker(clock)
    fail(b, 3)
    clock.now += 29
    assert b.acquire() is None
    clock.now += 1
    probe = b.acquire()
    assert probe is not None and probe.probe
    assert b.state is BreakerState.HALF_OPEN
    assert b.acquire() is None  # other calls keep skipping while the probe runs


def test_successful_probe_closes_the_breaker(clock):
    b = breaker(clock)
    fail(b, 3)
    clock.now += 30
    b.succeeded(b.acquire())
    assert b.state is BreakerState.CLOSED
    assert b.acquire() is not None


def test_failed_probe_reopens_for_another_cooldown(clock):
    b = breaker(clock)
    fail(b, 3)
    clock.now += 30
    b.failed(b.acquire())
    assert b.state is BreakerState.OPEN
    clock.now += 29
    assert b.acquire() is None
    clock.now += 1
    assert b.acquire() is not None


def test_late_success_from_before_the_breaker_opened_is_ignored(clock):
    b = breaker(clock)
    slow_call = b.acquire()
    fail(b, 3)
    b.succeeded(slow_call)
    assert b.state is BreakerState.OPEN


def test_probe_that_never_reports_back_is_replaced(clock):
    b = breaker(clock)
    fail(b, 3)
    clock.now += 30
    assert b.acquire() is not None  # this probe is lost
    clock.now += 30
    assert b.acquire() is not None


def test_single_failure_threshold_opens_immediately(clock):
    b = breaker(clock, threshold=1, cooldown=60)
    fail(b)
    assert b.state is BreakerState.OPEN


@pytest.mark.parametrize("threshold,cooldown", [(0, 30), (3, 0)])
def test_invalid_settings_are_rejected(clock, threshold, cooldown):
    with pytest.raises(ValueError):
        breaker(clock, threshold, cooldown)
