import redis


def test_redis_is_accessible(redis_container: tuple[str, int]):
    """Smoke tests that Redis container is running and accessible."""
    redis_host, redis_port = redis_container

    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    _ = r.set("test", "test-value")
    res = r.get("test")
    assert res == "test-value"
    r.close()
