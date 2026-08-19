"""Health route, plus importing app.main as a check on the whole import graph.

Real HTTP health is covered by the CI container smoke job.
"""

import asyncio

from app.api.health import health_check
from app.main import app


def test_app_registers_health_route():
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/api/health" in paths


def test_health_payload_is_healthy():
    body = asyncio.run(health_check())
    assert body["status"] == "healthy"
    assert "timestamp" in body
