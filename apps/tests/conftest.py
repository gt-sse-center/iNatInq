"""Shared pytest configuration and fixtures.

This module provides global fixtures and configuration that are available
to all tests in the test suite.

Fixtures defined here are automatically available to all tests without explicit
import statements. Keep fixtures small, composable, and focused on setup/teardown.
Do NOT put business logic in fixtures.
"""


# The foundation package is installed in the workspace environment via uv,
# so no path manipulation is needed for imports.
