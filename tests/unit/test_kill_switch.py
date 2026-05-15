import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from synapse_mcp.server import lifespan_context


class TestKillSwitch(unittest.IsolatedAsyncioTestCase):
    """Test the autonomous engine kill-switch logic."""

    async def test_kill_switch_off_by_default(self):
        """Verify that autonomous processing is NOT started by default (or when 'off')."""
        mcp_mock = MagicMock()

        # Mock SynapseServer and its dependencies
        with patch("synapse_mcp.server.synapse_server") as mock_server:
            # Set up mock components
            mock_server.initialize = AsyncMock()
            mock_server.cleanup = AsyncMock()
            mock_server.insight_engine = MagicMock()
            mock_server.insight_engine.start_autonomous_processing = AsyncMock()
            mock_server.background_tasks = set()

            # Mock os.getenv to return 'off' for the kill-switch
            with patch(
                "os.getenv",
                side_effect=lambda k, d=None: (
                    "off" if k == "SYNAPSE_AUTONOMOUS_INSIGHTS" else d
                ),
            ):
                async with lifespan_context(mcp_mock):
                    # Check if start_autonomous_processing was called
                    mock_server.insight_engine.start_autonomous_processing.assert_not_called()
                    self.assertEqual(len(mock_server.background_tasks), 0)

    async def test_kill_switch_on(self):
        """Verify that autonomous processing IS started when 'on'."""
        mcp_mock = MagicMock()

        # Mock SynapseServer and its dependencies
        with patch("synapse_mcp.server.synapse_server") as mock_server:
            # Set up mock components
            mock_server.initialize = AsyncMock()
            mock_server.cleanup = AsyncMock()
            mock_server.insight_engine = MagicMock()
            # Important: make start_autonomous_processing a mock that doesn't hang
            mock_server.insight_engine.start_autonomous_processing = AsyncMock()
            mock_server.background_tasks = set()

            # Mock os.getenv to return 'on' for the kill-switch
            with patch(
                "os.getenv",
                side_effect=lambda k, d=None: (
                    "on" if k == "SYNAPSE_AUTONOMOUS_INSIGHTS" else d
                ),
            ):
                async with lifespan_context(mcp_mock):
                    # Check if start_autonomous_processing was called via asyncio.create_task
                    # We need to give it a tiny bit of time for the task to be created?
                    # Actually, lifespan_context does it immediately before yielding.
                    mock_server.insight_engine.start_autonomous_processing.assert_called_once()
                    self.assertEqual(len(mock_server.background_tasks), 1)


if __name__ == "__main__":
    unittest.main()
