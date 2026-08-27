import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from stokowski.config import (
    LinearStatesConfig,
    ServiceConfig,
    StateConfig,
    TrackerConfig,
    WorkflowDefinition,
    parse_workflow_file,
)
from stokowski.linear import (
    CANDIDATE_QUERY,
    CANDIDATE_QUERY_WITH_TEAM,
    ISSUES_BY_IDS_QUERY,
    ISSUES_BY_IDS_QUERY_WITH_TEAM,
    ISSUES_BY_STATES_QUERY,
    ISSUES_BY_STATES_QUERY_WITH_TEAM,
    LinearClient,
)
from stokowski.models import BlockerRef, Issue, RunAttempt
from stokowski.orchestrator import Orchestrator
from stokowski.pool import ConcurrencyPool


class TeamFilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_candidate_query_adds_configured_team_filter(self):
        client = LinearClient("https://example.test", "key", team="ENG")
        client._graphql = AsyncMock(
            return_value={"issues": {"nodes": [], "pageInfo": {}}}
        )
        self.addAsyncCleanup(client.close)

        await client.fetch_candidate_issues("project", ["Todo"])

        client._graphql.assert_awaited_once_with(
            CANDIDATE_QUERY_WITH_TEAM,
            {"projectSlug": "project", "states": ["Todo"], "team": "ENG"},
        )

    async def test_reconciliation_query_adds_configured_team_filter(self):
        client = LinearClient("https://example.test", "key", team="ENG")
        client._graphql = AsyncMock(return_value={"issues": {"nodes": []}})
        self.addAsyncCleanup(client.close)

        await client.fetch_issue_states_by_ids(["issue-id"])

        client._graphql.assert_awaited_once_with(
            ISSUES_BY_IDS_QUERY_WITH_TEAM,
            {"ids": ["issue-id"], "team": "ENG"},
        )

    async def test_state_reconciliation_adds_configured_team_filter(self):
        client = LinearClient("https://example.test", "key", team="ENG")
        client._graphql = AsyncMock(
            return_value={"issues": {"nodes": [], "pageInfo": {}}}
        )
        self.addAsyncCleanup(client.close)

        await client.fetch_issues_by_states("project", ["Human Review"])

        client._graphql.assert_awaited_once_with(
            ISSUES_BY_STATES_QUERY_WITH_TEAM,
            {
                "projectSlug": "project",
                "states": ["Human Review"],
                "team": "ENG",
            },
        )

    async def test_active_issue_moving_to_another_team_is_stopped_and_evicted(self):
        client = LinearClient("https://example.test", "key", team="ENG")
        client._graphql = AsyncMock(return_value={"issues": {"nodes": []}})
        self.addAsyncCleanup(client.close)

        pool = ConcurrencyPool(global_cap=1)
        orchestrator = Orchestrator("unused.yaml", project_name="project", pool=pool)
        orchestrator.workflow = WorkflowDefinition(
            config=ServiceConfig(tracker=TrackerConfig(team="ENG")),
            prompt_template="",
        )
        orchestrator._linear = client

        issue_id = "issue-id"
        task = MagicMock()
        orchestrator.running[issue_id] = RunAttempt(
            issue_id=issue_id,
            issue_identifier="ENG-1",
        )
        orchestrator._tasks[issue_id] = task
        orchestrator.claimed.add(issue_id)
        self.assertTrue(orchestrator._claim_slot(issue_id))
        orchestrator._issue_current_state[issue_id] = "implement"
        orchestrator._issue_state_runs[issue_id] = 2
        orchestrator._pending_gates[issue_id] = "review"
        orchestrator._last_session_ids[issue_id] = "session-id"

        await orchestrator._reconcile()

        task.cancel.assert_called_once_with()
        self.assertNotIn(issue_id, orchestrator.running)
        self.assertNotIn(issue_id, orchestrator._tasks)
        self.assertNotIn(issue_id, orchestrator.claimed)
        self.assertNotIn(issue_id, orchestrator._slot_held)
        self.assertEqual(pool.total_running(), 0)
        self.assertNotIn(issue_id, orchestrator._issue_current_state)
        self.assertNotIn(issue_id, orchestrator._issue_state_runs)
        self.assertNotIn(issue_id, orchestrator._pending_gates)
        self.assertNotIn(issue_id, orchestrator._last_session_ids)
        client._graphql.assert_awaited_once_with(
            ISSUES_BY_IDS_QUERY_WITH_TEAM,
            {"ids": [issue_id], "team": "ENG"},
        )

    async def test_queries_are_unchanged_when_team_is_omitted(self):
        client = LinearClient("https://example.test", "key")
        client._graphql = AsyncMock(
            side_effect=[
                {"issues": {"nodes": [], "pageInfo": {}}},
                {"issues": {"nodes": []}},
                {"issues": {"nodes": [], "pageInfo": {}}},
            ]
        )
        self.addAsyncCleanup(client.close)

        await client.fetch_candidate_issues("project", ["Todo"])
        await client.fetch_issue_states_by_ids(["issue-id"])
        await client.fetch_issues_by_states("project", ["Human Review"])

        self.assertEqual(
            client._graphql.await_args_list[0].args,
            (CANDIDATE_QUERY, {"projectSlug": "project", "states": ["Todo"]}),
        )
        self.assertEqual(
            client._graphql.await_args_list[1].args,
            (ISSUES_BY_IDS_QUERY, {"ids": ["issue-id"]}),
        )
        self.assertEqual(
            client._graphql.await_args_list[2].args,
            (
                ISSUES_BY_STATES_QUERY,
                {"projectSlug": "project", "states": ["Human Review"]},
            ),
        )

    def test_tracker_team_is_loaded_from_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow_path = Path(tmp) / "workflow.yaml"
            workflow_path.write_text(
                "tracker:\n"
                "  project_slug: project\n"
                "  api_key: key\n"
                "  team: ENG\n"
            )

            workflow = parse_workflow_file(workflow_path)

        self.assertEqual(workflow.config.tracker.team, "ENG")


class BlockerEligibilityTests(unittest.TestCase):
    def test_blockers_only_apply_to_configured_todo_state(self):
        linear_states = LinearStatesConfig(todo="Ready")
        config = ServiceConfig(
            linear_states=linear_states,
            states={"work": StateConfig(name="work", linear_state="Todo")},
        )
        orchestrator = Orchestrator("unused.yaml")
        orchestrator.workflow = WorkflowDefinition(config=config, prompt_template="")
        blocker = BlockerRef(id="blocker", identifier="ENG-1", state="In Progress")

        ready = Issue(
            id="ready",
            identifier="ENG-2",
            title="Ready issue",
            state="Ready",
            blocked_by=[blocker],
        )
        literal_todo = Issue(
            id="todo",
            identifier="ENG-3",
            title="Todo issue",
            state="Todo",
            blocked_by=[blocker],
        )

        self.assertFalse(orchestrator._is_eligible(ready))
        self.assertTrue(orchestrator._is_eligible(literal_todo))
