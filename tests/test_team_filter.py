import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from stokowski.config import (
    LinearStatesConfig,
    ServiceConfig,
    StateConfig,
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
from stokowski.models import BlockerRef, Issue
from stokowski.orchestrator import Orchestrator


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
