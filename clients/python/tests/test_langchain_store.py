"""
Tests for NeuralGraphStore LangChain integration.

These tests use mocking to test the LangChain integration
without requiring a running NeuralGraphDB server.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import sys

# Import directly from the module to avoid optional dependency issues
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from neuralgraph.langchain_store import NeuralGraphStore


class TestNeuralGraphStoreInit:
    """Tests for NeuralGraphStore initialization."""

    @patch("neuralgraph.langchain_store.requests.get")
    def test_init_default_params(self, mock_get):
        """Test initialization with default parameters."""
        mock_get.return_value.json.return_value = {
            "schema": "Node labels: Person\nRelationship types: KNOWS",
            "structured_schema": {
                "node_props": {"Person": ["name", "age"]},
                "rel_types": ["KNOWS"],
                "property_keys": ["name", "age"],
            },
        }
        mock_get.return_value.raise_for_status = Mock()

        store = NeuralGraphStore()

        assert store.host == "localhost"
        assert store.port == 3000
        assert store._base_url == "http://localhost:3000/api"
        mock_get.assert_called_once_with("http://localhost:3000/api/schema")

    @patch("neuralgraph.langchain_store.requests.get")
    def test_init_custom_params(self, mock_get):
        """Test initialization with custom host/port."""
        mock_get.return_value.json.return_value = {
            "schema": "",
            "structured_schema": {},
        }
        mock_get.return_value.raise_for_status = Mock()

        store = NeuralGraphStore(host="db.example.com", port=8080)

        assert store.host == "db.example.com"
        assert store.port == 8080
        assert store._base_url == "http://db.example.com:8080/api"

    def test_init_no_refresh(self):
        """Test initialization without schema refresh."""
        store = NeuralGraphStore(refresh_schema=False)

        assert store._schema is None
        assert store._structured_schema is None


class TestNeuralGraphStoreSchema:
    """Tests for schema methods."""

    @patch("neuralgraph.langchain_store.requests.get")
    def test_get_schema(self, mock_get):
        """Test get_schema returns human-readable schema."""
        schema_str = "Node labels: Person, Company\nRelationship types: WORKS_AT"
        mock_get.return_value.json.return_value = {
            "schema": schema_str,
            "structured_schema": {},
        }
        mock_get.return_value.raise_for_status = Mock()

        store = NeuralGraphStore()

        assert store.get_schema() == schema_str
        assert store.schema == schema_str

    @patch("neuralgraph.langchain_store.requests.get")
    def test_get_structured_schema(self, mock_get):
        """Test get_structured_schema returns dict."""
        structured = {
            "node_props": {"Person": ["name", "age"], "Company": ["name"]},
            "rel_types": ["WORKS_AT", "KNOWS"],
            "property_keys": ["name", "age"],
        }
        mock_get.return_value.json.return_value = {
            "schema": "",
            "structured_schema": structured,
        }
        mock_get.return_value.raise_for_status = Mock()

        store = NeuralGraphStore()

        assert store.get_structured_schema() == structured
        assert store.structured_schema == structured

    @patch("neuralgraph.langchain_store.requests.get")
    def test_refresh_schema_error(self, mock_get):
        """Test schema refresh handles connection errors gracefully."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        store = NeuralGraphStore()

        assert store.schema == "Schema unavailable"
        assert store.structured_schema == {
            "node_props": {},
            "rel_types": [],
            "property_keys": [],
        }


class TestNeuralGraphStoreQuery:
    """Tests for query method."""

    @patch("neuralgraph.langchain_store.requests.get")
    @patch("neuralgraph.langchain_store.requests.post")
    def test_query_success(self, mock_post, mock_get):
        """Test successful query execution."""
        mock_get.return_value.json.return_value = {"schema": "", "structured_schema": {}}
        mock_get.return_value.raise_for_status = Mock()

        mock_post.return_value.json.return_value = {
            "success": True,
            "result": {
                "type": "query",
                "rows": [
                    {"name": "Alice", "age": 30},
                    {"name": "Bob", "age": 25},
                ],
            },
        }
        mock_post.return_value.raise_for_status = Mock()

        store = NeuralGraphStore()
        results = store.query("MATCH (n:Person) RETURN n.name, n.age")

        assert len(results) == 2
        assert results[0] == {"name": "Alice", "age": 30}
        assert results[1] == {"name": "Bob", "age": 25}

    @patch("neuralgraph.langchain_store.requests.get")
    @patch("neuralgraph.langchain_store.requests.post")
    def test_query_mutation(self, mock_post, mock_get):
        """Test mutation query returns result dict."""
        mock_get.return_value.json.return_value = {"schema": "", "structured_schema": {}}
        mock_get.return_value.raise_for_status = Mock()

        mock_post.return_value.json.return_value = {
            "success": True,
            "result": {
                "type": "mutation",
                "operation": "create_nodes",
                "node_ids": [1],
            },
        }
        mock_post.return_value.raise_for_status = Mock()

        store = NeuralGraphStore()
        results = store.query("CREATE (n:Person {name: 'Charlie'})")

        assert len(results) == 1
        assert results[0]["operation"] == "create_nodes"

    @patch("neuralgraph.langchain_store.requests.get")
    @patch("neuralgraph.langchain_store.requests.post")
    def test_query_error(self, mock_post, mock_get):
        """Test query error raises exception."""
        mock_get.return_value.json.return_value = {"schema": "", "structured_schema": {}}
        mock_get.return_value.raise_for_status = Mock()

        mock_post.return_value.json.return_value = {
            "success": False,
            "error": "Syntax error in query",
        }
        mock_post.return_value.raise_for_status = Mock()

        store = NeuralGraphStore()

        with pytest.raises(Exception, match="Query failed: Syntax error in query"):
            store.query("INVALID QUERY")

    @patch("neuralgraph.langchain_store.requests.get")
    @patch("neuralgraph.langchain_store.requests.post")
    def test_query_with_params(self, mock_post, mock_get):
        """Test query with parameter substitution."""
        mock_get.return_value.json.return_value = {"schema": "", "structured_schema": {}}
        mock_get.return_value.raise_for_status = Mock()

        mock_post.return_value.json.return_value = {
            "success": True,
            "result": {"type": "query", "rows": [{"name": "Alice"}]},
        }
        mock_post.return_value.raise_for_status = Mock()

        store = NeuralGraphStore()
        store.query("MATCH (n:Person) WHERE n.name = $name RETURN n", {"name": "Alice"})

        # Check the query was sent with substituted parameter
        call_args = mock_post.call_args
        sent_query = call_args[1]["json"]["query"]
        assert '"Alice"' in sent_query


class TestNeuralGraphStoreConvenienceMethods:
    """Tests for convenience methods."""

    @patch("neuralgraph.langchain_store.requests.get")
    @patch("neuralgraph.langchain_store.requests.post")
    def test_add_node(self, mock_post, mock_get):
        """Test add_node creates a node."""
        mock_get.return_value.json.return_value = {"schema": "", "structured_schema": {}}
        mock_get.return_value.raise_for_status = Mock()

        mock_post.return_value.json.return_value = {
            "success": True,
            "result": {
                "type": "mutation",
                "operation": "create_nodes",
                "node_ids": [42],
            },
        }
        mock_post.return_value.raise_for_status = Mock()

        store = NeuralGraphStore()
        node_id = store.add_node("Person", {"name": "Alice", "age": 30})

        assert node_id == 42

    @patch("neuralgraph.langchain_store.requests.get")
    @patch("neuralgraph.langchain_store.requests.post")
    def test_add_edge(self, mock_post, mock_get):
        """Test add_edge creates an edge."""
        mock_get.return_value.json.return_value = {"schema": "", "structured_schema": {}}
        mock_get.return_value.raise_for_status = Mock()

        mock_post.return_value.json.return_value = {
            "success": True,
            "result": {
                "type": "mutation",
                "operation": "create_edges",
            },
        }
        mock_post.return_value.raise_for_status = Mock()

        store = NeuralGraphStore()
        success = store.add_edge(1, 2, "KNOWS", {"since": 2020})

        assert success is True


class TestNeuralGraphStoreContextManager:
    """Tests for context manager support."""

    def test_context_manager(self):
        """Test context manager protocol."""
        with NeuralGraphStore(refresh_schema=False) as store:
            assert isinstance(store, NeuralGraphStore)
        # No exception means __exit__ worked
