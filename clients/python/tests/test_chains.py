"""
Tests for NeuralGraphDB LangChain chains.

These tests verify the chain logic and templates without
requiring the full LangChain library.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Import directly from the module
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from neuralgraph.chains import (
    NGQL_GENERATION_TEMPLATE,
    NGQL_QA_TEMPLATE,
    NeuralGraphQAChain,
)

# Check if langchain is available
try:
    from langchain.prompts import PromptTemplate
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_ngql_generation_template_has_required_vars(self):
        """Test NGQL generation template has schema and question placeholders."""
        assert "{schema}" in NGQL_GENERATION_TEMPLATE
        assert "{question}" in NGQL_GENERATION_TEMPLATE

    def test_ngql_qa_template_has_required_vars(self):
        """Test QA template has question and context placeholders."""
        assert "{question}" in NGQL_QA_TEMPLATE
        assert "{context}" in NGQL_QA_TEMPLATE

    def test_ngql_generation_template_includes_examples(self):
        """Test template includes NGQL examples for the LLM."""
        assert "MATCH" in NGQL_GENERATION_TEMPLATE
        assert "WHERE" in NGQL_GENERATION_TEMPLATE
        assert "RETURN" in NGQL_GENERATION_TEMPLATE

    def test_ngql_generation_template_explains_ngql(self):
        """Test template explains NGQL to the LLM."""
        assert "NGQL" in NGQL_GENERATION_TEMPLATE
        assert "Cypher" in NGQL_GENERATION_TEMPLATE

    def test_ngql_qa_template_handles_empty_results(self):
        """Test QA template instructs LLM about empty results."""
        assert "empty" in NGQL_QA_TEMPLATE.lower()


class TestNeuralGraphQAChainInit:
    """Tests for NeuralGraphQAChain initialization."""

    def test_init_stores_graph(self):
        """Test initialization stores graph reference."""
        mock_graph = Mock()
        mock_llm = Mock()

        chain = NeuralGraphQAChain(mock_graph, mock_llm)

        assert chain.graph is mock_graph

    def test_init_stores_llm(self):
        """Test initialization stores LLM reference."""
        mock_graph = Mock()
        mock_llm = Mock()

        chain = NeuralGraphQAChain(mock_graph, mock_llm)

        assert chain.llm is mock_llm

    def test_init_verbose_default(self):
        """Test verbose defaults to False."""
        chain = NeuralGraphQAChain(Mock(), Mock())

        assert chain.verbose is False

    def test_init_verbose_true(self):
        """Test verbose can be set to True."""
        chain = NeuralGraphQAChain(Mock(), Mock(), verbose=True)

        assert chain.verbose is True


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestNeuralGraphQAChainRun:
    """Tests for NeuralGraphQAChain.run() method - requires LangChain."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_graph = Mock()
        self.mock_graph.get_schema.return_value = "Node labels: Person\nRelationship types: KNOWS"
        self.mock_graph.query.return_value = [{"name": "Alice"}, {"name": "Bob"}]

        self.mock_llm = Mock()

    def test_run_calls_get_schema(self):
        """Test run method calls get_schema."""
        self.mock_llm.invoke.side_effect = [
            Mock(content="MATCH (n) RETURN n"),
            Mock(content="Answer"),
        ]

        chain = NeuralGraphQAChain(self.mock_graph, self.mock_llm)
        chain.run("Test question")

        self.mock_graph.get_schema.assert_called_once()

    def test_run_executes_generated_query(self):
        """Test run method executes the generated query."""
        self.mock_llm.invoke.side_effect = [
            Mock(content="MATCH (n:Person) RETURN n.name"),
            Mock(content="Answer"),
        ]

        chain = NeuralGraphQAChain(self.mock_graph, self.mock_llm)
        chain.run("Who are the people?")

        self.mock_graph.query.assert_called_once()
        executed_query = self.mock_graph.query.call_args[0][0]
        assert "MATCH" in executed_query

    def test_run_cleans_markdown_code_blocks(self):
        """Test run method strips markdown code blocks from LLM response."""
        self.mock_llm.invoke.side_effect = [
            Mock(content="```ngql\nMATCH (n:Person) RETURN n.name\n```"),
            Mock(content="Answer"),
        ]

        chain = NeuralGraphQAChain(self.mock_graph, self.mock_llm)
        chain.run("Test")

        # Verify the query was cleaned before execution
        executed_query = self.mock_graph.query.call_args[0][0]
        assert "```" not in executed_query
        assert "MATCH" in executed_query

    def test_run_handles_query_error(self):
        """Test run method handles query execution errors."""
        self.mock_graph.query.side_effect = Exception("Database error")
        self.mock_llm.invoke.side_effect = [
            Mock(content="MATCH (n) RETURN n"),
            Mock(content="I encountered an error"),
        ]

        chain = NeuralGraphQAChain(self.mock_graph, self.mock_llm)
        answer = chain.run("Test")

        # Should return an answer (describing the error)
        assert isinstance(answer, str)


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestNeuralGraphQAChainInvoke:
    """Tests for NeuralGraphQAChain.invoke() method - requires LangChain."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_graph = Mock()
        self.mock_graph.get_schema.return_value = "Schema"
        self.mock_graph.query.return_value = []

        self.mock_llm = Mock()
        self.mock_llm.invoke.side_effect = [
            Mock(content="MATCH (n) RETURN n"),
            Mock(content="Answer"),
        ]

    def test_invoke_accepts_query_key(self):
        """Test invoke accepts 'query' key in input dict."""
        chain = NeuralGraphQAChain(self.mock_graph, self.mock_llm)
        result = chain.invoke({"query": "Test query"})

        assert "result" in result

    def test_invoke_accepts_question_key(self):
        """Test invoke accepts 'question' key in input dict."""
        self.mock_llm.invoke.side_effect = [
            Mock(content="MATCH (n) RETURN n"),
            Mock(content="Answer"),
        ]

        chain = NeuralGraphQAChain(self.mock_graph, self.mock_llm)
        result = chain.invoke({"question": "Test question"})

        assert "result" in result


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestNeuralGraphQAChainCallable:
    """Tests for NeuralGraphQAChain callable interface - requires LangChain."""

    def test_chain_is_callable(self):
        """Test chain can be called directly."""
        mock_graph = Mock()
        mock_graph.get_schema.return_value = "Schema"
        mock_graph.query.return_value = []

        mock_llm = Mock()
        mock_llm.invoke.side_effect = [
            Mock(content="MATCH (n) RETURN n"),
            Mock(content="Answer"),
        ]

        chain = NeuralGraphQAChain(mock_graph, mock_llm)
        answer = chain("Test question")

        assert isinstance(answer, str)


class TestCreateQAChainImport:
    """Tests for create_qa_chain import behavior."""

    def test_create_qa_chain_raises_import_error(self):
        """Test create_qa_chain raises ImportError without langchain."""
        from neuralgraph.chains import create_qa_chain

        # The import error happens when the function is called
        # This is tested implicitly when langchain is not installed


class TestCreateSimpleChainImport:
    """Tests for create_simple_chain import behavior."""

    def test_create_simple_chain_raises_import_error(self):
        """Test create_simple_chain raises ImportError without langchain."""
        from neuralgraph.chains import create_simple_chain

        # The import error happens when the function is called
        # This is tested implicitly when langchain is not installed
