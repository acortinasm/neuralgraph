"""
NeuralGraphDB LangChain Chains (Sprint 65)

Provides pre-configured LangChain chains for common GraphRAG patterns
with NeuralGraphDB.

Example:
    from neuralgraph import NeuralGraphStore
    from neuralgraph.chains import create_qa_chain
    from langchain_openai import ChatOpenAI

    graph = NeuralGraphStore()
    llm = ChatOpenAI(temperature=0)

    chain = create_qa_chain(llm, graph)
    result = chain.invoke({"query": "Who knows Alice?"})
"""

from typing import Any, Dict, Optional

# Default prompts for NGQL generation
NGQL_GENERATION_TEMPLATE = """Task: Generate a NeuralGraphDB NGQL query from a natural language question.

Instructions:
- Use only the node labels, relationship types, and properties provided in the schema.
- NGQL is similar to Cypher but has some differences.
- Use MATCH to find patterns, WHERE for filtering, RETURN for output.
- Use id(n) to get node IDs.
- Do not use any clauses or functions not shown in the examples.

Schema:
{schema}

Examples:
# Find all people
MATCH (p:Person) RETURN p.name

# Find relationships
MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name

# Filter by property
MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age

# Count nodes
MATCH (p:Person) RETURN COUNT(*)

# Aggregate
MATCH (p:Person) RETURN p.department, COUNT(*) GROUP BY p.department

Question: {question}

NGQL Query:"""


NGQL_QA_TEMPLATE = """You are an assistant that answers questions based on graph database query results.

Question: {question}

Query Results:
{context}

Based on the query results above, provide a natural language answer to the question.
If the results are empty or don't contain relevant information, say so clearly.

Answer:"""


def create_qa_chain(
    llm: Any,
    graph: "NeuralGraphStore",
    *,
    cypher_llm: Optional[Any] = None,
    qa_llm: Optional[Any] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    validate_cypher: bool = False,
    top_k: int = 10,
    allow_dangerous_requests: bool = False,
) -> Any:
    """
    Create a GraphCypherQAChain for NeuralGraphDB.

    This function provides a convenient way to create a LangChain
    GraphCypherQAChain configured for NeuralGraphDB.

    Args:
        llm: The language model to use for both query generation and QA
        graph: NeuralGraphStore instance
        cypher_llm: Optional separate LLM for query generation
        qa_llm: Optional separate LLM for answering
        verbose: Whether to print intermediate steps
        return_intermediate_steps: Whether to return query and results
        validate_cypher: Whether to validate generated queries (not implemented)
        top_k: Maximum results to return from queries
        allow_dangerous_requests: Must be True to allow mutations

    Returns:
        Configured GraphCypherQAChain instance

    Raises:
        ImportError: If langchain is not installed

    Example:
        >>> from neuralgraph import NeuralGraphStore
        >>> from neuralgraph.chains import create_qa_chain
        >>> from langchain_openai import ChatOpenAI
        >>>
        >>> graph = NeuralGraphStore()
        >>> llm = ChatOpenAI(temperature=0)
        >>> chain = create_qa_chain(llm, graph, verbose=True)
        >>> result = chain.invoke({"query": "How many people are in the graph?"})
    """
    try:
        from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
    except ImportError:
        raise ImportError(
            "LangChain Community is required for this feature. "
            "Install with: pip install langchain-community"
        )

    return GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_llm=cypher_llm,
        qa_llm=qa_llm,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        validate_cypher=validate_cypher,
        top_k=top_k,
        allow_dangerous_requests=allow_dangerous_requests,
    )


def create_simple_chain(
    llm: Any,
    graph: "NeuralGraphStore",
    *,
    verbose: bool = False,
) -> Any:
    """
    Create a simple NGQL generation and execution chain.

    This is a lightweight alternative to GraphCypherQAChain that
    uses custom prompts optimized for NGQL.

    Args:
        llm: The language model to use
        graph: NeuralGraphStore instance
        verbose: Whether to print intermediate steps

    Returns:
        A callable chain

    Example:
        >>> chain = create_simple_chain(llm, graph)
        >>> result = chain("Find all Person nodes")
    """
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnableLambda
    except ImportError:
        raise ImportError(
            "LangChain Core is required for this feature. "
            "Install with: pip install langchain-core"
        )

    # Create the NGQL generation prompt
    ngql_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=NGQL_GENERATION_TEMPLATE,
    )

    # Create the QA prompt
    qa_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=NGQL_QA_TEMPLATE,
    )

    def generate_and_execute(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate NGQL query and execute it."""
        question = inputs["question"]
        schema = graph.get_schema()

        # Generate NGQL query
        ngql_query = llm.invoke(
            ngql_prompt.format(schema=schema, question=question)
        ).content.strip()

        # Clean up the query (remove markdown code blocks if present)
        if ngql_query.startswith("```"):
            lines = ngql_query.split("\n")
            ngql_query = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        if verbose:
            print(f"Generated NGQL: {ngql_query}")

        # Execute the query
        try:
            results = graph.query(ngql_query)
            context = str(results) if results else "No results found."
        except Exception as e:
            context = f"Query error: {e}"

        if verbose:
            print(f"Results: {context}")

        return {
            "question": question,
            "ngql_query": ngql_query,
            "context": context,
        }

    def generate_answer(inputs: Dict[str, Any]) -> str:
        """Generate natural language answer from results."""
        answer = llm.invoke(
            qa_prompt.format(
                question=inputs["question"],
                context=inputs["context"]
            )
        ).content

        return answer

    # Build the chain
    chain = RunnableLambda(generate_and_execute) | RunnableLambda(generate_answer)

    return chain


class NeuralGraphQAChain:
    """
    A simple question-answering chain for NeuralGraphDB.

    This class provides a straightforward interface for natural language
    queries over a NeuralGraphDB graph without requiring the full
    LangChain GraphCypherQAChain setup.

    Attributes:
        graph: NeuralGraphStore instance
        llm: Language model for query generation and answering
        verbose: Whether to print intermediate steps
    """

    def __init__(
        self,
        graph: "NeuralGraphStore",
        llm: Any,
        *,
        verbose: bool = False,
    ):
        """
        Initialize the QA chain.

        Args:
            graph: NeuralGraphStore instance
            llm: Language model (must have .invoke() method)
            verbose: Whether to print intermediate steps
        """
        self.graph = graph
        self.llm = llm
        self.verbose = verbose

    def run(self, question: str) -> str:
        """
        Answer a natural language question about the graph.

        Args:
            question: Natural language question

        Returns:
            Natural language answer

        Example:
            >>> chain = NeuralGraphQAChain(graph, llm)
            >>> answer = chain.run("How many Person nodes are there?")
        """
        try:
            from langchain_core.prompts import PromptTemplate
        except ImportError:
            raise ImportError(
                "LangChain Core is required for this feature. "
                "Install with: pip install langchain-core"
            )

        ngql_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=NGQL_GENERATION_TEMPLATE,
        )

        qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=NGQL_QA_TEMPLATE,
        )

        # Get schema
        schema = self.graph.get_schema()

        # Generate NGQL
        ngql_query = self.llm.invoke(
            ngql_prompt.format(schema=schema, question=question)
        ).content.strip()

        # Clean up
        if ngql_query.startswith("```"):
            lines = ngql_query.split("\n")
            ngql_query = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        if self.verbose:
            print(f"Generated NGQL: {ngql_query}")

        # Execute
        try:
            results = self.graph.query(ngql_query)
            context = str(results) if results else "No results found."
        except Exception as e:
            context = f"Query error: {e}"

        if self.verbose:
            print(f"Results: {context}")

        # Generate answer
        answer = self.llm.invoke(
            qa_prompt.format(question=question, context=context)
        ).content

        return answer

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the chain with a dictionary input.

        Args:
            inputs: Dictionary with "query" or "question" key

        Returns:
            Dictionary with "result" key containing the answer
        """
        question = inputs.get("query") or inputs.get("question", "")
        answer = self.run(question)
        return {"result": answer}

    def __call__(self, question: str) -> str:
        """Allow calling the chain directly."""
        return self.run(question)
