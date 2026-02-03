"""
NeuralGraphDB LangChain Integration (Sprint 65)

Provides LangChain-compatible graph store interface for use with
GraphCypherQAChain and other LangChain graph integrations.

Example:
    from neuralgraph import NeuralGraphStore
    from langchain.chains import GraphCypherQAChain
    from langchain_openai import ChatOpenAI

    # Connect to NeuralGraphDB
    graph = NeuralGraphStore(host="localhost", port=3000)

    # Use with LangChain
    chain = GraphCypherQAChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        graph=graph,
        verbose=True
    )

    result = chain.run("Who knows Alice?")
"""

import requests
from typing import Dict, Any, Optional, List


class NeuralGraphStore:
    """
    NeuralGraphDB wrapper for LangChain integration.

    Implements the graph store interface required by LangChain's
    GraphCypherQAChain and related components.

    Attributes:
        host: Database host address
        port: HTTP API port
        schema: Human-readable schema string (cached)
        structured_schema: Structured schema dict (cached)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3000,
        *,
        refresh_schema: bool = True,
    ):
        """
        Initialize connection to NeuralGraphDB.

        Args:
            host: Database host address (default: localhost)
            port: HTTP API port (default: 3000)
            refresh_schema: Whether to fetch schema on initialization
        """
        self.host = host
        self.port = port
        self._base_url = f"http://{host}:{port}/api"

        self._schema: Optional[str] = None
        self._structured_schema: Optional[Dict[str, Any]] = None

        if refresh_schema:
            self.refresh_schema()

    def refresh_schema(self) -> None:
        """
        Refresh the schema information from the database.

        Updates both the human-readable schema string and
        structured schema dictionary.
        """
        try:
            response = requests.get(f"{self._base_url}/schema")
            response.raise_for_status()
            data = response.json()

            self._schema = data.get("schema", "")
            self._structured_schema = data.get("structured_schema", {})
        except requests.exceptions.RequestException as e:
            # If schema endpoint fails, initialize with empty values
            self._schema = "Schema unavailable"
            self._structured_schema = {
                "node_props": {},
                "rel_types": [],
                "property_keys": []
            }

    @property
    def schema(self) -> str:
        """
        Return the schema as a human-readable string.

        This is used by LangChain to provide context to the LLM
        when generating NGQL queries.

        Returns:
            Human-readable schema description
        """
        if self._schema is None:
            self.refresh_schema()
        return self._schema or ""

    def get_schema(self) -> str:
        """
        Return the schema as a human-readable string.

        Alias for the schema property, matching LangChain's interface.

        Returns:
            Human-readable schema description
        """
        return self.schema

    @property
    def structured_schema(self) -> Dict[str, Any]:
        """
        Return the structured schema dictionary.

        Contains:
            - node_props: Dict mapping labels to property lists
            - rel_types: List of relationship type names
            - property_keys: List of all property keys

        Returns:
            Structured schema dictionary
        """
        if self._structured_schema is None:
            self.refresh_schema()
        return self._structured_schema or {}

    def get_structured_schema(self) -> Dict[str, Any]:
        """
        Return the structured schema dictionary.

        Alias for the structured_schema property.

        Returns:
            Structured schema dictionary
        """
        return self.structured_schema

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute an NGQL query and return the results.

        This is the primary method used by LangChain for query execution.
        Results are returned as a list of dictionaries, matching the
        format expected by GraphCypherQAChain.

        Args:
            query: NGQL query string
            params: Optional query parameters (not yet supported)

        Returns:
            List of result dictionaries

        Raises:
            Exception: If query execution fails

        Example:
            >>> graph = NeuralGraphStore()
            >>> results = graph.query("MATCH (n:Person) RETURN n.name AS name")
            >>> print(results)
            [{'name': 'Alice'}, {'name': 'Bob'}]
        """
        # TODO: Add parameter support when backend supports it
        if params:
            # For now, we can do simple string substitution for parameters
            for key, value in params.items():
                if isinstance(value, str):
                    query = query.replace(f"${key}", f'"{value}"')
                else:
                    query = query.replace(f"${key}", str(value))

        try:
            response = requests.post(
                f"{self._base_url}/query",
                json={"query": query}
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                error_msg = data.get("error", "Unknown error")
                raise Exception(f"Query failed: {error_msg}")

            result = data.get("result", {})

            # Handle query results
            if result.get("type") == "query":
                return result.get("rows", [])

            # Handle mutation results (return as single-item list)
            elif result.get("type") == "mutation":
                return [result]

            return []

        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection error: {e}")

    def close(self) -> None:
        """
        Close the database connection.

        For HTTP-based connections, this is a no-op but included
        for interface compatibility.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    # Convenience methods for common operations

    def add_node(self, label: str, properties: Dict[str, Any]) -> Optional[int]:
        """
        Add a node to the graph.

        Args:
            label: Node label
            properties: Node properties

        Returns:
            Node ID if created successfully, None otherwise
        """
        import json
        props_str = ", ".join([f'{k}: {json.dumps(v)}' for k, v in properties.items()])
        query = f"CREATE (n:{label} {{{props_str}}})"

        result = self.query(query)
        if result and result[0].get("operation") == "create_nodes":
            node_ids = result[0].get("node_ids", [])
            return node_ids[0] if node_ids else None
        return None

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        edge_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add an edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Relationship type
            properties: Optional edge properties

        Returns:
            True if edge was created successfully
        """
        import json
        props_str = ""
        if properties:
            props_str = " {" + ", ".join([f'{k}: {json.dumps(v)}' for k, v in properties.items()]) + "}"

        query = f"MATCH (a), (b) WHERE id(a) = {source_id} AND id(b) = {target_id} CREATE (a)-[:{edge_type}{props_str}]->(b)"
        result = self.query(query)
        return bool(result and result[0].get("operation") == "create_edges")
