import requests
import json
import pandas as pd
import pyarrow.flight as flight
from typing import Dict, Any, Optional, List, Union

class NGraphClient:
    """
    Client for NeuralGraphDB.
    
    Combines HTTP API for general queries/mutations and Arrow Flight for high-performance bulk data retrieval.
    """
    
    def __init__(self, host="localhost", http_port=3000, flight_port=50051):
        self.host = host
        self.http_base_url = f"http://{host}:{http_port}/api"
        self.flight_location = f"grpc://{host}:{flight_port}"
        self.flight_client = None
        
        self._flight_initialized = False

    def _get_flight_client(self):
        if not self._flight_initialized:
            try:
                self.flight_client = flight.FlightClient(self.flight_location)
                self._flight_initialized = True
            except Exception as e:
                print(f"Warning: Could not connect to Arrow Flight at {self.flight_location}: {e}")
                self.flight_client = None
        return self.flight_client

    def query(self, ngql: str) -> Dict[str, Any]:
        """
        Execute an NGQL query or mutation against the database.
        
        Args:
            ngql: The Neural Graph Query Language string.
            
        Returns:
            A dictionary containing the results. 
        """
        url = f"{self.http_base_url}/query"
        try:
            response = requests.post(url, json={"query": ngql})
            response.raise_for_status()
            data = response.json()
            
            if not data.get("success"):
                raise Exception(f"Query failed: {data.get('error')}")
            
            return data.get("result", {})
        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection error: {e}")

    def execute(self, ngql: str) -> pd.DataFrame:
        """
        Execute a query and return the results as a Pandas DataFrame.
        """
        result = self.query(ngql)
        
        if result.get("type") == "query":
            return pd.DataFrame(result.get("rows", []))
        elif result.get("type") == "mutation":
            return pd.DataFrame([result])
        elif result.get("type") == "explain":
             print(result.get("plan", "No plan available."))
             return pd.DataFrame()
        return pd.DataFrame()

    def create_node(self, label: str, properties: Dict[str, Any] = {}) -> int:
        """
        Create a new node. Returns the ID of the created node.
        """
        props_str = ", ".join([f'{k}: {json.dumps(v)}' for k, v in properties.items()])
        query = f"CREATE (n:{label} {{{props_str}}})"
        
        res = self.query(query)
        if res.get("operation") == "create_nodes" and res.get("node_ids"):
            return res["node_ids"][0]
        return -1

    def add_edge(self, from_id: int, to_id: int, edge_type: str) -> bool:
        """
        Create an edge between two nodes.
        """
        query = f"MATCH (a), (b) WHERE id(a) = {from_id} AND id(b) = {to_id} CREATE (a)-[:{edge_type}]->(b)"
        res = self.query(query)
        return res.get("operation") == "create_edges" and res.get("count", 0) > 0
        
    def delete_node(self, node_id: int, detach: bool = True) -> bool:
        """
        Delete a node by its ID.
        """
        detach_str = "DETACH" if detach else ""
        query = f"MATCH (n) WHERE id(n) = {node_id} {detach_str} DELETE n"
        res = self.query(query)
        return res.get("operation") == "delete_nodes" and res.get("count", 0) > 0

    def set_property(self, node_id: int, property_name: str, value: Any) -> bool:
        """
        Set a property on a node.
        """
        value_str = json.dumps(value)
        query = f"MATCH (n) WHERE id(n) = {node_id} SET n.{property_name} = {value_str}"
        res = self.query(query)
        return res.get("operation") == "set_properties" and res.get("count", 0) > 0

    def get_nodes(self) -> pd.DataFrame:
        """Fetch all nodes using Arrow Flight (High Performance)."""
        client = self._get_flight_client()
        if client:
            ticket = flight.Ticket(b"nodes")
            reader = client.do_get(ticket)
            return reader.read_all().to_pandas()
        else:
            return self.execute("MATCH (n) RETURN n")

    def get_edges(self) -> pd.DataFrame:
        """Fetch all edges using Arrow Flight (High Performance)."""
        client = self._get_flight_client()
        if client:
            ticket = flight.Ticket(b"edges")
            reader = client.do_get(ticket)
            return reader.read_all().to_pandas()
        else:
             return self.execute("MATCH (a)-[r]->(b) RETURN id(a) AS source, type(r) AS type, id(b) AS target")

    def delete_all(self):
        """Clear the database."""
        self.query("MATCH (n) DETACH DELETE n")
