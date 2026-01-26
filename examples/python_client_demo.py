from neuralgraph import NGraphClient
import pandas as pd

def main():
    # Connect to the client
    client = NGraphClient()

    print("Clearing the database...")
    client.delete_all()

    # Create nodes
    print("Creating nodes...")
    alice_id = client.create_node("Person", {"name": "Alice", "age": 30})
    bob_id = client.create_node("Person", {"name": "Bob", "age": 25})
    charlie_id = client.create_node("Person", {"name": "Charlie", "age": 35})
    
    print(f"  - Created Alice with ID: {alice_id}")
    print(f"  - Created Bob with ID: {bob_id}")
    print(f"  - Created Charlie with ID: {charlie_id}")

    # Add edges
    print("\nCreating edges...")
    knows_ab = client.add_edge(alice_id, bob_id, "KNOWS")
    knows_bc = client.add_edge(bob_id, charlie_id, "KNOWS")
    print(f"  - Alice KNOWS Bob: {knows_ab}")
    print(f"  - Bob KNOWS Charlie: {knows_bc}")

    # Query data
    print("\nQuerying all people:")
    people_df = client.execute("MATCH (p:Person) RETURN p.name, p.age")
    print(people_df)

    # Find who Bob knows
    print("\nWho does Bob know?")
    bob_knows_df = client.execute(f"MATCH (a)-[:KNOWS]->(b) WHERE a.name = 'Bob' RETURN b.name")
    print(bob_knows_df)
    
    # Update a property
    print("\nUpdating Alice's age...")
    set_ok = client.set_property(alice_id, "age", 31)
    print(f"  - Set property successful: {set_ok}")

    # Query again to see the change
    print("\nQuerying Alice's new age:")
    alice_age_df = client.execute(f"MATCH (p:Person) WHERE id(p) = {alice_id} RETURN p.age")
    print(alice_age_df)

    # Delete a node
    print("\nDeleting Charlie...")
    delete_ok = client.delete_node(charlie_id)
    print(f"  - Delete successful: {delete_ok}")

    # Verify deletion
    print("\nQuerying all people after deletion:")
    people_after_delete_df = client.execute("MATCH (p:Person) RETURN p.name, p.age")
    print(people_after_delete_df)


if __name__ == "__main__":
    main()
