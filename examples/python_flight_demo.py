import sys
import os
import subprocess
import time

# Add the local package to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "clients", "python")))

from neuralgraph import NGraphClient, to_pyg_data

def run_demo():
    print("🚀 NeuralGraphDB Python Demo")
    
    # 1. Start the Rust server in the background
    print("   Starting Rust Flight Server...")
    # Using port 50053 to avoid conflicts
    server_process = subprocess.Popen(
        ["cargo", "run", "-p", "neural-cli", "--", "serve-flight", "50053"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    time.sleep(5) # Wait for server to load data
    
    try:
        # 2. Connect client
        print("   Connecting Python client...")
        client = NGraphClient(port=50053)
        
        # 3. Fetch Data
        print("   Fetching nodes...")
        nodes = client.get_nodes()
        print(f"   ✓ Received {len(nodes)} nodes")
        print(nodes.head())
        
        print("\n   Fetching edges...")
        edges = client.get_edges()
        print(f"   ✓ Received {len(edges)} edges")
        print(edges.head())
        
        # 4. Convert to PyTorch
        print("\n   Converting to PyTorch Geometric format...")
        data = to_pyg_data(nodes, edges)
        print("   ✓ Conversion successful")
        print(f"   Graph Data Object: {data}")
        
    finally:
        print("\n   Shutting down server...")
        server_process.terminate()

if __name__ == "__main__":
    run_demo()
