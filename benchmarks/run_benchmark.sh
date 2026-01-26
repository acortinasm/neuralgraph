#!/bin/bash
# Unified Benchmark Runner
# Compares NeuralGraphDB vs Neo4j vs FalkorDB
#
# Usage:
#   ./benchmarks/run_benchmark.sh              # Run all (1000 papers)
#   ./benchmarks/run_benchmark.sh 5000         # Custom paper count
#   ./benchmarks/run_benchmark.sh 1000 quick   # Skip complex queries

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NUM_PAPERS="${1:-1000}"
MODE="${2:-full}"

echo "=============================================================="
echo "  NeuralGraphDB Unified Benchmark"
echo "=============================================================="
echo "  Papers: $NUM_PAPERS"
echo "  Mode: $MODE"
echo "=============================================================="
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."

    # Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    print_status "Python 3 found"

    # Docker
    if ! command -v docker &> /dev/null; then
        print_warning "Docker not found - competitor databases unavailable"
    else
        print_status "Docker found"
    fi

    # NeuralGraphDB binary
    if [ ! -f "$PROJECT_DIR/target/release/neuralgraph" ]; then
        print_warning "NeuralGraphDB binary not found. Building..."
        cd "$PROJECT_DIR"
        cargo build --release -p neural-cli
    fi
    print_status "NeuralGraphDB binary found"

    echo
}

# Start Docker services
start_docker_services() {
    echo "Starting Docker services..."

    if ! command -v docker &> /dev/null; then
        print_warning "Skipping Docker services (Docker not available)"
        return
    fi

    cd "$PROJECT_DIR"

    # Check if compose file exists
    if [ ! -f "benchmarks/docker-compose.benchmark.yml" ]; then
        print_error "Docker compose file not found"
        return
    fi

    # Start services
    docker compose -f benchmarks/docker-compose.benchmark.yml up -d

    # Wait for services to be healthy
    echo "Waiting for services to be ready..."
    sleep 10

    # Check Neo4j
    if docker ps --filter "name=benchmark-neo4j" --filter "status=running" | grep -q neo4j; then
        print_status "Neo4j ready"
    else
        print_warning "Neo4j not ready"
    fi

    # Check FalkorDB
    if docker ps --filter "name=benchmark-falkordb" --filter "status=running" | grep -q falkordb; then
        print_status "FalkorDB ready"
    else
        print_warning "FalkorDB not ready"
    fi

    echo
}

# Start NeuralGraphDB server
start_neuralgraph_server() {
    echo "Starting NeuralGraphDB server..."

    # Check if already running
    if curl -s http://localhost:3000/api/query -d '{"query": "RETURN 1"}' > /dev/null 2>&1; then
        print_status "NeuralGraphDB already running"
        return
    fi

    cd "$PROJECT_DIR"

    # Start server in background
    ./target/release/neuralgraph server --http-port 3000 > /tmp/neuralgraph_benchmark.log 2>&1 &
    NEURALGRAPH_PID=$!
    echo $NEURALGRAPH_PID > /tmp/neuralgraph_benchmark.pid

    # Wait for server to start
    echo "Waiting for NeuralGraphDB to start..."
    for i in {1..30}; do
        if curl -s http://localhost:3000/api/query -d '{"query": "RETURN 1"}' > /dev/null 2>&1; then
            print_status "NeuralGraphDB ready (PID: $NEURALGRAPH_PID)"
            return
        fi
        sleep 1
    done

    print_error "NeuralGraphDB failed to start"
    cat /tmp/neuralgraph_benchmark.log
    exit 1
}

# Run the benchmark
run_benchmark() {
    echo "Running unified benchmark..."
    echo

    cd "$PROJECT_DIR"

    # Build command
    CMD="python3 benchmarks/unified_benchmark.py -n $NUM_PAPERS"

    if [ "$MODE" = "quick" ]; then
        CMD="$CMD --skip-complex"
    fi

    # Create output directory
    OUTPUT_DIR="benchmarks/results/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"
    CMD="$CMD -o $OUTPUT_DIR"

    echo "Command: $CMD"
    echo "Output: $OUTPUT_DIR"
    echo

    # Run benchmark
    $CMD

    echo
    print_status "Benchmark complete!"
    echo
    echo "Results saved to: $OUTPUT_DIR"
    echo

    # Show summary
    if [ -f "$OUTPUT_DIR/benchmark_report.md" ]; then
        echo "=============================================================="
        echo "  SUMMARY"
        echo "=============================================================="
        cat "$OUTPUT_DIR/benchmark_report.md"
    fi
}

# Cleanup
cleanup() {
    echo
    echo "Cleaning up..."

    # Stop NeuralGraphDB if we started it
    if [ -f /tmp/neuralgraph_benchmark.pid ]; then
        PID=$(cat /tmp/neuralgraph_benchmark.pid)
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID 2>/dev/null || true
            print_status "Stopped NeuralGraphDB (PID: $PID)"
        fi
        rm -f /tmp/neuralgraph_benchmark.pid
    fi

    # Optionally stop Docker services
    read -p "Stop Docker services? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose -f "$PROJECT_DIR/benchmarks/docker-compose.benchmark.yml" down
        print_status "Docker services stopped"
    fi
}

# Main
main() {
    check_prerequisites
    start_docker_services
    start_neuralgraph_server
    run_benchmark
    cleanup
}

# Handle Ctrl+C
trap cleanup EXIT

main
