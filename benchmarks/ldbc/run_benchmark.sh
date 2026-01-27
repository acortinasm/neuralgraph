#!/bin/bash
# LDBC-SNB Benchmark Runner for NeuralGraphDB
# Usage: ./run_benchmark.sh [SF] [DATABASES]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
SF="${1:-SF1}"
DATABASES="${2:-neuralgraph}"
EXECUTIONS="${3:-3}"

echo "=============================================="
echo "LDBC-SNB Benchmark Suite"
echo "=============================================="
echo "Scale Factor: $SF"
echo "Databases: $DATABASES"
echo "Executions: $EXECUTIONS"
echo "=============================================="

# Check if data exists, generate if not
DATA_DIR="$SCRIPT_DIR/data/$SF"
if [ ! -d "$DATA_DIR" ]; then
    echo ""
    echo "Generating $SF dataset..."
    cd "$PROJECT_ROOT"
    python benchmarks/ldbc/ldbc_datagen.py --sf "$SF"
fi

# Check NeuralGraphDB server
if [[ "$DATABASES" == *"neuralgraph"* ]]; then
    echo ""
    echo "Checking NeuralGraphDB server..."
    if ! curl -s http://localhost:3000/api/query -X POST -H "Content-Type: application/json" \
         -d '{"query": "MATCH (n) RETURN count(n) LIMIT 1"}' > /dev/null 2>&1; then
        echo "ERROR: NeuralGraphDB server not running on localhost:3000"
        echo "Start with: ./target/release/neuralgraph server --port 3000"
        exit 1
    fi
    echo "  NeuralGraphDB: OK"
fi

# Run benchmark
echo ""
echo "Running benchmark..."
cd "$PROJECT_ROOT"
python benchmarks/ldbc/ldbc_benchmark.py \
    --sf "$SF" \
    --db "$DATABASES" \
    --executions "$EXECUTIONS" \
    --warmup 3 \
    --iterations 10

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "Results: benchmarks/ldbc/results/$SF/"
echo "=============================================="
