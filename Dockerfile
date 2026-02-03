# =============================================================================
# NeuralGraphDB Dockerfile
# =============================================================================
# Multi-stage build for optimized runtime image
#
# Build: docker build -t neuralgraph:0.1.0 .
# Run:   docker run -it neuralgraph:0.1.0
# =============================================================================

# Build stage
FROM rust:1.88-bookworm AS builder

# Install protobuf compiler for prost
RUN apt-get update && \
    apt-get install -y --no-install-recommends protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifest files first for better caching
COPY Cargo.toml Cargo.lock ./
COPY crates/neural-core/Cargo.toml crates/neural-core/
COPY crates/neural-storage/Cargo.toml crates/neural-storage/
COPY crates/neural-parser/Cargo.toml crates/neural-parser/
COPY crates/neural-executor/Cargo.toml crates/neural-executor/
COPY crates/neural-cli/Cargo.toml crates/neural-cli/

# Create dummy source files for dependency caching
RUN mkdir -p crates/neural-core/src && echo "fn main(){}" > crates/neural-core/src/lib.rs && \
    mkdir -p crates/neural-storage/src && echo "fn main(){}" > crates/neural-storage/src/lib.rs && \
    mkdir -p crates/neural-parser/src && echo "fn main(){}" > crates/neural-parser/src/lib.rs && \
    mkdir -p crates/neural-executor/src && echo "fn main(){}" > crates/neural-executor/src/lib.rs && \
    mkdir -p crates/neural-cli/src && echo "fn main(){}" > crates/neural-cli/src/main.rs

# Build dependencies only (cached layer)
RUN cargo build --release --bin neuralgraph 2>/dev/null || true

# Copy actual source code
COPY proto/ proto/
COPY crates/ crates/

# Touch source files to invalidate cache and rebuild
RUN touch crates/*/src/*.rs

# Build the actual application
RUN cargo build --release --bin neuralgraph

# =============================================================================
# Runtime stage
# =============================================================================
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 && \
    rm -rf /var/lib/apt/lists/*

# Copy the binary
COPY --from=builder /app/target/release/neuralgraph /usr/local/bin/

# Create data directory
RUN mkdir -p /data

WORKDIR /data

# Set the entrypoint
ENTRYPOINT ["neuralgraph"]

# Default to interactive mode
CMD []
