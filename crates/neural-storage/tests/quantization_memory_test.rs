//! Memory savings test for Flash Quantization (Sprint 60)

use neural_storage::vector_index::{VectorIndex, VectorIndexConfig, QuantizationMethod};
use neural_core::NodeId;

#[test]
fn test_actual_memory_savings() {
    let dimension = 768;
    let num_vectors = 10_000;
    
    // Generate random-ish vectors
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            (0..dimension)
                .map(|j| ((i * 7 + j * 13) % 1000) as f32 / 1000.0 - 0.5)
                .collect()
        })
        .collect();
    
    // Create non-quantized index
    let config_none = VectorIndexConfig::new(dimension);
    let mut index_none = VectorIndex::with_config(config_none);
    
    // Create int8 quantized index
    let config_int8 = VectorIndexConfig::quantized(dimension);
    let mut index_int8 = VectorIndex::with_config(config_int8);
    
    // Add vectors to both
    for (i, vec) in vectors.iter().enumerate() {
        index_none.add(NodeId::new(i as u64), vec);
        index_int8.add(NodeId::new(i as u64), vec);
    }
    
    // Get memory stats
    let (vec_bytes_none, _, count_none) = index_none.memory_stats();
    let (vec_bytes_int8, _, count_int8) = index_int8.memory_stats();
    
    println!("\n=== Flash Quantization Memory Test ===");
    println!("Vectors: {} x {} dimensions", num_vectors, dimension);
    println!();
    println!("Non-quantized (f32):");
    println!("  Vector bytes: {} ({:.2} MB)", vec_bytes_none, vec_bytes_none as f64 / 1_000_000.0);
    println!("  Per vector: {} bytes", vec_bytes_none / count_none);
    println!();
    println!("Int8 quantized:");
    println!("  Vector bytes: {} ({:.2} MB)", vec_bytes_int8, vec_bytes_int8 as f64 / 1_000_000.0);
    println!("  Per vector: {} bytes", vec_bytes_int8 / count_int8);
    println!();
    println!("Savings: {:.1}x ({:.1}% reduction)", 
             vec_bytes_none as f64 / vec_bytes_int8 as f64,
             (1.0 - vec_bytes_int8 as f64 / vec_bytes_none as f64) * 100.0);
    
    // Verify the savings are as expected
    assert_eq!(vec_bytes_none, num_vectors * dimension * 4); // f32 = 4 bytes
    assert_eq!(vec_bytes_int8, num_vectors * dimension * 1); // i8 = 1 byte
    assert_eq!(vec_bytes_none / vec_bytes_int8, 4); // 4x savings
    
    // Verify search still works on quantized index
    let query = &vectors[0];
    let results = index_int8.search(query, 10);
    assert!(!results.is_empty());
    // Original vector should be in top results (quantization may affect ranking slightly)
    let found = results.iter().any(|(id, _)| *id == NodeId::new(0));
    assert!(found, "Original vector should be found in top-10 results");
}

#[test]
fn test_binary_quantization_memory() {
    let dimension = 768;
    let num_vectors = 1_000;
    
    // Generate vectors
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            (0..dimension)
                .map(|j| ((i * 7 + j * 13) % 1000) as f32 / 1000.0 - 0.5)
                .collect()
        })
        .collect();
    
    // Create binary quantized index
    let mut config = VectorIndexConfig::new(dimension);
    config.quantization = QuantizationMethod::Binary;
    let mut index = VectorIndex::with_config(config);
    
    for (i, vec) in vectors.iter().enumerate() {
        index.add(NodeId::new(i as u64), vec);
    }
    
    let (vec_bytes, _, count) = index.memory_stats();
    let expected_bytes = num_vectors * ((dimension + 7) / 8); // 1 bit per dim
    
    println!("\n=== Binary Quantization Memory Test ===");
    println!("Vectors: {} x {} dimensions", num_vectors, dimension);
    println!("Vector bytes: {} ({:.2} KB)", vec_bytes, vec_bytes as f64 / 1000.0);
    println!("Per vector: {} bytes (vs {} for f32)", vec_bytes / count, dimension * 4);
    println!("Savings: {:.1}x", (dimension * 4) as f64 / (vec_bytes / count) as f64);
    
    assert_eq!(vec_bytes, expected_bytes);
}
