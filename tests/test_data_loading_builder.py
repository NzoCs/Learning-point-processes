#!/usr/bin/env python3
"""
Test DataLoadingSpecsBuilder and the new methods of DataConfigBuilder
"""

from new_ltpp.configs import DataConfigBuilder

def test_data_loading_specs_builder():
    """Test the new features of DataConfigBuilder."""
    print("🧪 Test of DataLoadingSpecsBuilder and enhanced DataConfigBuilder")
    
    # Test 1: Manual configuration using new convenience methods
    print("\n=== Test 1: Convenience methods ===")
    builder = DataConfigBuilder()
    
    # Base configuration
    builder.set_dataset_id("test_dataset")
    builder.set_data_format("json")
    builder.set_src_dir("data/test")
    
    # Use the new convenience methods
    builder.set_batch_size(64)
    builder.set_num_workers(4)
    builder.set_shuffle(True)
    
    builder.set_num_event_types(10)
    builder.set_max_len(256)
    builder.set_padding_side("left")
    
    config_dict = builder.get_config_dict()
    
    print("📋 Configuration générée:")
    print(f"   Dataset ID: {config_dict.get('dataset_id')}")
    print(f"   Batch size: {config_dict.get('data_loading_specs', {}).get('batch_size')}")
    print(f"   Num workers: {config_dict.get('data_loading_specs', {}).get('num_workers')}")
    print(f"   Shuffle: {config_dict.get('data_loading_specs', {}).get('shuffle')}")
    print(f"   Event types: {config_dict.get('tokenizer_specs', {}).get('num_event_types')}")
    print(f"   Max length: {config_dict.get('tokenizer_specs', {}).get('max_len')}")
    print(f"   Padding: {config_dict.get('tokenizer_specs', {}).get('padding_side')}")
    
    # Test 2: Configuration with dictionaries (traditional method)
    print("\n=== Test 2: Traditional methods ===")
    builder2 = DataConfigBuilder()
    
    builder2.set_dataset_id("traditional_test")
    builder2.set_src_dir("data/traditional")
    
    # Traditional method with dictionaries
    builder2.set_data_loading_specs({
        "batch_size": 32,
        "num_workers": 2,
        "shuffle": False,
        "pin_memory": True,
        "drop_last": True
    })
    
    builder2.set_data_specs({
        "num_event_types": 5,
        "max_len": 128,
        "padding_side": "right",
        "truncation_side": "left"
    })
    
    config_dict2 = builder2.get_config_dict()
    
    print("📋 Traditional configuration:")
    print(f"   Dataset ID: {config_dict2.get('dataset_id')}")
    print(f"   Loading specs: {config_dict2.get('data_loading_specs')}")
    print(f"   Data specs: {config_dict2.get('tokenizer_specs')}")
    
    # Test 3: Combination of both approaches
    print("\n=== Test 3: Hybrid approach ===")
    builder3 = DataConfigBuilder()
    
    builder3.set_dataset_id("hybrid_test")
    builder3.set_src_dir("data/hybrid")
    
    # Start with a dictionary
    builder3.set_data_loading_specs({
        "batch_size": 16,
        "pin_memory": True
    })
    
    # Then use convenience methods to modify/add
    builder3.set_num_workers(8)  # Add/modify num_workers
    builder3.set_shuffle(True)   # Add shuffle
    
    builder3.set_num_event_types(15)
    builder3.set_max_len(512)
    
    config_dict3 = builder3.get_config_dict()
    
    print("📋 Configuration hybride:")
    print(f"   Dataset ID: {config_dict3.get('dataset_id')}")
    print(f"   Loading specs: {config_dict3.get('data_loading_specs')}")
    print(f"   Data specs: {config_dict3.get('tokenizer_specs')}")
    
    print("\n✅ Tests completed successfully!")
    
    return True

if __name__ == "__main__":
    test_data_loading_specs_builder()