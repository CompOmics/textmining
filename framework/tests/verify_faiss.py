
import os
import shutil
import pickle
import numpy as np
import logging
from normalization.index import OntologyIndex
from normalization.config import NormalizationConfig
from normalization.ontology import OntologyGraph, OntologyNode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_graph():
    """Create a small dummy ontology graph."""
    graph = OntologyGraph()
    # Create manual nodes
    n1 = OntologyNode(id="ID:1", name="epithelial cell")
    n2 = OntologyNode(id="ID:2", name="muscle cell")
    n3 = OntologyNode(id="ID:3", name="neuron")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    return graph

def test_faiss_build_and_search():
    """Test building and searching FAISS index."""
    print("\n=== Testing FAISS Build & Search ===")
    config = NormalizationConfig(
        index_backend="faiss", 
        use_quantization=False, # Too small for quantization
        use_gpu=False
    )
    
    index = OntologyIndex(config)
    graph = create_dummy_graph()
    
    # Build
    index.build(graph)
    
    # Search
    results = index.search("epithelial cell", top_k=1)
    print(f"Search results: {results}")
    
    assert len(results) == 1
    assert results[0][0] == "ID:1"
    assert results[0][2] > 0.9 # High similarity
    print("✓ FAISS build and search passed")

def test_save_load():
    """Test saving and loading FAISS index."""
    print("\n=== Testing Save & Load ===")
    config = NormalizationConfig(index_backend="faiss", use_gpu=False, use_quantization=False)
    index = OntologyIndex(config)
    graph = create_dummy_graph()
    index.build(graph)
    
    save_path = "test_index.pkl"
    index.save(save_path)
    
    # Load back
    new_index = OntologyIndex(config)
    new_index.load(save_path)
    
    # Verify
    assert new_index.index is not None
    import faiss
    assert isinstance(new_index.index, faiss.Index), "Backend should be FAISS"
    
    results = new_index.search("neuron", top_k=1)
    assert results[0][0] == "ID:3"
    print("✓ Save and Load passed")
    
    # Cleanup
    if os.path.exists(save_path): os.remove(save_path)
    if os.path.exists(save_path + ".faiss"): os.remove(save_path + ".faiss")

def test_legacy_upgrade():
    """Test upgrading from legacy sklearn index."""
    print("\n=== Testing Legacy Upgrade ===")
    save_path = "legacy_index.pkl"
    
    # 1. Create "legacy" index (sklearn)
    config_legacy = NormalizationConfig(index_backend="sklearn", use_gpu=False)
    index = OntologyIndex(config_legacy)
    graph = create_dummy_graph()
    index.build(graph)
    
    # Manually save as if it were old code (no separate .faiss file)
    # We use the current save method but it respects config.index_backend="sklearn"
    index.save(save_path)
    
    # 2. Load with new config (faiss)
    config_new = NormalizationConfig(index_backend="faiss", use_gpu=False, use_quantization=False)
    new_index = OntologyIndex(config_new)
    
    print("Loading legacy sklearn index with FAISS config...")
    new_index.load(save_path)
    
    # 3. Verify it upgraded
    import faiss
    assert isinstance(new_index.index, faiss.Index), "Should have upgraded to FAISS"
    assert os.path.exists(save_path + ".faiss"), "Should have saved upgraded index"
    
    results = new_index.search("muscle", top_k=1)
    assert results[0][0] == "ID:2"
    print("✓ Legacy upgrade passed")
    
    # Cleanup
    if os.path.exists(save_path): os.remove(save_path)
    if os.path.exists(save_path + ".faiss"): os.remove(save_path + ".faiss")

if __name__ == "__main__":
    try:
        test_faiss_build_and_search()
        test_save_load()
        test_legacy_upgrade()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
