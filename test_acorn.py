#!/usr/bin/env python3
"""
Validation test suite for ACORN Python wrapper.
Runs comprehensive tests to ensure correctness.
"""

import numpy as np
from ACORNIndex import ACORNIndex
import sys


def test_basic_operations():
    """Test basic insert, search, delete operations."""
    print("Testing basic operations...", end=" ")
    
    index = ACORNIndex(dimension=64)
    
    # Insert
    vec1 = np.random.randn(64).astype('float32')
    index.insert(vec1, {'id': 1}, doc_id=1)
    assert len(index) == 1
    
    # Search
    _, results = index.search(vec1, lambda m: True, k=10)
    assert 1 in results
    
    # Delete
    index.delete(1)
    assert len(index) == 0
    
    print("✓ PASSED")
    return True


def test_exact_search():
    """Test that exact vector is found."""
    print("Testing exact search...", end=" ")
    
    index = ACORNIndex(dimension=64)
    
    vec = np.random.randn(64).astype('float32')
    index.insert(vec, {'id': 1}, doc_id=1)
    
    # Add some noise vectors
    for i in range(10):
        noise = np.random.randn(64).astype('float32')
        index.insert(noise, {'id': i+2}, doc_id=i+2)
    
    # Search for exact vector
    _, results = index.search(vec, lambda m: True, k=1)
    assert results[0] == 1, f"Expected doc_id=1, got {results[0]}"
    
    print("✓ PASSED")
    return True


def test_filtering():
    """Test predicate filtering."""
    print("Testing filtering...", end=" ")
    
    index = ACORNIndex(dimension=64)
    
    # Insert vectors with different categories
    for i in range(100):
        vec = np.random.randn(64).astype('float32')
        category = 'A' if i < 50 else 'B'
        index.insert(vec, {'category': category}, doc_id=i)
    
    query = np.random.randn(64).astype('float32')
    
    # Search for category A
    _, results_a = index.search(query, lambda m: m['category'] == 'A', k=10)
    assert len(results_a) > 0
    assert all(index.metadata[i]['category'] == 'A' for i in results_a)
    
    # Search for category B
    _, results_b = index.search(query, lambda m: m['category'] == 'B', k=10)
    assert len(results_b) > 0
    assert all(index.metadata[i]['category'] == 'B' for i in results_b)
    
    print("✓ PASSED")
    return True


def test_complex_predicates():
    """Test complex multi-attribute predicates."""
    print("Testing complex predicates...", end=" ")
    
    index = ACORNIndex(dimension=64)
    
    # Insert vectors with multiple attributes
    for i in range(100):
        vec = np.random.randn(64).astype('float32')
        meta = {
            'price': i,
            'category': 'A' if i % 2 == 0 else 'B',
            'in_stock': i % 3 == 0
        }
        index.insert(vec, meta, doc_id=i)
    
    query = np.random.randn(64).astype('float32')
    
    # Complex predicate
    predicate = lambda m: (
        m['category'] == 'A' and
        m['price'] < 50 and
        m['in_stock']
    )
    
    _, results = index.search(query, predicate, k=10)
    
    # Verify all results satisfy predicate
    for doc_id in results:
        meta = index.metadata[doc_id]
        assert meta['category'] == 'A'
        assert meta['price'] < 50
        assert meta['in_stock']
    
    print("✓ PASSED")
    return True


def test_empty_results():
    """Test handling of predicates that match nothing."""
    print("Testing empty results...", end=" ")
    
    index = ACORNIndex(dimension=64)
    
    for i in range(10):
        vec = np.random.randn(64).astype('float32')
        index.insert(vec, {'value': i}, doc_id=i)
    
    query = np.random.randn(64).astype('float32')
    
    # Predicate that matches nothing
    _, results = index.search(query, lambda m: m['value'] > 1000, k=10)
    assert len(results) == 0
    
    print("✓ PASSED")
    return True


def test_dimension_mismatch():
    """Test that dimension mismatches are caught."""
    print("Testing dimension validation...", end=" ")
    
    index = ACORNIndex(dimension=64)
    
    try:
        wrong_vec = np.random.randn(128).astype('float32')
        index.insert(wrong_vec, {'id': 1}, doc_id=1)
        print("✗ FAILED (should have raised ValueError)")
        return False
    except ValueError:
        pass
    
    print("✓ PASSED")
    return True


def test_duplicate_ids():
    """Test handling of duplicate document IDs."""
    print("Testing duplicate IDs...", end=" ")
    
    index = ACORNIndex(dimension=64)
    
    vec1 = np.random.randn(64).astype('float32')
    vec2 = np.random.randn(64).astype('float32')
    
    index.insert(vec1, {'version': 1}, doc_id=1)
    index.insert(vec2, {'version': 2}, doc_id=1)  # Overwrites
    
    assert len(index) == 1
    assert index.metadata[1]['version'] == 2
    
    print("✓ PASSED")
    return True


def test_large_scale():
    """Test with larger dataset."""
    print("Testing large scale (10k vectors)...", end=" ")
    
    index = ACORNIndex(dimension=128)
    
    # Insert 10k vectors
    for i in range(10000):
        vec = np.random.randn(128).astype('float32')
        index.insert(vec, {'id': i, 'group': i % 10}, doc_id=i)
    
    assert len(index) == 10000
    
    # Search with filter
    query = np.random.randn(128).astype('float32')
    _, results = index.search(query, lambda m: m['group'] == 0, k=10)
    
    assert len(results) == 10
    assert all(index.metadata[i]['group'] == 0 for i in results)
    
    print("✓ PASSED")
    return True


def test_recall():
    """Test recall against brute force."""
    print("Testing recall...", end=" ")
    
    index = ACORNIndex(dimension=64)
    vectors = []
    
    # Insert vectors
    for i in range(100):
        vec = np.random.randn(64).astype('float32')
        vectors.append(vec)
        index.insert(vec, {'pass': i % 2 == 0}, doc_id=i)
    
    # Query
    query = np.random.randn(64).astype('float32')
    predicate = lambda m: m['pass']
    
    _, index_results = index.search(query, predicate, k=10)
    
    # Brute force
    distances = []
    for i, vec in enumerate(vectors):
        if i % 2 == 0:  # passes predicate
            dist = np.sum((query - vec) ** 2)
            distances.append((dist, i))
    
    distances.sort()
    true_results = [i for _, i in distances[:10]]
    
    # Check recall
    recall = len(set(index_results) & set(true_results)) / len(true_results)
    assert recall == 1.0, f"Recall {recall} < 1.0"
    
    print("✓ PASSED")
    return True


def main():
    print("="*60)
    print("ACORN Python Wrapper - Validation Test Suite")
    print("="*60)
    print()
    
    tests = [
        test_basic_operations,
        test_exact_search,
        test_filtering,
        test_complex_predicates,
        test_empty_results,
        test_dimension_mismatch,
        test_duplicate_ids,
        test_large_scale,
        test_recall,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    print("="*60)
    
    if failed == 0:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
