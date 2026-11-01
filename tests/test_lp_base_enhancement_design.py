from evaluate_lp_base_enhancement import (FAMILIES, SCHEMES, base_key, extend,
                                           parent_key, unique_progressive_keys)
from lp_base_enhancement_storage import all_keys, lineage


def test_factorial_and_complete_prefix_tree():
    bases = [base_key(family) for family in FAMILIES]
    factorial = [extend(base_key(base), family) for base in FAMILIES for family in FAMILIES]
    complete = unique_progressive_keys()
    keys = all_keys()
    assert len(bases) == 5 and len(factorial) == 25
    assert len(complete) == 45 and len(keys) == 60
    assert len(set(keys)) == 60
    assert all(key in keys for key in bases+factorial+complete)
    for base in FAMILIES:
        for scheme in SCHEMES:
            key = base_key(base)
            for family in scheme:
                child = extend(key, family)
                assert parent_key(child) == key
                assert lineage(child)[0] == base_key(base)
                key = child
            assert len(lineage(key)) == 4
