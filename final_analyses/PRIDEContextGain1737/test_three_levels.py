"""Lightweight checks of conservative matching primitives."""
import unittest
from pathlib import Path
from types import SimpleNamespace

from compare_three_levels import scalar, numeric, key, load_module, HAMLET


class ComparisonTests(unittest.TestCase):
    def test_scalar_units(self):
        self.assertEqual(scalar('1 mm'), scalar('1000 µm'))
        self.assertNotEqual(scalar('1'), scalar('1 mm'))
        self.assertNotEqual(scalar('1%'), scalar('1'))
        self.assertNotEqual(scalar('1'), scalar('2'))
        self.assertIsNone(scalar('1-3'))
        self.assertIsNone(scalar('three biological replicates'))

    def test_field_safety(self):
        self.assertTrue(numeric('number_of_samples'))
        self.assertTrue(numeric('collision_energy'))
        self.assertFalse(numeric('instrument'))
        self.assertEqual(key('Orbitrap™'), key('orbitrap'))

    def test_hierarchy_direction_and_limit(self):
        mod = load_module('test_hamlet_matcher', HAMLET / 'textmining/framework/benchmark/semantic_matcher.py')
        nodes = {str(i): SimpleNamespace(parents=[str(i-1)] if i else []) for i in range(6)}
        matcher = mod.HierarchicalMatcher()
        matcher.term_normalizer = SimpleNamespace(graphs={'mini': SimpleNamespace(get_node=nodes.get)})
        self.assertEqual(matcher._find_ancestor_distance('0', '4', 'mini'), 4)
        self.assertIsNone(matcher._find_ancestor_distance('0', '5', 'mini'))
        self.assertIsNone(matcher._find_ancestor_distance('4', '0', 'mini'))


if __name__ == '__main__':
    unittest.main()
