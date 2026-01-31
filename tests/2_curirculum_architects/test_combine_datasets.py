"""
Comprehensive tests for combine_datasets.py
Covers all 4 strategies: domain, band, strict, discard
"""

import sys
import os
import unittest

# Add experiments path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', '2_curirculum_architects'))

from combine_datasets import MockTokenizer, LinearPacker


class TestMockTokenizer(unittest.TestCase):
    """Tests for the MockTokenizer class."""

    def setUp(self):
        self.tokenizer = MockTokenizer()

    def test_empty_string_returns_zero(self):
        self.assertEqual(self.tokenizer.count_tokens(''), 0)

    def test_short_string_returns_minimum_one(self):
        self.assertEqual(self.tokenizer.count_tokens('Hi'), 1)

    def test_token_count_approximation(self):
        # 4 chars = 1 token, 100 chars = 25 tokens
        self.assertEqual(self.tokenizer.count_tokens('A' * 100), 25)
        self.assertEqual(self.tokenizer.count_tokens('A' * 4096), 1024)

    def test_eos_token_exists(self):
        self.assertEqual(self.tokenizer.eos_token, '<EOS>')


class TestStrategyDomain(unittest.TestCase):
    """Tests for Strategy 1: Domain-based grouping."""

    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.samples = [
            {'text': 'A' * 4000, 'metadata': {'source_id': '1', 'band': 'B1', 'domain': 'wiki'}},
            {'text': 'B' * 4000, 'metadata': {'source_id': '2', 'band': 'B3', 'domain': 'wiki'}},
            {'text': 'C' * 4000, 'metadata': {'source_id': '3', 'band': 'B1', 'domain': 'code'}},
            {'text': 'D' * 4000, 'metadata': {'source_id': '4', 'band': 'B2', 'domain': 'code'}},
        ]

    def test_domain_groups_same_domain_together(self):
        packer = LinearPacker(self.tokenizer, min_tokens=1500, strategy='domain')
        output = list(packer.process(iter(self.samples)))

        # Wiki samples (1,2) should be grouped, Code samples (3,4) should be grouped
        wiki_outputs = [o for o in output if o['metadata'].get('domain') == 'wiki']
        code_outputs = [o for o in output if o['metadata'].get('domain') == 'code']

        self.assertTrue(len(wiki_outputs) >= 1)
        self.assertTrue(len(code_outputs) >= 1)

        # Verify source_ids are aggregated correctly
        for item in wiki_outputs:
            ids = item['metadata'].get('source_ids', [])
            # Should contain wiki source_ids, not code
            for sid in ids:
                self.assertIn(sid, ['1', '2'])

    def test_domain_mixes_different_bands(self):
        """Domain strategy can mix B1, B2, B3 if same domain."""
        packer = LinearPacker(self.tokenizer, min_tokens=1500, strategy='domain')
        output = list(packer.process(iter(self.samples)))

        # Find wiki group - should have both B1 and B3 mixed
        wiki_output = [o for o in output if o['metadata'].get('domain') == 'wiki'][0]
        ids = wiki_output['metadata'].get('source_ids', [])
        
        # Both id '1' (B1) and '2' (B3) should be in same group
        self.assertIn('1', ids)
        self.assertIn('2', ids)


class TestStrategyBand(unittest.TestCase):
    """Tests for Strategy 2: Band-based grouping (B0-B5)."""

    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.samples = [
            {'text': 'A' * 4000, 'metadata': {'source_id': '1', 'band': 'B1', 'domain': 'wiki'}},
            {'text': 'B' * 4000, 'metadata': {'source_id': '2', 'band': 'B1', 'domain': 'code'}},
            {'text': 'C' * 4000, 'metadata': {'source_id': '3', 'band': 'B2', 'domain': 'wiki'}},
            {'text': 'D' * 4000, 'metadata': {'source_id': '4', 'band': 'B2', 'domain': 'code'}},
        ]

    def test_band_groups_same_band_together(self):
        packer = LinearPacker(self.tokenizer, min_tokens=1500, strategy='band')
        output = list(packer.process(iter(self.samples)))

        b1_outputs = [o for o in output if o['metadata'].get('band') == 'B1']
        b2_outputs = [o for o in output if o['metadata'].get('band') == 'B2']

        self.assertTrue(len(b1_outputs) >= 1)
        self.assertTrue(len(b2_outputs) >= 1)

    def test_band_does_not_mix_bands(self):
        """B1 and B2 should never be in the same output."""
        packer = LinearPacker(self.tokenizer, min_tokens=1500, strategy='band')
        output = list(packer.process(iter(self.samples)))

        for item in output:
            ids = item['metadata'].get('source_ids', [])
            # If B1 (1,2) present, B2 (3,4) should not be
            has_b1 = any(sid in ['1', '2'] for sid in ids)
            has_b2 = any(sid in ['3', '4'] for sid in ids)
            self.assertFalse(has_b1 and has_b2, "B1 and B2 should not mix")

    def test_band_mixes_different_domains(self):
        """Band strategy allows mixing wiki and code if same band."""
        packer = LinearPacker(self.tokenizer, min_tokens=1500, strategy='band')
        output = list(packer.process(iter(self.samples)))

        b1_output = [o for o in output if o['metadata'].get('band') == 'B1'][0]
        ids = b1_output['metadata'].get('source_ids', [])
        
        # Both id '1' (wiki) and '2' (code) should be in same B1 group
        self.assertIn('1', ids)
        self.assertIn('2', ids)


class TestStrategyStrict(unittest.TestCase):
    """Tests for Strategy 3: Strict (Band + Domain) grouping."""

    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.samples = [
            {'text': 'A' * 4000, 'metadata': {'source_id': '1', 'band': 'B1', 'domain': 'wiki'}},
            {'text': 'B' * 4000, 'metadata': {'source_id': '2', 'band': 'B1', 'domain': 'wiki'}},
            {'text': 'C' * 4000, 'metadata': {'source_id': '3', 'band': 'B1', 'domain': 'code'}},
            {'text': 'D' * 4000, 'metadata': {'source_id': '4', 'band': 'B2', 'domain': 'wiki'}},
        ]

    def test_strict_groups_by_band_and_domain(self):
        packer = LinearPacker(self.tokenizer, min_tokens=1500, strategy='strict')
        output = list(packer.process(iter(self.samples)))

        # Should have separate groups for: B1+wiki, B1+code, B2+wiki
        self.assertTrue(len(output) >= 3)

    def test_strict_does_not_mix_different_domains_same_band(self):
        """B1+wiki and B1+code should NOT mix."""
        packer = LinearPacker(self.tokenizer, min_tokens=1500, strategy='strict')
        output = list(packer.process(iter(self.samples)))

        for item in output:
            ids = item['metadata'].get('source_ids', [])
            # '1','2' are B1+wiki, '3' is B1+code. Should never mix.
            has_b1_wiki = any(sid in ['1', '2'] for sid in ids)
            has_b1_code = '3' in ids
            self.assertFalse(has_b1_wiki and has_b1_code, "B1+wiki and B1+code should not mix")

    def test_strict_does_not_mix_different_bands_same_domain(self):
        """B1+wiki and B2+wiki should NOT mix."""
        packer = LinearPacker(self.tokenizer, min_tokens=1500, strategy='strict')
        output = list(packer.process(iter(self.samples)))

        for item in output:
            ids = item['metadata'].get('source_ids', [])
            # '1','2' are B1+wiki, '4' is B2+wiki. Should never mix.
            has_b1_wiki = any(sid in ['1', '2'] for sid in ids)
            has_b2_wiki = '4' in ids
            self.assertFalse(has_b1_wiki and has_b2_wiki, "B1+wiki and B2+wiki should not mix")


class TestStrategyDiscard(unittest.TestCase):
    """Tests for Strategy 4: Discard mode."""

    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.samples = [
            {'text': 'A' * 2000, 'metadata': {'source_id': '1', 'band': 'B1', 'domain': 'wiki'}},   # 500 tokens
            {'text': 'B' * 8000, 'metadata': {'source_id': '2', 'band': 'B1', 'domain': 'wiki'}},   # 2000 tokens
            {'text': 'C' * 20000, 'metadata': {'source_id': '3', 'band': 'B2', 'domain': 'code'}},  # 5000 tokens
            {'text': 'D' * 1000, 'metadata': {'source_id': '4', 'band': 'B2', 'domain': 'code'}},   # 250 tokens
        ]

    def test_discard_filters_below_threshold(self):
        packer = LinearPacker(self.tokenizer, min_tokens=1000, strategy='discard')
        output = list(packer.process(iter(self.samples)))

        # Only '2' (2000 tokens) and '3' (5000 tokens) should pass
        ids = [o['metadata']['source_id'] for o in output]
        self.assertEqual(len(output), 2)
        self.assertIn('2', ids)
        self.assertIn('3', ids)
        self.assertNotIn('1', ids)
        self.assertNotIn('4', ids)

    def test_discard_preserves_original_metadata(self):
        """Discard mode should NOT aggregate source_ids."""
        packer = LinearPacker(self.tokenizer, min_tokens=1000, strategy='discard')
        output = list(packer.process(iter(self.samples)))

        for item in output:
            # Original source_id should be preserved, not source_ids list
            self.assertIn('source_id', item['metadata'])

    def test_discard_keeps_samples_exactly_at_threshold(self):
        samples = [
            {'text': 'A' * 4000, 'metadata': {'source_id': '1'}},  # Exactly 1000 tokens
        ]
        packer = LinearPacker(self.tokenizer, min_tokens=1000, strategy='discard')
        output = list(packer.process(iter(samples)))

        self.assertEqual(len(output), 1)


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def setUp(self):
        self.tokenizer = MockTokenizer()

    def test_empty_input_stream(self):
        packer = LinearPacker(self.tokenizer, min_tokens=4096, strategy='strict')
        output = list(packer.process(iter([])))
        self.assertEqual(len(output), 0)

    def test_empty_text_skipped(self):
        samples = [
            {'text': '', 'metadata': {'source_id': '1'}},
            {'text': 'A' * 20000, 'metadata': {'source_id': '2', 'band': 'B1', 'domain': 'wiki'}},
        ]
        packer = LinearPacker(self.tokenizer, min_tokens=1000, strategy='strict')
        output = list(packer.process(iter(samples)))

        ids = []
        for item in output:
            ids.extend(item['metadata'].get('source_ids', []))
        
        self.assertNotIn('1', ids)
        self.assertIn('2', ids)

    def test_missing_metadata_uses_unknown(self):
        samples = [
            {'text': 'A' * 20000, 'metadata': {'source_id': '1'}},  # No band/domain
        ]
        packer = LinearPacker(self.tokenizer, min_tokens=1000, strategy='strict')
        output = list(packer.process(iter(samples)))
        
        # Should work and use 'unknown' as key
        self.assertEqual(len(output), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
