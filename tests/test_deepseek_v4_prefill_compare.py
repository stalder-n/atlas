# SPDX-License-Identifier: AGPL-3.0-only
"""Host tests for the Spark prefill comparison's fail-closed checks."""
import unittest

from deepseek_v4_prefill_compare import compare_logits, validate_pair


class PrefillComparisonTests(unittest.TestCase):
    def test_independent_absolute_and_relative_tolerance(self):
        result = compare_logits([0.0, 10.0], [0.01, 10.1], 0.02, 0.01)
        self.assertTrue(result['pass'])
        self.assertFalse(compare_logits([0.0], [0.03], 0.02, 0.01)['pass'])

    def test_rejects_missing_truncated_and_nonfinite_rows(self):
        for reference, candidate in [([], []), ([1.0], [1.0, 2.0]),
                                     ([float('nan')], [1.0]),
                                     ([1.0], [float('inf')])]:
            with self.assertRaises(ValueError):
                compare_logits(reference, candidate, 0.01, 0.01)

    def test_rejects_invalid_tolerances(self):
        for tolerance in [-1.0, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                compare_logits([1.0], [1.0], tolerance, 0.0)

    def test_argmax_flip_is_reported_even_within_tolerance(self):
        result = compare_logits([1.0, 1.01], [1.01, 1.0], 0.02, 0.0)
        self.assertTrue(result['pass'])
        self.assertFalse(result['argmax_equal'])

    def test_requires_identical_fingerprint_requests_and_complete_runs(self):
        baseline = {'fingerprint': {'source_sha': 'a'}, 'requests': [[1, 2]],
                    'complete': True, 'mode': 'baseline'}
        candidate = dict(baseline, mode='fp8')
        validate_pair(baseline, candidate)
        for changed in [dict(candidate, fingerprint={'source_sha': 'b'}),
                        dict(candidate, requests=[[1, 3]]),
                        dict(candidate, complete=False)]:
            with self.assertRaises(ValueError):
                validate_pair(baseline, changed)


if __name__ == '__main__':
    unittest.main()
