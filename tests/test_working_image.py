"""Tests for optional classic preprocess."""

from __future__ import annotations

import unittest

import numpy as np

from classic_enhance import working_image


class WorkingImageTest(unittest.TestCase):
    def test_skips_preprocess_when_disabled(self) -> None:
        rgb = np.full((10, 12, 3), 128, dtype=np.uint8)

        working, result = working_image(rgb, apply_preprocess=False)

        self.assertIsNone(result)
        np.testing.assert_array_equal(working, rgb)

    def test_applies_preprocess_when_enabled(self) -> None:
        rgb = np.full((10, 12, 3), 128, dtype=np.uint8)

        working, result = working_image(rgb, apply_preprocess=True, method="clahe")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.method, "clahe")
        np.testing.assert_array_equal(working, result.enhanced)
        self.assertEqual(working.shape, rgb.shape)
        self.assertFalse(np.array_equal(working, rgb))


if __name__ == "__main__":
    unittest.main()
