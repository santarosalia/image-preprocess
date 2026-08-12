"""Tests for PP-Layout box parse + draw."""

from __future__ import annotations

import unittest

import numpy as np

from layout_detect import LayoutBox, detect_layout, draw_layout_boxes, parse_layout_boxes


class ParseLayoutBoxesTest(unittest.TestCase):
    def test_parses_boxes_from_paddleocr_dict(self) -> None:
        raw = {
            "boxes": [
                {
                    "cls_id": 2,
                    "label": "text",
                    "score": 0.98,
                    "coordinate": [10.0, 20.0, 100.0, 80.0],
                },
                {
                    "cls_id": 8,
                    "label": "table",
                    "score": 0.91,
                    "coordinate": [5.2, 6.4, 50.8, 40.1],
                },
            ]
        }

        boxes = parse_layout_boxes(raw)

        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0].label, "text")
        self.assertAlmostEqual(boxes[0].score, 0.98)
        self.assertEqual(boxes[0].cls_id, 2)
        self.assertEqual(boxes[0].coordinate, (10.0, 20.0, 100.0, 80.0))
        self.assertEqual(boxes[1].label, "table")

    def test_parses_nested_res_wrapper(self) -> None:
        raw = {
            "res": {
                "input_path": "layout.jpg",
                "boxes": [
                    {
                        "cls_id": 0,
                        "label": "paragraph_title",
                        "score": 0.88,
                        "coordinate": [1, 2, 3, 4],
                    }
                ],
            }
        }

        boxes = parse_layout_boxes(raw)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].label, "paragraph_title")
        self.assertEqual(boxes[0].coordinate, (1.0, 2.0, 3.0, 4.0))

    def test_parses_object_with_json_attr(self) -> None:
        class FakeResult:
            json = {
                "res": {
                    "boxes": [
                        {
                            "cls_id": 1,
                            "label": "figure",
                            "score": 0.7,
                            "coordinate": [0, 0, 10, 10],
                        }
                    ]
                }
            }

        boxes = parse_layout_boxes(FakeResult())
        self.assertEqual(boxes[0].label, "figure")


class DrawLayoutBoxesTest(unittest.TestCase):
    def test_draws_box_border_on_copy(self) -> None:
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        boxes = [
            LayoutBox(
                label="text",
                score=0.95,
                coordinate=(20.0, 10.0, 80.0, 50.0),
                cls_id=2,
            )
        ]

        out = draw_layout_boxes(image, boxes)

        self.assertEqual(out.shape, image.shape)
        self.assertTrue(np.array_equal(image, np.full((80, 120, 3), 255, dtype=np.uint8)))
        self.assertFalse(np.array_equal(out[10, 20], image[10, 20]))
        self.assertTrue(np.array_equal(out[0, 0], image[0, 0]))


class DetectLayoutTest(unittest.TestCase):
    def test_uses_injected_predictor_and_annotates(self) -> None:
        image = np.full((40, 60, 3), 200, dtype=np.uint8)

        def predictor(_img: np.ndarray) -> dict:
            return {
                "boxes": [
                    {
                        "cls_id": 2,
                        "label": "text",
                        "score": 0.9,
                        "coordinate": [5.0, 5.0, 30.0, 25.0],
                    }
                ]
            }

        result = detect_layout(
            image,
            model_name="PP-DocLayoutV3",
            predictor=predictor,
        )

        self.assertEqual(len(result.boxes), 1)
        self.assertEqual(result.boxes[0].label, "text")
        self.assertEqual(result.model_name, "PP-DocLayoutV3")
        self.assertEqual(result.annotated.shape, image.shape)
        self.assertFalse(np.array_equal(result.annotated, image))


if __name__ == "__main__":
    unittest.main()
