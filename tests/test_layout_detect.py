"""Tests for PP-Layout box parse + draw."""

from __future__ import annotations

import unittest

import numpy as np

from layout_detect import (
    LayoutBox,
    detect_layout,
    detect_text_lines,
    draw_layout_boxes,
    parse_layout_boxes,
    parse_rec_result,
    parse_text_boxes,
    recognize_boxes,
)


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


class ParseTextBoxesTest(unittest.TestCase):
    def test_parses_dt_polys_to_axis_aligned_boxes(self) -> None:
        raw = {
            "dt_polys": [
                [[10, 20], [80, 20], [80, 40], [10, 40]],
                [[5, 50], [40, 52], [39, 70], [4, 68]],
            ],
            "dt_scores": [0.91, 0.77],
        }

        boxes = parse_text_boxes(raw)

        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0].label, "text")
        self.assertAlmostEqual(boxes[0].score, 0.91)
        self.assertEqual(boxes[0].coordinate, (10.0, 20.0, 80.0, 40.0))
        self.assertEqual(boxes[1].coordinate, (4.0, 50.0, 40.0, 70.0))

    def test_parses_nested_res_wrapper(self) -> None:
        raw = {
            "res": {
                "dt_polys": [[[0, 0], [10, 0], [10, 5], [0, 5]]],
                "dt_scores": [0.5],
            }
        }

        boxes = parse_text_boxes(raw)
        self.assertEqual(boxes[0].coordinate, (0.0, 0.0, 10.0, 5.0))

    def test_parses_numpy_dt_polys_without_truth_value_error(self) -> None:
        raw = {
            "dt_polys": np.array(
                [[[10, 20], [80, 20], [80, 40], [10, 40]]],
                dtype=np.int16,
            ),
            "dt_scores": np.array([0.91]),
        }

        boxes = parse_text_boxes(raw)

        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].score, 0.91)
        self.assertEqual(boxes[0].coordinate, (10.0, 20.0, 80.0, 40.0))


class ParseRecResultTest(unittest.TestCase):
    def test_parses_nested_rec_text(self) -> None:
        text, score = parse_rec_result({"res": {"rec_text": "카드번호", "rec_score": 0.95}})
        self.assertEqual(text, "카드번호")
        self.assertAlmostEqual(score, 0.95)


class RecognizeBoxesTest(unittest.TestCase):
    def test_fills_text_via_injected_predictor(self) -> None:
        image = np.full((40, 60, 3), 200, dtype=np.uint8)
        boxes = [
            LayoutBox(label="text", score=0.9, coordinate=(5.0, 5.0, 30.0, 20.0)),
            LayoutBox(label="text", score=0.8, coordinate=(5.0, 22.0, 40.0, 35.0)),
        ]
        calls: list[np.ndarray] = []

        def predictor(crops: list[np.ndarray]) -> list[dict]:
            calls.extend(crops)
            return [
                {"rec_text": "카드", "rec_score": 0.91},
                {"rec_text": "번호", "rec_score": 0.88},
            ]

        out = recognize_boxes(image, boxes, predictor=predictor)

        self.assertEqual(len(calls), 2)
        self.assertEqual(out[0].text, "카드")
        self.assertAlmostEqual(out[0].rec_score, 0.91)
        self.assertEqual(out[1].text, "번호")
        self.assertEqual(out[0].coordinate, boxes[0].coordinate)


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


class DetectTextLinesTest(unittest.TestCase):
    def test_uses_injected_predictor_and_annotates(self) -> None:
        image = np.full((40, 60, 3), 200, dtype=np.uint8)

        def predictor(_img: np.ndarray) -> dict:
            return {
                "dt_polys": [[[5, 5], [30, 5], [30, 20], [5, 20]]],
                "dt_scores": [0.88],
            }

        result = detect_text_lines(image, predictor=predictor)

        self.assertEqual(len(result.boxes), 1)
        self.assertEqual(result.boxes[0].label, "text")
        self.assertEqual(result.boxes[0].coordinate, (5.0, 5.0, 30.0, 20.0))
        self.assertFalse(np.array_equal(result.annotated, image))


if __name__ == "__main__":
    unittest.main()
