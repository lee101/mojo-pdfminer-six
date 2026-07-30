from __future__ import annotations

from io import BytesIO
import inspect
import random

import numpy as np
import pytest
from pdfminer import layout as upstream
from pdfminer.high_level import extract_pages as upstream_extract_pages

import mojo_pdfminer
from mojo_pdfminer import _lib
from mojo_pdfminer import layout as mojo_layout
from mojo_pdfminer._lib import (
    box_distances,
    character_relations,
    line_components,
)


class TextComponent(upstream.LTComponent):
    def __init__(self, bbox, text="x"):
        super().__init__(bbox)
        self.text = text

    def get_text(self):
        return self.text

    def analyze(self, laparams):
        pass


def line_signature(lines, objects):
    return [
        (type(line).__name__, [objects.index(obj) for obj in line])
        for line in lines
    ]


def box_signature(boxes, lines):
    return sorted(
        (
            type(box).__name__,
            tuple(sorted(lines.index(line) for line in box)),
        )
        for box in boxes
    )


def tree_signature(obj, leaves):
    if isinstance(obj, upstream.LTTextBox):
        return ("box", leaves.index(obj))
    return (
        type(obj).__name__,
        tuple(tree_signature(child, leaves) for child in obj),
    )


def layout_signature(obj):
    bbox = (
        tuple(round(value, 8) for value in obj.bbox)
        if hasattr(obj, "bbox")
        else None
    )
    text = obj.get_text() if hasattr(obj, "get_text") else None
    children = (
        tuple(layout_signature(child) for child in obj)
        if hasattr(obj, "__iter__")
        else ()
    )
    return type(obj).__name__, bbox, text, children


def reference_relation(a, b, laparams):
    halign = (
        a.is_voverlap(b)
        and min(a.height, b.height) * laparams.line_overlap < a.voverlap(b)
        and a.hdistance(b) < max(a.width, b.width) * laparams.char_margin
    )
    valign = (
        laparams.detect_vertical
        and a.is_hoverlap(b)
        and min(a.width, b.width) * laparams.line_overlap < a.hoverlap(b)
        and a.vdistance(b) < max(a.height, b.height) * laparams.char_margin
    )
    return int(halign) | (int(valign) << 1)


@pytest.mark.parametrize("detect_vertical", [False, True])
def test_character_relation_kernel_matches_component_predicates(detect_vertical):
    rng = random.Random(42)
    objects = []
    for _ in range(2_000):
        x0 = rng.uniform(-100, 500)
        y0 = rng.uniform(-100, 700)
        objects.append(
            TextComponent(
                (
                    x0,
                    y0,
                    x0 + rng.uniform(0.1, 40),
                    y0 + rng.uniform(0.1, 40),
                )
            )
        )
    laparams = upstream.LAParams(
        line_overlap=0.37,
        char_margin=1.73,
        detect_vertical=detect_vertical,
    )
    geometry = np.array([obj.bbox for obj in objects])
    got = character_relations(
        geometry,
        laparams.line_overlap,
        laparams.char_margin,
        laparams.detect_vertical,
    )
    expected = np.array(
        [
            reference_relation(a, b, laparams)
            for a, b in zip(objects, objects[1:])
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(got, expected)


def test_character_relation_uses_upstream_strict_boundaries():
    objects = [
        TextComponent((0, 0, 10, 10)),
        TextComponent((10, 5, 20, 15)),
    ]
    laparams = upstream.LAParams(line_overlap=0.5, char_margin=0)
    got = character_relations(
        np.array([obj.bbox for obj in objects]),
        laparams.line_overlap,
        laparams.char_margin,
        False,
    )
    assert got.tolist() == [0]


def test_character_relation_simd_tail_matches_component_predicates():
    objects = [
        TextComponent((0, 0, 5, 8)),
        TextComponent((4, 1, 9, 9)),
        TextComponent((20, 20, 27, 30)),
        TextComponent((21, 28, 28, 38)),
        TextComponent((40, 0, 50, 10)),
        TextComponent((40, 9, 50, 19)),
        TextComponent((60, 4, 70, 14)),
        TextComponent((71, 4, 81, 14)),
        TextComponent((80, 20, 90, 30)),
        TextComponent((82, 20, 92, 30)),
    ]
    laparams = upstream.LAParams(
        line_overlap=0.4,
        char_margin=1.5,
        detect_vertical=True,
    )
    got = character_relations(
        np.array([obj.bbox for obj in objects]),
        laparams.line_overlap,
        laparams.char_margin,
        laparams.detect_vertical,
    )
    expected = [
        reference_relation(a, b, laparams)
        for a, b in zip(objects, objects[1:])
    ]
    assert got.tolist() == expected


@pytest.mark.parametrize(
    "geometry",
    [
        np.zeros(8),
        np.zeros((2, 3)),
        np.zeros((2, 5)),
    ],
)
def test_kernels_reject_geometry_with_unsafe_shape(geometry):
    with pytest.raises(ValueError, match=r"shape \(n, 4\)"):
        character_relations(geometry, 0.5, 2.0, False)
    with pytest.raises(ValueError, match=r"shape \(n, 4\)"):
        box_distances(geometry)


def test_kernels_reject_complex_coordinates_instead_of_narrowing():
    geometry = np.zeros((2, 4), dtype=np.complex128)
    with pytest.raises(TypeError, match="real numbers"):
        character_relations(geometry, 0.5, 2.0, False)


@pytest.mark.parametrize(
    "kinds, error",
    [
        (np.array([0], dtype=np.uint8), ValueError),
        (np.array([0, 256]), ValueError),
        (np.array([0.0, 1.0]), TypeError),
    ],
)
def test_line_kernel_rejects_invalid_kinds_without_narrowing(kinds, error):
    with pytest.raises(error):
        line_components(np.zeros((2, 4)), kinds, 0.5)


def test_missing_configured_library_is_reported(monkeypatch, tmp_path):
    missing = str(tmp_path / "missing.so")
    monkeypatch.setattr(_lib, "_library", None)
    monkeypatch.setattr(_lib, "_CUSTOM_LIB", missing)
    monkeypatch.setattr(_lib, "LIB", missing)
    with pytest.raises(FileNotFoundError, match="MOJO_PDFMINER_LIB"):
        _lib.lib()


@pytest.mark.parametrize("count", [65_536, 65_538])
def test_character_relation_parallel_threshold(count):
    x0 = np.arange(count, dtype=np.float64)
    geometry = np.column_stack(
        (x0, np.zeros(count), x0 + 1.25, np.full(count, 10.0))
    )
    got = character_relations(geometry, 0.5, 1.0, False)
    assert got.shape == (count - 1,)
    assert np.all(got == 1)


@pytest.mark.parametrize("seed", range(5))
def test_group_objects_matches_upstream_on_random_layouts(seed):
    rng = random.Random(seed)
    objects = []
    x = y = 0.0
    for _ in range(400):
        x += rng.uniform(-3, 8) if rng.random() > 0.12 else rng.uniform(20, 70)
        y += rng.uniform(-5, 5)
        objects.append(
            TextComponent(
                (x, y, x + rng.uniform(2, 14), y + rng.uniform(5, 18))
            )
        )
    laparams = upstream.LAParams(
        line_overlap=rng.uniform(0.2, 0.8),
        char_margin=rng.uniform(0.5, 3),
        detect_vertical=bool(seed % 2),
    )
    container = upstream.LTLayoutContainer((-500, -500, 2000, 2000))
    expected = list(
        mojo_layout._ORIGINAL_GROUP_OBJECTS(container, laparams, objects)
    )
    got = list(mojo_layout._group_objects(container, laparams, objects))
    assert line_signature(got, objects) == line_signature(expected, objects)


def test_group_objects_constructs_vertical_text_line():
    objects = [
        TextComponent((10, 20, 20, 30)),
        TextComponent((10, 10, 20, 19)),
    ]
    laparams = upstream.LAParams(detect_vertical=True)
    container = upstream.LTLayoutContainer((0, 0, 100, 100))
    lines = list(mojo_layout._group_objects(container, laparams, objects))
    assert len(lines) == 1
    assert isinstance(lines[0], upstream.LTTextLineVertical)
    assert list(lines[0]) == objects


def make_lines(seed, count, mixed=True):
    rng = random.Random(seed)
    lines = []
    for i in range(count):
        x = rng.uniform(10, 550)
        y = rng.uniform(10, 750)
        width = rng.uniform(12, 100)
        height = rng.uniform(4, 25)
        cls = (
            upstream.LTTextLineVertical
            if mixed and i % 7 == 0
            else upstream.LTTextLineHorizontal
        )
        line = cls(0.1)
        line.add(TextComponent((x, y, x + width, y + height), str(i)))
        lines.append(line)
    return lines


@pytest.mark.parametrize("seed", range(4))
def test_group_textlines_matches_upstream_components(seed):
    lines = make_lines(seed, 250)
    laparams = upstream.LAParams(line_margin=0.15 + seed * 0.25)
    container = upstream.LTLayoutContainer((0, 0, 612, 792))
    expected = list(
        mojo_layout._ORIGINAL_GROUP_TEXTLINES(container, laparams, lines)
    )
    got = list(mojo_layout._group_textlines(container, laparams, lines))
    assert box_signature(got, lines) == box_signature(expected, lines)


def test_group_textlines_preserves_transitive_neighbor_groups():
    lines = []
    for y in (100, 109, 118):
        line = upstream.LTTextLineHorizontal(0.1)
        line.add(TextComponent((50, y, 150, y + 10)))
        lines.append(line)
    laparams = upstream.LAParams(line_margin=0.5)
    container = upstream.LTLayoutContainer((0, 0, 612, 792))
    boxes = list(mojo_layout._group_textlines(container, laparams, lines))
    assert len(boxes) == 1
    assert list(boxes[0]) == lines


def test_group_textlines_matches_plane_clipping_outside_page():
    lines = []
    for x in (-100, -100):
        line = upstream.LTTextLineHorizontal(0.1)
        line.add(TextComponent((x, 100, x + 20, 110)))
        lines.append(line)
    laparams = upstream.LAParams(line_margin=1)
    container = upstream.LTLayoutContainer((0, 0, 612, 792))
    expected = list(
        mojo_layout._ORIGINAL_GROUP_TEXTLINES(container, laparams, lines)
    )
    got = list(mojo_layout._group_textlines(container, laparams, lines))
    assert box_signature(got, lines) == box_signature(expected, lines)


def test_box_distance_kernel_matches_upstream_formula():
    rng = np.random.default_rng(7)
    starts = rng.uniform(-100, 700, size=(300, 2))
    sizes = rng.uniform(0.1, 100, size=(300, 2))
    geometry = np.column_stack((starts, starts + sizes))
    got = box_distances(geometry)
    expected = []
    for i, a in enumerate(geometry):
        for b in geometry[i + 1 :]:
            expected.append(
                (max(a[2], b[2]) - min(a[0], b[0]))
                * (max(a[3], b[3]) - min(a[1], b[1]))
                - (a[2] - a[0]) * (a[3] - a[1])
                - (b[2] - b[0]) * (b[3] - b[1])
            )
    assert got == pytest.approx(expected, rel=2e-15, abs=2e-10)


@pytest.mark.parametrize("seed", range(3))
def test_group_textboxes_matches_upstream_merge_tree(seed):
    lines = make_lines(seed + 20, 35, mixed=False)
    boxes = []
    for line in lines:
        box = upstream.LTTextBoxHorizontal()
        box.add(line)
        boxes.append(box)
    container = upstream.LTLayoutContainer((0, 0, 612, 792))
    laparams = upstream.LAParams()
    expected = mojo_layout._ORIGINAL_GROUP_TEXTBOXES(
        container, laparams, boxes
    )
    got = mojo_layout._group_textboxes(container, laparams, boxes)
    assert [tree_signature(x, boxes) for x in got] == [
        tree_signature(x, boxes) for x in expected
    ]


def test_install_and_uninstall_patch_upstream_methods():
    mojo_layout.install()
    assert upstream.LTLayoutContainer.group_objects is mojo_layout._group_objects
    assert (
        upstream.LTLayoutContainer.group_textlines
        is mojo_layout._group_textlines
    )
    assert (
        upstream.LTLayoutContainer.group_textboxes
        is mojo_layout._group_textboxes
    )
    mojo_layout.uninstall()
    assert (
        upstream.LTLayoutContainer.group_objects
        is mojo_layout._ORIGINAL_GROUP_OBJECTS
    )


def test_layout_method_parameter_names_match_upstream():
    pairs = [
        (mojo_layout._group_objects, mojo_layout._ORIGINAL_GROUP_OBJECTS),
        (mojo_layout._group_textlines, mojo_layout._ORIGINAL_GROUP_TEXTLINES),
        (mojo_layout._group_textboxes, mojo_layout._ORIGINAL_GROUP_TEXTBOXES),
    ]
    for mojo_fn, upstream_fn in pairs:
        assert list(inspect.signature(mojo_fn).parameters) == list(
            inspect.signature(upstream_fn).parameters
        )


def test_extract_pages_matches_real_pdfminer_layout(sample_pdf):
    laparams = upstream.LAParams()
    expected = list(
        upstream_extract_pages(BytesIO(sample_pdf), laparams=laparams)
    )
    mojo_layout.uninstall()
    got = list(
        mojo_pdfminer.extract_pages(BytesIO(sample_pdf), laparams=laparams)
    )
    assert [layout_signature(page) for page in got] == [
        layout_signature(page) for page in expected
    ]


def test_extract_text_matches_real_pdfminer(sample_pdf):
    from pdfminer.high_level import extract_text

    expected = extract_text(BytesIO(sample_pdf))
    mojo_layout.uninstall()
    got = mojo_pdfminer.extract_text(BytesIO(sample_pdf))
    assert got == expected
    assert got == "Hello Mojo\nLayout parity\n\nSecond column\n\n\x0c"


def test_extract_text_to_fp_matches_real_pdfminer(sample_pdf):
    from mojo_pdfminer import high_level
    from pdfminer import high_level as reference

    expected = BytesIO()
    reference.extract_text_to_fp(BytesIO(sample_pdf), expected)
    mojo_layout.uninstall()
    got = BytesIO()
    high_level.extract_text_to_fp(BytesIO(sample_pdf), got)
    assert got.getvalue() == expected.getvalue()


def test_high_level_defaults_match_upstream():
    from mojo_pdfminer import high_level
    from pdfminer import high_level as reference

    for name in ("extract_pages", "extract_text", "extract_text_to_fp"):
        got = inspect.signature(getattr(high_level, name))
        expected = inspect.signature(getattr(reference, name))
        assert [
            (p.name, p.kind, p.default)
            for p in got.parameters.values()
        ] == [
            (p.name, p.kind, p.default)
            for p in expected.parameters.values()
        ]
