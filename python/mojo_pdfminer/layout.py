from __future__ import annotations

import heapq
from typing import Iterable, Iterator, Sequence, cast

import numpy as np
from pdfminer.layout import *  # noqa: F403
from pdfminer import layout as _upstream
from pdfminer.utils import Plane, uniq

from ._lib import box_distances, boxes as geometry_of
from ._lib import character_relations, line_components

_ORIGINAL_GROUP_OBJECTS = _upstream.LTLayoutContainer.group_objects
_ORIGINAL_GROUP_TEXTLINES = _upstream.LTLayoutContainer.group_textlines
_ORIGINAL_GROUP_TEXTBOXES = _upstream.LTLayoutContainer.group_textboxes


def _group_objects(
    self: _upstream.LTLayoutContainer,
    laparams: _upstream.LAParams,
    objs: Iterable[_upstream.LTComponent],
) -> Iterator[_upstream.LTTextLine]:
    objects = objs if isinstance(objs, list) else list(objs)
    if not objects:
        yield from _ORIGINAL_GROUP_OBJECTS(self, laparams, objects)
        return
    relations = memoryview(
        character_relations(
            geometry_of(objects),
            laparams.line_overlap,
            laparams.char_margin,
            laparams.detect_vertical,
        )
    )
    obj0 = None
    line = None
    line_kind = 0
    for i, obj1 in enumerate(objects):
        if obj0 is not None:
            relation = relations[i - 1]
            halign = relation & 1
            valign = relation & 2
            if (halign and line_kind == 1) or (valign and line_kind == 2):
                line.add(obj1)
            elif line is not None:
                yield line
                line = None
                line_kind = 0
            elif valign and not halign:
                line = _upstream.LTTextLineVertical(laparams.word_margin)
                line_kind = 2
                line.add(obj0)
                line.add(obj1)
            elif halign and not valign:
                line = _upstream.LTTextLineHorizontal(laparams.word_margin)
                line_kind = 1
                line.add(obj0)
                line.add(obj1)
            else:
                line = _upstream.LTTextLineHorizontal(laparams.word_margin)
                line.add(obj0)
                yield line
                line = None
        obj0 = obj1
    if line is None:
        line = _upstream.LTTextLineHorizontal(laparams.word_margin)
        line.add(obj0)
    yield line


def _group_textlines(
    self: _upstream.LTLayoutContainer,
    laparams: _upstream.LAParams,
    lines: Iterable[_upstream.LTTextLine],
) -> Iterator[_upstream.LTTextBox]:
    line_list = list(lines)
    if not line_list:
        return
    kinds = np.fromiter(
        (
            0 if isinstance(line, _upstream.LTTextLineHorizontal) else 1
            for line in line_list
        ),
        dtype=np.uint8,
        count=len(line_list),
    )
    labels = line_components(
        geometry_of(line_list), kinds, laparams.line_margin, self.bbox
    )
    grouped: dict[int, _upstream.LTTextBox] = {}
    order: list[int] = []
    for label, line in zip(labels.tolist(), line_list):
        if label not in grouped:
            if isinstance(line, _upstream.LTTextLineHorizontal):
                box = _upstream.LTTextBoxHorizontal()
            else:
                box = _upstream.LTTextBoxVertical()
            grouped[label] = box
            order.append(label)
        grouped[label].add(line)
    for label in order:
        box = grouped[label]
        if not box.is_empty():
            yield box


def _group_textboxes(
    self: _upstream.LTLayoutContainer,
    laparams: _upstream.LAParams,
    boxes: Sequence[_upstream.LTTextBox],
) -> list[_upstream.LTTextGroup]:
    plane = Plane(self.bbox)

    def dist(obj1, obj2):
        x0 = min(obj1.x0, obj2.x0)
        y0 = min(obj1.y0, obj2.y0)
        x1 = max(obj1.x1, obj2.x1)
        y1 = max(obj1.y1, obj2.y1)
        return (
            (x1 - x0) * (y1 - y0)
            - obj1.width * obj1.height
            - obj2.width * obj2.height
        )

    def isany(obj1, obj2):
        x0 = min(obj1.x0, obj2.x0)
        y0 = min(obj1.y0, obj2.y0)
        x1 = max(obj1.x1, obj2.x1)
        y1 = max(obj1.y1, obj2.y1)
        for obj in plane.find((x0, y0, x1, y1)):
            if obj is not obj1 and obj is not obj2:
                return True
        return False

    initial = box_distances(geometry_of(boxes))
    dists = []
    k = 0
    for i, box1 in enumerate(boxes):
        for box2 in boxes[i + 1 :]:
            dists.append(
                (False, float(initial[k]), id(box1), id(box2), box1, box2)
            )
            k += 1
    heapq.heapify(dists)

    plane.extend(boxes)
    done = set()
    while dists:
        skip_isany, d, id1, id2, obj1, obj2 = heapq.heappop(dists)
        if id1 not in done and id2 not in done:
            if not skip_isany and isany(obj1, obj2):
                heapq.heappush(dists, (True, d, id1, id2, obj1, obj2))
                continue
            if isinstance(
                obj1, (_upstream.LTTextBoxVertical, _upstream.LTTextGroupTBRL)
            ) or isinstance(
                obj2, (_upstream.LTTextBoxVertical, _upstream.LTTextGroupTBRL)
            ):
                group = _upstream.LTTextGroupTBRL([obj1, obj2])
            else:
                group = _upstream.LTTextGroupLRTB([obj1, obj2])
            plane.remove(obj1)
            plane.remove(obj2)
            done.update([id1, id2])
            for other in plane:
                heapq.heappush(
                    dists,
                    (
                        False,
                        dist(group, other),
                        id(group),
                        id(other),
                        group,
                        other,
                    ),
                )
            plane.add(group)
    return [cast(_upstream.LTTextGroup, group) for group in plane]


class LTLayoutContainer(_upstream.LTLayoutContainer):
    group_objects = _group_objects
    group_textlines = _group_textlines
    group_textboxes = _group_textboxes


def install() -> None:
    _upstream.LTLayoutContainer.group_objects = _group_objects
    _upstream.LTLayoutContainer.group_textlines = _group_textlines
    _upstream.LTLayoutContainer.group_textboxes = _group_textboxes


def uninstall() -> None:
    _upstream.LTLayoutContainer.group_objects = _ORIGINAL_GROUP_OBJECTS
    _upstream.LTLayoutContainer.group_textlines = _ORIGINAL_GROUP_TEXTLINES
    _upstream.LTLayoutContainer.group_textboxes = _ORIGINAL_GROUP_TEXTBOXES
