"""mojo-pdfminer-six against pdfminer.six on identical layout objects."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from pdfminer import layout as upstream  # noqa: E402
from mojo_pdfminer import layout as mojo  # noqa: E402
from mojo_pdfminer._lib import box_distances  # noqa: E402


class TextComponent(upstream.LTComponent):
    def get_text(self):
        return "x"

    def analyze(self, laparams):
        pass


def timeit(function, repetitions=3):
    best = float("inf")
    value = None
    for _ in range(repetitions):
        start = time.perf_counter()
        value = function()
        best = min(best, time.perf_counter() - start)
    return best, value


def reference_distances(geometry):
    result = []
    for i, a in enumerate(geometry):
        for b in geometry[i + 1 :]:
            result.append(
                (max(a[2], b[2]) - min(a[0], b[0]))
                * (max(a[3], b[3]) - min(a[1], b[1]))
                - (a[2] - a[0]) * (a[3] - a[1])
                - (b[2] - b[0]) * (b[3] - b[1])
            )
    return result


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def print_row(name, mojo_time, reference_time):
    speedup = reference_time / mojo_time
    print(
        f"| {name} | {mojo_time * 1e3:.2f} ms | "
        f"{reference_time * 1e3:.2f} ms | {speedup:.2f}x |"
    )


def main():
    print(f"Machine: {cpu_name()}; Python {platform.python_version()}")
    print()
    print("| benchmark | Mojo-backed | pdfminer.six/Python | speedup |")
    print("| --- | ---: | ---: | ---: |")

    count = 300_000
    chars = [
        TextComponent((i * 5.0, 100.0, i * 5.0 + 6.0, 110.0))
        for i in range(count)
    ]
    container = upstream.LTLayoutContainer((0, 0, count * 6, 800))
    laparams = upstream.LAParams()
    mojo_time, _ = timeit(
        lambda: list(mojo._group_objects(container, laparams, chars)), 2
    )
    reference_time, _ = timeit(
        lambda: list(
            mojo._ORIGINAL_GROUP_OBJECTS(container, laparams, chars)
        ),
        2,
    )
    print_row("group_objects, 300k characters", mojo_time, reference_time)

    line_count = 10_000
    columns = 50
    lines = []
    for i in range(line_count):
        row, column = divmod(i, columns)
        x = column * 110.0
        y = row * 12.0
        line = upstream.LTTextLineHorizontal(0.1)
        line.add(TextComponent((x, y, x + 100, y + 10)))
        lines.append(line)
    container = upstream.LTLayoutContainer(
        (0, 0, columns * 110, line_count / columns * 12 + 20)
    )
    mojo_time, _ = timeit(
        lambda: list(mojo._group_textlines(container, laparams, lines)), 3
    )
    reference_time, _ = timeit(
        lambda: list(
            mojo._ORIGINAL_GROUP_TEXTLINES(container, laparams, lines)
        ),
        3,
    )
    print_row("group_textlines, 10k lines", mojo_time, reference_time)

    rng = np.random.default_rng(0)
    starts = rng.uniform(0, 700, size=(1_000, 2))
    geometry = np.column_stack(
        (starts, starts + rng.uniform(1, 100, size=(1_000, 2)))
    )
    mojo_time, _ = timeit(lambda: box_distances(geometry), 3)
    reference_time, _ = timeit(lambda: reference_distances(geometry), 2)
    print_row("initial box distances, 1k boxes", mojo_time, reference_time)

    boxes = []
    for coords in geometry[:200]:
        line = upstream.LTTextLineHorizontal(0.1)
        line.add(TextComponent(tuple(coords)))
        box = upstream.LTTextBoxHorizontal()
        box.add(line)
        boxes.append(box)
    container = upstream.LTLayoutContainer((0, 0, 850, 850))
    mojo_time, _ = timeit(
        lambda: mojo._group_textboxes(container, laparams, boxes), 1
    )
    reference_time, _ = timeit(
        lambda: mojo._ORIGINAL_GROUP_TEXTBOXES(
            container, laparams, boxes
        ),
        1,
    )
    print_row("group_textboxes, 200 boxes", mojo_time, reference_time)


if __name__ == "__main__":
    main()
