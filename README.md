# mojo-pdfminer-six

Mojo-accelerated PDF layout analysis for
[pdfminer.six](https://github.com/pdfminer/pdfminer.six).

This is a focused port of the compute-heavy geometry inside `pdfminer.layout`,
not a PDF parser rewrite. It keeps pdfminer.six's `LTChar`, `LTTextLine`,
`LTTextBox`, `LTTextGroup`, and `LAParams` objects, while moving bulk spatial
classification and distance calculations into one compiled Mojo library.

## Covered subset

The Mojo-backed `LTLayoutContainer` implements the same method names and
signatures as upstream:

- `group_objects(laparams, objs)`: adjacent character alignment and line
  construction, including vertical-text detection.
- `group_textlines(laparams, lines)`: horizontal and vertical neighbor
  discovery followed by transitive text-box grouping.
- `group_textboxes(laparams, boxes)`: Mojo computes the initial pairwise
  distance matrix; the heap, obstruction checks, and group objects retain
  upstream behavior in Python.

`mojo_pdfminer.high_level` mirrors `extract_pages`, `extract_text`, and
`extract_text_to_fp`. These install the accelerated methods and then delegate
PDF parsing, font handling, and output conversion to the real pdfminer.six
package.

Not covered are PDF syntax and stream decoding, fonts and CMaps, image
extraction, converters, or the non-layout parts of pdfminer.six. Dynamic
hierarchical heap updates remain in Python because they operate on pdfminer's
object graph rather than dense numeric buffers.

## Install and build

The repository pins the tested Mojo nightly and installs pdfminer.six from
conda-forge:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-pdfminer-six.so`. Set
`MOJO_PDFMINER_LIB=/absolute/path/to/libmojo-pdfminer-six.so` to use a
prebuilt library in another environment.

## Usage

Use the mirrored high-level API:

```python
from mojo_pdfminer import LAParams, extract_pages

for page in extract_pages("document.pdf", laparams=LAParams()):
    for item in page:
        if hasattr(item, "get_text"):
            print(item.get_text(), end="")
```

Existing code can keep importing pdfminer directly after a one-time install:

```python
from mojo_pdfminer import install

install()

from pdfminer.high_level import extract_text

text = extract_text("document.pdf")
```

`install()` is idempotent. `mojo_pdfminer.uninstall()` restores the original
three methods, which is useful for parity testing or controlled benchmarks.

## Benchmarks

Measured with `pixi run bench` on an Intel(R) Xeon(R) CPU E5-2697 v4 @
2.30GHz using Python 3.13.14:

| benchmark | Mojo-backed | pdfminer.six/Python | speedup |
| --- | ---: | ---: | ---: |
| `group_objects`, 300k characters | 752.05 ms | 868.37 ms | 1.15x |
| `group_textlines`, 10k lines | 36.81 ms | 1282.62 ms | 34.85x |
| initial box distances, 1k boxes | 2.75 ms | 1896.65 ms | 689.63x |
| `group_textboxes`, 200 boxes | 347.16 ms | 3024.16 ms | 8.71x |

Attribute extraction limits the complete `group_objects` speedup. The
line-neighbor and pairwise-distance stages have more geometry work per Python
object and show larger gains.

These are best-of-run wall-clock measurements on synthetic but genuine
pdfminer layout objects. The benchmark constructs all inputs before timing and
uses the same objects and `LAParams` for both implementations.

## How it works

Python flattens each bounding box to a C-contiguous float64 row
`[x0, y0, x1, y1]`. Small uint8 buffers carry writing-mode and relation flags;
int64 buffers carry union-find component labels. Buffers cross the C ABI as
integer addresses and remain owned by NumPy.

The Mojo line-neighbor kernel sorts by the relevant low coordinate and uses a
prefix maximum of the high coordinate to bound each spatial search. It applies
pdfminer.six's strict intersection, size-tolerance, and edge/center-alignment
rules before joining connected lines. The box-distance kernel writes the
packed upper triangle of the pairwise matrix in upstream iteration order.

Adjacent-character classification uses the native float64 SIMD width with
strided loads over the interleaved bbox rows, contiguous flag stores, and a
scalar remainder loop. Large inputs are divided into independent tasks;
smaller inputs stay serial to avoid launch overhead.

No GPU path is included. These geometry kernels do little arithmetic per
coordinate loaded, so host/device transfer and launch overhead would dominate.

All exports live in one Mojo compilation unit. They use `@export` with the C
ABI, allocate no cross-language memory, and are loaded through `ctypes`.

## Validation

`pixi run test` compares against the installed pdfminer.six implementation.
The suite covers randomized horizontal and vertical layouts, strict boundary
conditions, transitive neighbors, page clipping, distance values, hierarchical
merge trees, method installation, signature/default compatibility, and
end-to-end extraction from a real PDF byte stream.

MIT licensed.
