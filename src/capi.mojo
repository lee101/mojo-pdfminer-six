from std.sys import simd_width_of


comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]


def _box(boxes: FPtr, i: Int, axis: Int) -> Float64:
    return boxes[unsafe_offset=i * 4 + axis]


def _find_root(parents: IPtr, i: Int) -> Int:
    var root = i
    while Int(parents[unsafe_offset=root]) != root:
        root = Int(parents[unsafe_offset=root])
    return root


def _unite(parents: IPtr, a: Int, b: Int):
    var ra = _find_root(parents, a)
    var rb = _find_root(parents, b)
    if ra == rb:
        return
    if ra < rb:
        parents[unsafe_offset=rb] = Int64(ra)
    else:
        parents[unsafe_offset=ra] = Int64(rb)


def _line_neighbor(
    boxes: FPtr, kinds: BPtr, i: Int, j: Int, ratio: Float64
) -> Bool:
    if kinds[unsafe_offset=i] != kinds[unsafe_offset=j]:
        return False
    var ax0 = _box(boxes, i, 0)
    var ay0 = _box(boxes, i, 1)
    var ax1 = _box(boxes, i, 2)
    var ay1 = _box(boxes, i, 3)
    var bx0 = _box(boxes, j, 0)
    var by0 = _box(boxes, j, 1)
    var bx1 = _box(boxes, j, 2)
    var by1 = _box(boxes, j, 3)
    if kinds[unsafe_offset=i] == 0:
        var d = ratio * (ay1 - ay0)
        if bx1 <= ax0 or ax1 <= bx0 or by1 <= ay0 - d or ay1 + d <= by0:
            return False
        var same = abs((by1 - by0) - (ay1 - ay0)) <= d
        var aligned = (
            abs(bx0 - ax0) <= d
            or abs(bx1 - ax1) <= d
            or abs((bx0 + bx1) * 0.5 - (ax0 + ax1) * 0.5) <= d
        )
        return same and aligned
    var d = ratio * (ax1 - ax0)
    if bx1 <= ax0 - d or ax1 + d <= bx0 or by1 <= ay0 or ay1 <= by0:
        return False
    var same = abs((bx1 - bx0) - (ax1 - ax0)) <= d
    var aligned = (
        abs(by0 - ay0) <= d
        or abs(by1 - ay1) <= d
        or abs((by0 + by1) * 0.5 - (ay0 + ay1) * 0.5) <= d
    )
    return same and aligned


def _group_char_one(
    boxes: FPtr,
    relations: BPtr,
    i: Int,
    line_overlap: Float64,
    char_margin: Float64,
    detect_vertical: Int,
):
    var ax0 = _box(boxes, i, 0)
    var ay0 = _box(boxes, i, 1)
    var ax1 = _box(boxes, i, 2)
    var ay1 = _box(boxes, i, 3)
    var bx0 = _box(boxes, i + 1, 0)
    var by0 = _box(boxes, i + 1, 1)
    var bx1 = _box(boxes, i + 1, 2)
    var by1 = _box(boxes, i + 1, 3)
    var aw = ax1 - ax0
    var ah = ay1 - ay0
    var bw = bx1 - bx0
    var bh = by1 - by0

    var vover = min(abs(ay0 - by1), abs(ay1 - by0))
    var vdist = 0.0
    if ay1 <= by0 or by1 <= ay0:
        vover = 0.0
        vdist = min(abs(ay0 - by1), abs(ay1 - by0))
    var hover = min(abs(ax0 - bx1), abs(ax1 - bx0))
    var hdist = 0.0
    if ax1 <= bx0 or bx1 <= ax0:
        hover = 0.0
        hdist = min(abs(ax0 - bx1), abs(ax1 - bx0))

    var flags = UInt8(0)
    if (
        vover > min(ah, bh) * line_overlap
        and hdist < max(aw, bw) * char_margin
    ):
        flags = flags | UInt8(1)
    if (
        detect_vertical != 0
        and hover > min(aw, bw) * line_overlap
        and vdist < max(ah, bh) * char_margin
    ):
        flags = flags | UInt8(2)
    relations[unsafe_offset=i] = flags


def _group_char_range(
    boxes: FPtr,
    relations: BPtr,
    start: Int,
    end: Int,
    line_overlap: Float64,
    char_margin: Float64,
    detect_vertical: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var i = start
    while i + W <= end:
        var ax0: SIMD[DType.float64, W] = (
            boxes.unsafe_offset(i * 4)
        ).unsafe_strided_load[width=W](4)
        var ay0: SIMD[DType.float64, W] = (
            boxes.unsafe_offset(i * 4 + 1)
        ).unsafe_strided_load[width=W](4)
        var ax1: SIMD[DType.float64, W] = (
            boxes.unsafe_offset(i * 4 + 2)
        ).unsafe_strided_load[width=W](4)
        var ay1: SIMD[DType.float64, W] = (
            boxes.unsafe_offset(i * 4 + 3)
        ).unsafe_strided_load[width=W](4)
        var bx0: SIMD[DType.float64, W] = (
            boxes.unsafe_offset((i + 1) * 4)
        ).unsafe_strided_load[width=W](4)
        var by0: SIMD[DType.float64, W] = (
            boxes.unsafe_offset((i + 1) * 4 + 1)
        ).unsafe_strided_load[width=W](4)
        var bx1: SIMD[DType.float64, W] = (
            boxes.unsafe_offset((i + 1) * 4 + 2)
        ).unsafe_strided_load[width=W](4)
        var by1: SIMD[DType.float64, W] = (
            boxes.unsafe_offset((i + 1) * 4 + 3)
        ).unsafe_strided_load[width=W](4)
        var aw = ax1 - ax0
        var ah = ay1 - ay0
        var bw = bx1 - bx0
        var bh = by1 - by0

        var vseparated = ay1.le(by0) | by1.le(ay0)
        var vedge = min(abs(ay0 - by1), abs(ay1 - by0))
        var vover = vseparated.select(0.0, vedge)
        var vdist = vseparated.select(vedge, 0.0)
        var hseparated = ax1.le(bx0) | bx1.le(ax0)
        var hedge = min(abs(ax0 - bx1), abs(ax1 - bx0))
        var hover = hseparated.select(0.0, hedge)
        var hdist = hseparated.select(hedge, 0.0)

        var halign = (
            vover.gt(min(ah, bh) * line_overlap)
            & hdist.lt(max(aw, bw) * char_margin)
        )
        var flags = halign.cast[DType.uint8]()
        if detect_vertical != 0:
            var valign = (
                hover.gt(min(aw, bw) * line_overlap)
                & vdist.lt(max(ah, bh) * char_margin)
            )
            flags = flags | (valign.cast[DType.uint8]() * UInt8(2))
        relations.unsafe_store(i, flags)
        i += W
    while i < end:
        _group_char_one(
            boxes,
            relations,
            i,
            line_overlap,
            char_margin,
            detect_vertical,
        )
        i += 1


@export("mpdf_group_chars")
def mpdf_group_chars(
    boxes_addr: Int,
    n: Int,
    line_overlap: Float64,
    char_margin: Float64,
    detect_vertical: Int,
    relations_addr: Int,
) abi("C"):
    if n <= 1:
        return
    var boxes = FPtr(unsafe_from_address=boxes_addr)
    var relations = BPtr(unsafe_from_address=relations_addr)
    var count = n - 1
    if count < 65536:
        _group_char_range(
            boxes,
            relations,
            0,
            count,
            line_overlap,
            char_margin,
            detect_vertical,
        )
        return

    comptime chunk_size = 16384
    var task_count = (count + chunk_size - 1) // chunk_size

    for task in range(task_count):
        var start = task * chunk_size
        var end = min(start + chunk_size, count)
        _group_char_range(
            boxes,
            relations,
            start,
            end,
            line_overlap,
            char_margin,
            detect_vertical,
        )


@export("mpdf_group_lines")
def mpdf_group_lines(
    boxes_addr: Int,
    kinds_addr: Int,
    n: Int,
    line_margin: Float64,
    page_x0: Float64,
    page_y0: Float64,
    page_x1: Float64,
    page_y1: Float64,
    h_order_addr: Int,
    h_prefix_addr: Int,
    v_order_addr: Int,
    v_prefix_addr: Int,
    parents_addr: Int,
) abi("C"):
    if n <= 0:
        return
    var boxes = FPtr(unsafe_from_address=boxes_addr)
    var kinds = BPtr(unsafe_from_address=kinds_addr)
    var h_order = IPtr(unsafe_from_address=h_order_addr)
    var h_prefix = FPtr(unsafe_from_address=h_prefix_addr)
    var v_order = IPtr(unsafe_from_address=v_order_addr)
    var v_prefix = FPtr(unsafe_from_address=v_prefix_addr)
    var parents = IPtr(unsafe_from_address=parents_addr)
    for i in range(n):
        parents[unsafe_offset=i] = Int64(i)

    for i in range(n):
        if (
            _box(boxes, i, 2) <= page_x0
            or page_x1 <= _box(boxes, i, 0)
            or _box(boxes, i, 3) <= page_y0
            or page_y1 <= _box(boxes, i, 1)
        ):
            continue
        var order = h_order
        var prefix = h_prefix
        var low_axis = 1
        var high_axis = 3
        var size = _box(boxes, i, 3) - _box(boxes, i, 1)
        if kinds[unsafe_offset=i] != 0:
            order = v_order
            prefix = v_prefix
            low_axis = 0
            high_axis = 2
            size = _box(boxes, i, 2) - _box(boxes, i, 0)
        var d = line_margin * size
        var query_low = _box(boxes, i, low_axis) - d
        var query_high = _box(boxes, i, high_axis) + d

        var lo = 0
        var hi = n
        while lo < hi:
            var mid = (lo + hi) // 2
            if prefix[unsafe_offset=mid] <= query_low:
                lo = mid + 1
            else:
                hi = mid
        var begin = lo

        lo = 0
        hi = n
        while lo < hi:
            var mid = (lo + hi) // 2
            var j = Int(order[unsafe_offset=mid])
            if _box(boxes, j, low_axis) < query_high:
                lo = mid + 1
            else:
                hi = mid
        var end = lo
        for pos in range(begin, end):
            var j = Int(order[unsafe_offset=pos])
            if (
                _box(boxes, j, 2) <= page_x0
                or page_x1 <= _box(boxes, j, 0)
                or _box(boxes, j, 3) <= page_y0
                or page_y1 <= _box(boxes, j, 1)
            ):
                continue
            if _line_neighbor(boxes, kinds, i, j, line_margin):
                _unite(parents, i, j)

    for i in range(n):
        parents[unsafe_offset=i] = Int64(_find_root(parents, i))


@export("mpdf_box_distances")
def mpdf_box_distances(
    boxes_addr: Int, n: Int, distances_addr: Int
) abi("C"):
    if n <= 1:
        return
    var boxes = FPtr(unsafe_from_address=boxes_addr)
    var distances = FPtr(unsafe_from_address=distances_addr)
    var k = 0
    for i in range(n):
        var ax0 = _box(boxes, i, 0)
        var ay0 = _box(boxes, i, 1)
        var ax1 = _box(boxes, i, 2)
        var ay1 = _box(boxes, i, 3)
        var area_a = (ax1 - ax0) * (ay1 - ay0)
        for j in range(i + 1, n):
            var bx0 = _box(boxes, j, 0)
            var by0 = _box(boxes, j, 1)
            var bx1 = _box(boxes, j, 2)
            var by1 = _box(boxes, j, 3)
            var area_b = (bx1 - bx0) * (by1 - by0)
            distances[unsafe_offset=k] = (
                (max(ax1, bx1) - min(ax0, bx0))
                * (max(ay1, by1) - min(ay0, by0))
                - area_a
                - area_b
            )
            k += 1
