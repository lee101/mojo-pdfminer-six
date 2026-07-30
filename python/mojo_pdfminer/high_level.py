from __future__ import annotations

from collections.abc import Container, Iterator
from typing import Any, BinaryIO

from pdfminer import high_level as _upstream
from pdfminer.layout import LAParams, LTPage

from .layout import install


def extract_pages(
    pdf_file,
    password: str = "",
    page_numbers: Container[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    laparams: LAParams | None = None,
) -> Iterator[LTPage]:
    install()
    yield from _upstream.extract_pages(
        pdf_file, password, page_numbers, maxpages, caching, laparams
    )


def extract_text(
    pdf_file,
    password: str = "",
    page_numbers: Container[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    codec: str = "utf-8",
    laparams: LAParams | None = None,
) -> str:
    install()
    return _upstream.extract_text(
        pdf_file, password, page_numbers, maxpages, caching, codec, laparams
    )


def extract_text_to_fp(
    inf: BinaryIO,
    outfp,
    output_type: str = "text",
    codec: str = "utf-8",
    laparams: LAParams | None = None,
    maxpages: int = 0,
    page_numbers: Container[int] | None = None,
    password: str = "",
    scale: float = 1.0,
    rotation: int = 0,
    layoutmode: str = "normal",
    output_dir: str | None = None,
    strip_control: bool = False,
    debug: bool = False,
    disable_caching: bool = False,
    **kwargs: Any,
) -> None:
    install()
    _upstream.extract_text_to_fp(
        inf,
        outfp,
        output_type,
        codec,
        laparams,
        maxpages,
        page_numbers,
        password,
        scale,
        rotation,
        layoutmode,
        output_dir,
        strip_control,
        debug,
        disable_caching,
        **kwargs,
    )
