"""
High-performance C++ helper operations for CameraTracker
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
__all__: list[str] = ['FrameRingBuffer', 'process_pipeline']
class FrameRingBuffer:
    def __init__(self, capacity: typing.SupportsInt | typing.SupportsIndex = 30) -> None:
        ...
    def clear(self) -> None:
        ...
    def get_as_batch(self) -> list:
        ...
    def get_capacity(self) -> int:
        ...
    def get_size(self) -> int:
        ...
    def push(self, frame: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8]) -> None:
        ...
def process_pipeline(frame: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8]) -> None:
    """
    Executes Sobel filtering, thresholding, contour detection, and annotation in C++
    """
