#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "FrameRingBuffer.hpp"

namespace py = pybind11;

void init_frame_ring_buffer(py::module_& m) {
    py::class_<FrameRingBuffer>(m, "FrameRingBuffer")
        .def(py::init<int>(), py::arg("capacity") = 30)
        .def("push", &FrameRingBuffer::push, py::arg("frame"))
        .def("get_as_batch", &FrameRingBuffer::getAsBatch)
        .def("clear", &FrameRingBuffer::clear)
        .def("get_size", &FrameRingBuffer::getSize)
        .def("get_capacity", &FrameRingBuffer::getCapacity);
}
