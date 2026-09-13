#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace NumpyMat
{
inline cv::Mat numpyToMat(const py::array_t<std::uint8_t>& input)
{
    const py::buffer_info buffer = input.request();

    if (buffer.ndim != 2 && buffer.ndim != 3)
    {
        throw py::value_error("Expected a 2D grayscale or 3D color uint8 NumPy array");
    }

    const int channels = buffer.ndim == 3 ? static_cast<int>(buffer.shape[2]) : 1;
    if (channels != 1 && channels != 3 && channels != 4)
    {
        throw py::value_error("Expected 1, 3, or 4 channels");
    }

    const int type = CV_MAKETYPE(CV_8U, channels);
    return cv::Mat(
        static_cast<int>(buffer.shape[0]),
        static_cast<int>(buffer.shape[1]),
        type,
        buffer.ptr,
        static_cast<std::size_t>(buffer.strides[0]));
}

inline py::array_t<std::uint8_t> matToNumpy(const cv::Mat& mat)
{
    if (mat.empty())
    {
        throw py::value_error("Cannot convert an empty cv::Mat");
    }

    // Якщо mat володіє пам'яттю (mat.u != nullptr), робимо поверхневу копію.
    // Якщо не володіє (наприклад, це був view/crop чи сторонній буфер) — робимо .clone()
    cv::Mat* mat_owner = mat.u ? new cv::Mat(mat) : new cv::Mat(mat.clone());

    py::capsule owner(mat_owner, [](void* pointer)
                      { delete static_cast<cv::Mat*>(pointer); });

    if (mat_owner->channels() == 1)
    {
        return py::array_t<std::uint8_t>(
            {mat_owner->rows, mat_owner->cols},
            {static_cast<std::size_t>(mat_owner->step[0]), static_cast<std::size_t>(mat_owner->elemSize1())},
            mat_owner->data,
            owner);
    }
    if (mat.channels() == 1)
    {
        return py::array_t<std::uint8_t>(
            {mat.rows, mat.cols},
            {static_cast<std::size_t>(mat.step[0]), static_cast<std::size_t>(mat.elemSize1())},
            mat.data,
            owner);
    }

    std::vector<py::ssize_t> shape{
        mat.rows,
        mat.cols,
        mat.channels()};
    std::vector<py::ssize_t> strides{
        static_cast<py::ssize_t>(mat.step[0]),
        static_cast<py::ssize_t>(mat.elemSize()),
        static_cast<py::ssize_t>(mat.elemSize1())};

    return py::array_t<std::uint8_t>(shape, strides, mat.data, owner);
}
} // namespace NumpyMat
