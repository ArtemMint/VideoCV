#pragma once

#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <mutex>

namespace py = pybind11;

class FrameRingBuffer
{
  private:
    std::vector<cv::Mat> m_buffer{};
    std::size_t m_capacity{30};
    std::size_t m_head{0};
    std::size_t m_size{0};
    mutable std::mutex m_mutex;

  public:
    FrameRingBuffer(std::size_t capacity = 30);

    void push(const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& frame);

    py::list getAsBatch() const;

    std::size_t getCapacity() const;
    std::size_t getSize() const;
    void clear();
};
