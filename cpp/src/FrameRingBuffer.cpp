#include "FrameRingBuffer.hpp"
#include "NumpyMat.hpp"
#include <stdexcept>

FrameRingBuffer::FrameRingBuffer(std::size_t capacity)
    : m_capacity(capacity), m_head(0), m_size(0)
{
    if (capacity == 0)
    {
        throw std::invalid_argument("FrameRingBuffer capacity must be greater than zero");
    }
    m_buffer.resize(capacity);
}

std::size_t FrameRingBuffer::getCapacity() const
{
    return m_capacity;
}

std::size_t FrameRingBuffer::getSize() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_size;
}

void FrameRingBuffer::clear()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_buffer.clear();
    m_head = 0;
    m_size = 0;
}

void FrameRingBuffer::push(
    const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& frame)
{
    cv::Mat input_frame = NumpyMat::numpyToMat(frame);
    if (input_frame.empty())
    {
        throw std::invalid_argument("FrameRingBuffer cannot push an empty frame");
    }

    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(m_mutex);
        input_frame.copyTo(m_buffer[m_head]);

        m_head = (m_head + 1) % m_capacity;
        if (m_size < m_capacity)
        {
            ++m_size;
        }
    }
}

py::list FrameRingBuffer::getAsBatch() const
{
    std::vector<cv::Mat> frames_copy;
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(m_mutex);

        frames_copy.reserve(m_size);

        std::size_t startIdx = (m_size < m_capacity) ? 0 : m_head;

        for (std::size_t i = 0; i < m_size; ++i)
        {
            std::size_t idx = (startIdx + i) % m_capacity;
            frames_copy.push_back(m_buffer[idx].clone());
        }
    }
    py::list frames;
    {
        for (const auto& frame : frames_copy)
        {
            frames.append(NumpyMat::matToNumpy(frame));
        }
    }
    return frames;
}
