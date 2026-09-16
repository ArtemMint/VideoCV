#include <cstdint>
#include <stdexcept>
#include <vector>

#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "NumpyMat.hpp"

namespace py = pybind11;

void process_pipeline(const py::array_t<uint8_t>& frame)
{
    cv::Mat curr = NumpyMat::numpyToMat(frame);
    if (curr.empty())
        throw std::invalid_argument("process_pipeline cannot process an empty frame");

    double mean_motion = 0.0;

    {
        py::gil_scoped_release release;

        double lowThreshold = 50;
        double highThreshold = 150; // Commonly a 1:2 or 1:3 ratio with lowThreshold
        int kernel_size = 3;

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::Mat gray, blurred, edges;

        cv::cvtColor(curr, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 1);
        cv::Canny(blurred, edges, lowThreshold, highThreshold, kernel_size);
        cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours)
        {
            // Filter out small noise artifacts by area size
            if (cv::contourArea(contour) > 100)
            {
                cv::Rect rect = cv::boundingRect(contour);

                // Draw a green rectangle with a thickness of 2
                cv::rectangle(curr, rect, cv::Scalar(0, 255, 0), 2);
            }
        }
    }
}

void init_frame_pipeline(py::module_& m)
{
    m.def("process_pipeline", &process_pipeline,
          py::arg("frame"),
          "Executes Canny filtering, contour detection, and annotation in C++");
}
