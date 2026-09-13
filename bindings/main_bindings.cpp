#include <cstdint>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>

#include "FrameRingBuffer.hpp"
#include "NumpyMat.hpp"

namespace py = pybind11;

std::tuple<double, py::array_t<uint8_t>>
process_motion_and_blur(
    py::array_t<uint8_t> curr_frame_arr,
    py::array_t<uint8_t> prev_frame_arr,
    double threshold_val)
{
    cv::Mat curr = NumpyMat::numpyToMat(curr_frame_arr);
    cv::Mat prev = NumpyMat::numpyToMat(prev_frame_arr);

    cv::Mat output_frame;
    double mean_motion = 0.0;

    {
        py::gil_scoped_release release;

        curr.copyTo(output_frame);

        cv::Mat diff, gray, thresh;
        cv::absdiff(curr, prev, diff);
        cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, thresh, 25, 255, cv::THRESH_BINARY);

        mean_motion = cv::mean(thresh)[0];

        if (mean_motion > threshold_val)
        {
            cv::GaussianBlur(curr, output_frame, cv::Size(5, 5), 3);
        }
    }

    py::array_t<uint8_t> result = NumpyMat::matToNumpy(output_frame);

    return std::make_tuple(mean_motion, result);
}

py::array_t<uint8_t> combine_and_annotate(
    py::array_t<uint8_t> frame1_arr,
    py::array_t<uint8_t> frame2_arr,
    double fps)
{
    cv::Mat frame1 = NumpyMat::numpyToMat(frame1_arr);
    cv::Mat frame2 = NumpyMat::numpyToMat(frame2_arr);

    cv::Mat combined;

    {
        py::gil_scoped_release release;

        cv::Mat f1_copy = frame1.clone();
        cv::Mat f2_copy = frame2.clone();

        cv::putText(f1_copy, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(f2_copy, "Annotated", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(f1_copy, "FPS: " + std::to_string(fps), cv::Point(500, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::hconcat(f1_copy, f2_copy, combined);
    }

    return NumpyMat::matToNumpy(combined);
}

struct TrackBox
{
    int x1, y1, x2, y2;
    int track_id;
};

py::array_t<uint8_t> render_roi_zoom(
    py::array_t<uint8_t> frame_arr,
    py::list tracks_list,
    int selected_track_id,
    int zoom_size = 180)
{
    cv::Mat frame = NumpyMat::numpyToMat(frame_arr);
    cv::Mat output = frame.clone();

    if (selected_track_id < 0 || tracks_list.empty())
    {
        return NumpyMat::matToNumpy(output);
    }

    // Розпаковка tracks з Python list
    int crop_x1 = -1, crop_y1 = -1, crop_x2 = -1, crop_y2 = -1;
    for (auto item : tracks_list)
    {
        auto t = item.cast<py::tuple>();
        int tid = t[6].cast<int>();
        if (tid == selected_track_id)
        {
            crop_x1 = t[0].cast<int>();
            crop_y1 = t[1].cast<int>();
            crop_x2 = t[2].cast<int>();
            crop_y2 = t[3].cast<int>();
            break;
        }
    }

    // Якщо трек знайдено та його координати колізійно коректні
    if (crop_x1 >= 0 && crop_x2 > crop_x1 && crop_y2 > crop_y1)
    {
        {
            py::gil_scoped_release release;

            // Обрізка за межами кадру (Clipping)
            crop_x1 = std::max(0, crop_x1);
            crop_y1 = std::max(0, crop_y1);
            crop_x2 = std::min(frame.cols, crop_x2);
            crop_y2 = std::min(frame.rows, crop_y2);

            cv::Rect roi_rect(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);
            cv::Mat cropped = frame(roi_rect);

            if (!cropped.empty())
            {
                // Масштабування обрізаного об'єкта до zoom_size x zoom_size
                cv::Mat resized_crop;
                cv::resize(cropped, resized_crop, cv::Size(zoom_size, zoom_size), 0, 0, cv::INTER_CUBIC);

                // Додаємо рамку навколо PiP вікна
                cv::rectangle(resized_crop, cv::Rect(0, 0, zoom_size, zoom_size), cv::Scalar(0, 255, 255), 3);

                // Визначення позиції PiP (Правий нижній куток з відступом 10px)
                int px = frame.cols - zoom_size - 10;
                int py = frame.rows - zoom_size - 10;

                if (px >= 0 && py >= 0)
                {
                    cv::Rect pip_target(px, py, zoom_size, zoom_size);
                    resized_crop.copyTo(output(pip_target));

                    // Підпис ID обраного об'єкта
                    std::string label = "ZOOM #" + std::to_string(selected_track_id);
                    cv::putText(output, label, cv::Point(px + 5, py - 8),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
                }
            }
        }
    }

    return NumpyMat::matToNumpy(output);
}

void init_frame_ring_buffer(py::module_& m);


PYBIND11_MODULE(tracker_ops, m)
{
    m.doc() = "High-performance C++ helper operations for CameraTracker";

    init_frame_ring_buffer(m);

    // m.def("process_motion_and_blur", &process_motion_and_blur, "Detects motion and applies blur");
    // m.def("combine_and_annotate", &combine_and_annotate, "Hstacks two frames and draws telemetry");
    // m.def("render_roi_zoom", &render_roi_zoom, "Renders a zoomed-in PiP of the selected ROI");
}
