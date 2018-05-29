#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

void draw_histogram(const cv::Mat& hist, const cv::Mat& img, const cv::Scalar& color);

float median(const cv::Mat& m);

void mean_stddev(const cv::Mat& src, cv::Mat& mean, cv::Mat& stddev, int window_size);

cv::Mat calc_countours(const cv::Mat& src, int range);

void calc_local_max_deviation(float* data, uint32_t size, float dev_border, float& max, float& deviation);

cv::Mat generate_by_row( const std::vector<float>& row );

bool read_frame(cv::Mat& src, cv::VideoCapture& capture);

cv::Mat calc_crosses(const cv::Mat& src);

cv::Mat derivative(const cv::Mat& src, int window_size);

void max_gradient_direction(const cv::Matx33d& m, double (&directions)[8], int& max_index);

cv::Vec2f scale_min_component_to_one(const cv::Vec2f v);

cv::Vec2f closest_direction(const cv::Vec2f v, int degree);

std::vector<cv::Vec2f> all_directions( int degree );

cv::Mat paint_angles( const cv::Mat& dx, const cv::Mat& dy, double norm_threshold = 10.0 );


