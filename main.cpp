#include <iostream>
#include <typeinfo>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat convertToYCbCr(cv::Mat image) {
    // converts an RGB image to YCbCr
    // cv::Mat: B-G-R
    std::cout << "Converting image to YCbCr color space." << std::endl;
    int i, j;
    for (i = 0; i < image.cols; i++) {
        // std::cout << "At column " << i << " of total " << image.cols << std::endl;
        for (j = 0; j < image.rows; j++) {

            // R, G, B values
            auto R = image.at<cv::Vec3d>(j,i)[2];
            auto G = image.at<cv::Vec3d>(j,i)[1];
            auto B = image.at<cv::Vec3d>(j,i)[0];

            // Y'
            // image.at<cv::Vec3d>(j,i)[0] = 16 + (65.481 * R + 128.553 * G + 24.966 * B);
            image.at<cv::Vec3d>(j,i)[0] = 0.299 * R + 0.587 * G + 0.114 * B;

            // Cb
            // image.at<cv::Vec3d>(j,i)[1] = 128 + (-37.979 * R - 74.203 * G + 112 * B);
            image.at<cv::Vec3d>(j,i)[2] = (R - image.at<cv::Vec3d>(j,i)[0]) * 0.713 + 0.5;

            // Cr
            image.at<cv::Vec3d>(j,i)[1] = (B - image.at<cv::Vec3d>(j,i)[0]) * 0.564 + 0.5;
        }
    }
    std::cout << "Converting finished." << std::endl;
    return image;
}

cv::Mat applyGaussianBlur(cv::Mat image) {
    std::cout << "Applying Gaussian Blur on image." << std::endl;
    int i, j;

    for (i = 0; i < image.cols; i++) {
        for (j = 0; j < image.rows; j++) {
        

        }
    }

    return image;
}

int main(int argc, char *argv[]) {
    if (argc != 2)
    {
        std::cout <<"Usage: ./blur sourceimage" << std::endl;
        return -1;
    }

    // cv::Mat image;
    auto image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

    std::cout << "type of image is: " << typeid(image).name() << std::endl;

    if(! image.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    std::cout << "Image dimension: " << image.dims << std::endl
        << "Image rows: " << image.rows << std::endl
        << "Image columns: " << image.cols << std::endl
        << "Channels: " << image.channels() << std::endl
        << "Size: " << image.size() << std::endl
        << "Type: " << image.type() << std::endl;

    // std::cout << "Image: " << std::endl << image;

    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.


    /*
     *
     * Matrix operations
     * 1) Color space conversion RGB -> YCbCr
     *
     */
    
    // loop over array
    auto convertedImage = convertToYCbCr(image);
    cv::Mat newImage;
    cv::cvtColor(image, newImage, cv::COLOR_BGR2YCrCb);
    cv::imwrite("converted.png", convertedImage);
    cv::imwrite("comparison.png", newImage);
    // imshow( "Display window", convertedImage);                   // Show our image inside it.
    // cv::waitKey(0);
    return  0;
}
