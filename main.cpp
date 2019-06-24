#include <iostream>
#include <typeinfo>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void convertToYCbCr(cv::Mat image) {
    // converts an RGB image to YCbCr
    // cv::Mat: B-G-R
    std::cout << "Converting image to YCbCr color space." << std::endl;
    int i, j;
    for (i = 0; i <= image.cols; i++) {
        // std::cout << "At column " << i << " of total " << image.cols << std::endl;
        for (j = 0; j <= image.rows; j++) {

            // R, G, B values
            auto R = image.at<cv::Vec3b>(j, i)[2];
            auto G = image.at<cv::Vec3b>(j, i)[1];
            auto B = image.at<cv::Vec3b>(j, i)[0];

            // Y'
            image.at<cv::Vec3b>(j,i)[0] = 0.299 * R + 0.587 * G + 0.114 * B + 16;

            // Cb
            image.at<cv::Vec3b>(j,i)[1] = 128 + (-0.169 * R -0.331 * G + 0.5 * B);

            // Cr
            image.at<cv::Vec3b>(j,i)[2] = 128 + (0.5 * R -0.419 * G -0.081 * B);
            // std::cout << "At conversion: Y = " << Y << ", Cb = " << Cb << ", " 
            //    << Cr << std::endl;
        }
    }
    std::cout << "Converting finished." << std::endl;
    return;
}


void applyGaussianBlur(cv::Mat image) {
    std::cout << "Applying Gaussian Blur on image." << std::endl;
    std::cout << "Creating kernel." << std::endl;


    double sigma = 1;
    int W = 5;
    double kernel[W][W];
    double mean = W/2;
    double sum = 0.0; // For accumulating the kernel values
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < W; ++y) {
            kernel[x][y] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0)) )
                / (2 * M_PI * sigma * sigma);

            // Accumulate the kernel values
            sum += kernel[x][y];
        }

    // Normalize the kernel
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < W; ++y)
            kernel[x][y] /= sum;
    std::cout << "Kernel is: " << std::endl;
    for (int a = 0; a < W; a++) {
        for (int b = 0; b < W; b++) {
            std::cout << kernel[a][b] << " ";
        }
        std::cout << std::endl;
    }


    std::cout << "Applying kernel." << std::endl;
    int i, j;
    int radius = W;
    double val1 = 0, val2 = 0, val3 = 0;

    // 1) loop over all pixels
    for (i = 0; i < image.cols; i++) {
        // std::cout << std::endl << "Column " << i << std::endl;
        for (j = 0; j < image.rows; j++) {
            // std::cout << j << ", ";

            // 2) loop over kernel
            int iy;
            int ix;
            // std::cout << "New kernel loop for pixel " << j << "," << i << std::endl;
            for (iy = int(-radius/2); iy <= int(radius/2); iy++) {
                // looping over y value of kernel
                for (ix = int(-radius/2); ix <= int(radius/2); ix++) {
                    // looping over x value of kernel
                    
                    int x = std::min(image.rows, std::max(0, i-ix));
                    int y = std::min(image.cols, std::max(0, j-iy));
                   /* 
                    std::cout << "Applying at positions: " << y << "," << x;
                    std::cout << ". Value: " << int(image.at<cv::Vec3b>(y,x)[0]) << " * "
                        << kernel[ix][iy] << std::endl;
                    */
                    val1 += image.at<cv::Vec3b>(y, x)[0] * kernel[ix][iy];
                    val2 += image.at<cv::Vec3b>(y, x)[1] * kernel[ix][iy];
                    val3 += image.at<cv::Vec3b>(y, x)[2] * kernel[ix][iy];
                    // std::cout << kernel[ix][iy] << ", ";
                } 
                // std::cout << std::endl;
            }

            image.at<cv::Vec3b>(j, i)[0] = val1;
            image.at<cv::Vec3b>(j, i)[1] = val2;
            image.at<cv::Vec3b>(j, i)[2] = val3;
            val1 = val2 = val3 = 0;
            // std::cout << val1 << " " << val2 << " " << val3 << " | " ;
        }
    }
    return;
}

int main(int argc, char *argv[]) {
    if (argc != 2)
    {
        std::cout <<"Usage: ./blur sourceimage" << std::endl;
        return -1;
    }

    // cv::Mat image;
    auto image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::imwrite("original.png", image);

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
    // auto convertedImage = convertToYCbCr(image);
    // cv::imwrite("converted.png", convertedImage);
    applyGaussianBlur(image);
    cv::imwrite("gaussian.png", image);
    // auto gaussianBlurComparison = applyGaussianBlur(image);
    // cv::imwrite("gaussiancomparison.png", gaussianBlurComparison);
    cv::Mat newImage;
    cv::cvtColor(cv::imread("dice.png"), newImage, cv::COLOR_BGR2YCrCb);
    cv::imwrite("comparison.png", newImage);
    std::cout << "Image dimension: " << newImage.dims << std::endl
        << "Image rows: " << newImage.rows << std::endl
        << "Image columns: " << newImage.cols << std::endl
        << "Channels: " << newImage.channels() << std::endl
        << "Size: " << newImage.size() << std::endl
        << "Type: " << newImage.type() << std::endl;
    std::cout << "type of comparison image is: " << typeid(newImage).name() << std::endl;
    // imshow( "Display window", convertedImage);                   // Show our image inside it.
    // cv::waitKey(0);
    return  0;
}
