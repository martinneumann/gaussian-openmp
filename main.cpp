#include <iostream>
#include <fstream>
#include <typeinfo>
#include <math.h>
#include <cstdio>
#include <omp.h>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void convertToYCbCr(cv::Mat image) {
    // converts an RGB image to YCbCr
    int i, j;
    for (i = 0; i < image.cols; i++) {
        for (j = 0; j < image.rows; j++) {

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
        }
    }
    return;
}

void convertToYCbCr_omp(cv::Mat image, int num_procs) {
    // converts an RGB image to YCbCr
    // cv::Mat: B-G-R
    omp_set_num_threads(num_procs);
    int i, j;

#pragma omp parallel for shared(image) private(i,j)
    for (i = 0; i < image.cols; i++) {
        for (j = 0; j < image.rows; j++) {

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
        }
    }

    // std::cout << "Converting finished using OpenMP." << std::endl;
    return;
}


void convertToYCbCr_omp_2(cv::Mat image, int num_procs) {
    // converts an RGB image to YCbCr
    // cv::Mat: B-G-R
    omp_set_num_threads(num_procs);
    int i, j;

    #pragma omp parallel for shared(image) private(i)
    for (i = 0; i < image.cols; i++) {
        #pragma omp parallel for shared(image) private(j)
        for (j = 0; j < image.rows; j++) {

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
        }
    }

    return;
}

cv::Mat applyGaussianBlur(cv::Mat image) {
    cv::Mat returnImage = image;

    double sigma = 1;
    int W = 5;
    double mean = W/2;
    double kernel[W][W];
    double sum = 0.0;
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < W; ++y) {
            kernel[x][y] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0)) )
                / (2 * M_PI * sigma * sigma);
            sum += kernel[x][y];
        }

    for (int x = 0; x < W; ++x)
        for (int y = 0; y < W; ++y)
            kernel[x][y] /= sum;

    int i, j;
    int radius = W;
    double val1 = 0, val2 = 0, val3 = 0;

    // 1) loop over all pixels
    for (i = 0; i < image.cols; i++) {
        for (j = 0; j < image.rows; j++) {

            // 2) loop over kernel
            int iy;
            int ix;
            for (ix = -radius/2 ; ix <= radius/2; ix++) {
                for (iy = -radius/2; iy <= radius/2; iy++) {

                    // values for multiplication 
                    int x = std::min(image.cols, std::max(0, i-ix));
                    int y = std::min(image.rows, std::max(0, j-iy));
                    val1 += (image.at<cv::Vec3b>(y, x)[0] * kernel[iy+2][ix+2]);
                    val2 += (image.at<cv::Vec3b>(y, x)[1] * kernel[iy+2][ix+2]);
                    val3 += (image.at<cv::Vec3b>(y, x)[2] * kernel[iy+2][ix+2]);
                } 
            }

            returnImage.at<cv::Vec3b>(j, i)[0] = int(val1);
            returnImage.at<cv::Vec3b>(j, i)[1] = int(val2);
            returnImage.at<cv::Vec3b>(j, i)[2] = int(val3);
            val1 = val2 = val3 = 0;
        }
    }
    return returnImage;
}


cv::Mat applyGaussianBlur_omp(cv::Mat image, int num_procs) {
    // std::cout << "Applying Gaussian Blur on image." << std::endl;
    // std::cout << "Creating kernel." << std::endl;
    cv::Mat returnImage = image;


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



    omp_set_num_threads(num_procs);

    // std::cout << "Applying kernel." << std::endl;
    int i, j;
    int radius = W;
    double val1 = 0, val2 = 0, val3 = 0;

    #pragma omp parallel for shared(image) private(i,j,val1,val2,val3) 
    for (i = 0; i < image.cols; i++) {
        for (j = 0; j < image.rows; j++) {

            // 2) loop over kernel
            int iy;
            int ix;
            // std::cout << "New kernel loop for pixel " << j << "," << i << std::endl;

            for (iy = -radius/2; iy <= radius/2; iy++) {
                // looping over y value of kernel
                for (ix = -radius/2; ix <= radius/2; ix++) {
                    // looping over x value of kernel

                    // values for multiplication 
                    int x = std::min(image.cols, std::max(0, i-ix));
                    int y = std::min(image.rows, std::max(0, j-iy));

                    val1 += (image.at<cv::Vec3b>(y, x)[0] * kernel[iy+2][ix+2]);
                    val2 += (image.at<cv::Vec3b>(y, x)[1] * kernel[iy+2][ix+2]);
                    val3 += (image.at<cv::Vec3b>(y, x)[2] * kernel[iy+2][ix+2]);
                } 
            }

            returnImage.at<cv::Vec3b>(j, i)[0] = int(val1);
            returnImage.at<cv::Vec3b>(j, i)[1] = int(val2);
            returnImage.at<cv::Vec3b>(j, i)[2] = int(val3);
            val1 = val2 = val3 = 0;
        }
    }
    return returnImage;
}

cv::Mat applyGaussianBlur_omp_2(cv::Mat image, int num_procs) {
    cv::Mat returnImage = image;


    double sigma = 1;
    int W = 5;
    double kernel[W][W];
    double mean = W/2;
    double sum = 0.0; // For accumulating the kernel values
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < W; ++y) {
            kernel[x][y] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0)) )
                / (2 * M_PI * sigma * sigma);

            sum += kernel[x][y];
        }

    // Normalize the kernel
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < W; ++y)
            kernel[x][y] /= sum;

    omp_set_num_threads(num_procs);

    int i, j;
    int radius = W;
    double val1 = 0, val2 = 0, val3 = 0;
    #pragma omp parallel for shared(image) private(i) 
    for (i = 0; i < image.cols; i++) {
        #pragma omp parallel for shared(image) private(j,val1,val2,val3) 
        for (j = 0; j < image.rows; j++) {

            // 2) loop over kernel
            int iy;
            int ix;

            for (iy = -radius/2; iy <= radius/2; iy++) {
                // looping over y value of kernel
                for (ix = -radius/2; ix <= radius/2; ix++) {
                    // looping over x value of kernel

                    // values for multiplication 
                    int x = std::min(image.cols, std::max(0, i-ix));
                    int y = std::min(image.rows, std::max(0, j-iy));

                    val1 += (image.at<cv::Vec3b>(y, x)[0] * kernel[iy+2][ix+2]);
                    val2 += (image.at<cv::Vec3b>(y, x)[1] * kernel[iy+2][ix+2]);
                    val3 += (image.at<cv::Vec3b>(y, x)[2] * kernel[iy+2][ix+2]);
                } 
            }

            returnImage.at<cv::Vec3b>(j, i)[0] = int(val1);
            returnImage.at<cv::Vec3b>(j, i)[1] = int(val2);
            returnImage.at<cv::Vec3b>(j, i)[2] = int(val3);
            val1 = val2 = val3 = 0;
        }
    }
    return returnImage;
}

int main(int argc, char *argv[]) {
    if (argc != 2)
    {
        std::cout <<"Usage: ./blur sourceimage" << std::endl;
        return -1;
    }

    std::ofstream logs;
    logs.open("logfile.txt");


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


    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.


    auto gaussianImage = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

    double duration, startTime;

    logs << "BGR->YCbCr\n";
    std::cout << "TESTS STARTING\nFirst test: BGR -> YCbCr without parallelization. Starting timer..." 
        << std::endl;
    for (int i = 0; i < 10; i++) {
        image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        startTime = omp_get_wtime();
        convertToYCbCr(image);
        duration = omp_get_wtime() - startTime;
        logs << duration << "\n";
        std::cout << "FINISHED. Conversion without OpenMP took " << duration << " seconds." << std::endl;
        // imshow( "Display window", image);                   // Show our image inside it.
        // cv::waitKey(0);
    }
    cv::imwrite("converted_normal.png", image);

    std::cout << "Second test: BGR -> YCbCr WITH parallelization and 4 threads. Starting timer..." 
        << std::endl;
    logs << "BGR->YCbCr w/ OpenMP, simple, 4 kernels\n";
    for (int i = 0; i < 10; i++) {
        image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        startTime = omp_get_wtime();
        convertToYCbCr_omp(image, 4);
        duration = omp_get_wtime() - startTime;
        logs << duration << "\n";
        std::cout << "FINISHED. Conversion WITH OpenMP, simple and 4 threads took " << duration << " seconds." << std::endl;
        // imshow( "Display window", image);                   // Show our image inside it.
        // cv::imwrite("converted_omp.png", image);
        // cv::waitKey(0);
    }
    cv::imwrite("converted_omp_4.png", image);

    std::cout << "Third test: BGR -> YCbCr WITH parallelization and 8 threads. Starting timer..." 
        << std::endl;
    logs << "BGR->YCbCr w/ OpenMP, simple, 8 kernels\n";
    for (int i = 0; i < 10; i++) {
        image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        startTime = omp_get_wtime();
        convertToYCbCr_omp(image, 8);
        duration = omp_get_wtime() - startTime;
        logs << duration << "\n";
        std::cout << "FINISHED. Conversion WITH OpenMP, simple and 8 threads took " << duration << " seconds." << std::endl;
        // imshow( "Display window", image);                   // Show our image inside it.
        // cv::imwrite("converted_omp.png", image);
        // cv::waitKey(0);
    }
    cv::imwrite("converted_omp_8.png", image);

    std::cout << "Fourth test: BGR -> YCbCr WITH \"double\" parallelization and 4 threads. Starting timer..." 
        << std::endl;
    logs << "BGR->YCbCr w/ OpenMP, double, 4 kernels\n";
    for (int i = 0; i < 10; i++) {
        image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        startTime = omp_get_wtime();
        convertToYCbCr_omp_2(image, 4);
        duration = omp_get_wtime() - startTime;
        logs << duration << "\n";
        std::cout << "FINISHED. Conversion WITH OpenMP, double and 4 threads took " << duration << " seconds." << std::endl;
        // imshow( "Display window", image);                   // Show our image inside it.
        // cv::imwrite("converted_omp.png", image);
        // cv::waitKey(0);
    }
    cv::imwrite("converted_omp_double_4.png", image);

    logs << "BGR->YCbCr w/ OpenMP, double, 8 kernels\n";
    std::cout << "Fifth test: BGR -> YCbCr WITH \"double\" parallelization and 8 threads. Starting timer..." 
        << std::endl;
    for (int i = 0; i < 10; i++) {
        image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        startTime = omp_get_wtime();
        convertToYCbCr_omp_2(image, 8);
        duration = omp_get_wtime() - startTime;
        logs << duration << "\n";
        std::cout << "FINISHED. Conversion WITH OpenMP, double, 8 threads took " << duration << " seconds." << std::endl;
        // imshow( "Display window", image);                   // Show our image inside it.
        // cv::imwrite("converted_omp.png", image);
        // cv::waitKey(0);
    }
    cv::imwrite("converted_omp_double_8.png", image);

    logs << "Blur\n";
    std::cout <<"Sixth test: Gaussian blur without parallelization. Starting timer..."
        << std::endl;
    for (int i = 0; i < 10; i++) {
        image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        startTime = omp_get_wtime();
        gaussianImage = applyGaussianBlur(image);
        duration = omp_get_wtime() - startTime;
        logs << duration << "\n";
        std::cout << "FINISHED. Blurring without OpenMP took " << duration << " seconds." << std::endl;
        // imshow( "Display window", gaussianImage);                   // Show our image inside it.
        // cv::waitKey(0);
    }
    cv::imwrite("gaussian_normal.png", gaussianImage);

    logs << "Blur w/ OpenMP, simple, 4 kernels\n";
    std::cout << "Seventh test: Gaussian blur WITH parallelization, simple, 4. Starting timer..."
        << std::endl;
    for (int i = 0; i < 10; i++) {
        image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        startTime = omp_get_wtime();
        gaussianImage = applyGaussianBlur_omp(image, 4);
        duration = omp_get_wtime() - startTime;
        logs << duration << "\n";
        std::cout << "FINISHED. Blurring with OpenMP, simple, 4 took " << duration << " seconds." << std::endl;
        // imshow( "Display window", gaussianImage);                   // Show our image inside it.
        // cv::waitKey(0);
    }
    cv::imwrite("gaussian_omp_4.png", gaussianImage);


    logs << "Blur w/ OpenMP, simple, 8 kernels\n";
    std::cout << "Eigth test: Gaussian blur WITH parallelization, simple, 8. Starting timer..."
        << std::endl;
    for (int i = 0; i < 10; i++) {
        image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        startTime = omp_get_wtime();
        gaussianImage = applyGaussianBlur_omp(image, 8);
        duration = omp_get_wtime() - startTime;
        logs << duration << "\n";
        std::cout << "FINISHED. Blurring with OpenMP, simple, 8 took " << duration << " seconds." << std::endl;
        // imshow( "Display window", gaussianImage);                   // Show our image inside it.
        // cv::waitKey(0);
    }
    cv::imwrite("gaussian_omp_8.png", gaussianImage);

    logs << "Blur w/ OpenMP, double, 4 kernels\n";
    std::cout << "Nineth test: Gaussian blur WITH parallelization, double, 4 threads. Starting timer..."
        << std::endl;
    for (int i = 0; i < 10; i++) {
        image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        startTime = omp_get_wtime();
        gaussianImage = applyGaussianBlur_omp_2(image, 4);
        duration = omp_get_wtime() - startTime;
        logs << duration << "\n";
        std::cout << "FINISHED. Blurring with OpenMP, double, 4 took " << duration << " seconds." << std::endl;
        // imshow( "Display window", gaussianImage);                   // Show our image inside it.
        // cv::waitKey(0);
    }
    cv::imwrite("gaussian_omp_double_4.png", gaussianImage);

    logs << "Blur w/ OpenMP, double, 8 kernels\n";
    std::cout << "Tenth test: Gaussian blur WITH parallelization, double, 8 threads. Starting timer..."
        << std::endl;
    for (int i = 0; i < 10; i++) {
        image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        startTime = omp_get_wtime();
        gaussianImage = applyGaussianBlur_omp_2(image, 8);
        duration = omp_get_wtime() - startTime;
        logs << duration << "\n";
        std::cout << "FINISHED. Blurring with OpenMP, double, 8 took " << duration << " seconds." << std::endl;
        // imshow( "Display window", gaussianImage);                   // Show our image inside it.
        // cv::waitKey(0);
    }
    cv::imwrite("gaussian_omp_double_8.png", gaussianImage);

    std::cout << "Eleventh test: RGB->YCbCr w/ OpenCV."
        << std::endl;
    logs << "RGB->YCbCr w/ OpenCV\n";
    for (int i = 0; i < 10; i++) {
        image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        std::cout << "Fifth test: BGR->YCbCr blur with OpenCV. Starting timer..."
            << std::endl;
        startTime = omp_get_wtime();
        cv::cvtColor(image, image, cv::COLOR_BGR2YCrCb);
        duration = omp_get_wtime() - startTime;
        logs << duration << "\n";
        std::cout << "FINISHED. Conversion with OpenCV took " << duration << " seconds." << std::endl;
    }
    cv::imwrite("ycbcr_opencv.png", image);

    logs << "Gauss w/ OpenCV\n";
    std::cout << "Twelveth test: Gaussian blur w/ OpenCV."
        << std::endl;
    for (int i = 0; i < 10; i++) {
        image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        std::cout << "Sixth test: Gaussian blur with OpenCV. Starting timer..."
            << std::endl;
        startTime = omp_get_wtime();
        cv::GaussianBlur(image, image, cv::Size(5,5), 1 );
        duration = omp_get_wtime() - startTime;
        logs << duration << "\n";
        std::cout << "FINISHED. Gaussian blur with OpenCV took " << duration << " seconds." << std::endl;
    }
    cv::imwrite("gaussian_opencv.png", image);
    logs.close();
    return  0;


}
