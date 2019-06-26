#include <iostream>
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

void convertToYCbCr_omp(cv::Mat image) {
    // converts an RGB image to YCbCr
    // cv::Mat: B-G-R
    std::cout << "Converting image to YCbCr color space using OpenMP." << std::endl;
    std::cout << "Number of processors: " << omp_get_num_procs() << std::endl;
    omp_set_num_threads(omp_get_num_procs());
    int i, j;

    #pragma omp parallel for shared(image) private(i,j)
    for (i = 0; i <= image.cols; i++) {
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
        }
    }

    std::cout << "Converting finished using OpenMP." << std::endl;
    return;
}
cv::Mat applyGaussianBlur(cv::Mat image) {
    std::cout << "Applying Gaussian Blur on image." << std::endl;
    std::cout << "Creating kernel." << std::endl;
    cv::Mat returnImage = image;


    double sigma = 1;
    int W = 5;
    double kernel[W][W];
    double mean = W/3;
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
    /*std::cout << "Kernel is: " << std::endl;
      for (int a = 0; a < W; a++) {
      for (int b = 0; b < W; b++) {
      std::cout << kernel[a][b] << " ";
      }
      std::cout << std::endl;
      }
      */



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

            for (iy = 0; iy < radius; iy++) {
                // looping over y value of kernel
                for (ix = 0; ix < radius; ix++) {
                    // looping over x value of kernel

                    // values for multiplication 
                    int x = std::min(image.cols, std::max(0, i-ix));
                    int y = std::min(image.rows, std::max(0, j-iy));


                    // std::cout << "Applying at positions: " << y << "," << x;
                    /*std::cout << ". Value: " << image.at<cv::Vec3b>(y,x) << " * "
                      << kernel[ix][iy] << std::endl;

                      char str[20];
                      std::scanf("%s", str);

*/
                    val1 += (image.at<cv::Vec3b>(y, x)[0] * kernel[ix][iy]);
                    val2 += (image.at<cv::Vec3b>(y, x)[1] * kernel[ix][iy]);
                    val3 += (image.at<cv::Vec3b>(y, x)[2] * kernel[ix][iy]);
                    //  std::cout << val1 << ", " << val2 << ", " << val3 << std::endl;
                } 
                // std::cout << std::endl;
            }

            returnImage.at<cv::Vec3b>(j, i)[0] = int(val1);
            returnImage.at<cv::Vec3b>(j, i)[1] = int(val2);
            returnImage.at<cv::Vec3b>(j, i)[2] = int(val3);
            // std::cout << "Finished! " << val1 << " " << val2 << " " << val3 << " | " ;
            val1 = val2 = val3 = 0;
        }
    }
    return returnImage;
}

cv::Mat applyGaussianBlur_omp(cv::Mat image) {
    std::cout << "Applying Gaussian Blur on image." << std::endl;
    std::cout << "Creating kernel." << std::endl;
    cv::Mat returnImage = image;


    double sigma = 1;
    int W = 5;
    double kernel[W][W];
    double mean = W/3;
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
    /*std::cout << "Kernel is: " << std::endl;
      for (int a = 0; a < W; a++) {
      for (int b = 0; b < W; b++) {
      std::cout << kernel[a][b] << " ";
      }
      std::cout << std::endl;
      }

*/

    omp_set_num_threads(omp_get_num_procs());

    std::cout << "Applying kernel." << std::endl;
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

            for (iy = 0; iy < radius; iy++) {
                // looping over y value of kernel
                for (ix = 0; ix < radius; ix++) {
                    // looping over x value of kernel

                    // values for multiplication 
                    int x = std::min(image.cols, std::max(0, i-ix));
                    int y = std::min(image.rows, std::max(0, j-iy));


                    // std::cout << "Applying at positions: " << y << "," << x;
                    //std::cout << ". Value: " << image.at<cv::Vec3b>(y,x) << " * "
                    //  << kernel[ix][iy] << std::endl;

                    //  char str[20];
                    //  std::scanf("%s", str);


                    val1 += (image.at<cv::Vec3b>(y, x)[0] * kernel[ix][iy]);
                    val2 += (image.at<cv::Vec3b>(y, x)[1] * kernel[ix][iy]);
                    val3 += (image.at<cv::Vec3b>(y, x)[2] * kernel[ix][iy]);
                    //  std::cout << val1 << ", " << val2 << ", " << val3 << std::endl;
                } 
                // std::cout << std::endl;
            }

            returnImage.at<cv::Vec3b>(j, i)[0] = int(val1);
            returnImage.at<cv::Vec3b>(j, i)[1] = int(val2);
            returnImage.at<cv::Vec3b>(j, i)[2] = int(val3);
            // std::cout << "Finished! " << val1 << " " << val2 << " " << val3 << " | " ;
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
    double duration, startTime;
    startTime = omp_get_wtime();
    std::clock_t start;
    // loop over array
    std::cout << "TESTS STARTING\nFirst test: BGR -> YCbCr without parallelization. Starting timer..." 
        << std::endl;
    start = std::clock();
    convertToYCbCr(image);
    duration = omp_get_wtime() - startTime;
    std::cout << "\nFINISHED. Conversion without OpenMP took " << duration << " seconds.\n" << std::endl;
    cv::imwrite("converted.png", image);
    imshow( "Display window", image);                   // Show our image inside it.
    cv::waitKey(0);

    image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    std::cout << "Second test: BGR -> YCbCr WITH parallelization. Starting timer..." 
        << std::endl;
    start = std::clock();
    startTime = omp_get_wtime();
    convertToYCbCr_omp(image);
    duration = omp_get_wtime() - startTime;
    std::cout << "\nFINISHED. Conversion WITH OpenMP took " << duration << " seconds.\n" << std::endl;
    imshow( "Display window", image);                   // Show our image inside it.
    cv::imwrite("converted_omp.png", image);
    cv::waitKey(0);

    image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    std::cout << "Third test: Gaussian blur without parallelization. Starting timer..."
        << std::endl;
    start = std::clock();
    startTime = omp_get_wtime();
    auto gaussianImage = applyGaussianBlur(image);
    duration = omp_get_wtime() - startTime;
    std::cout << "\nFINISHED. Blurring without OpenMP took " << duration << " seconds.\n" << std::endl;
    imshow( "Display window", gaussianImage);                   // Show our image inside it.
    cv::imwrite("gaussian.png", gaussianImage);
    cv::waitKey(0);

    image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    std::cout << "Fourth test: Gaussian blur WITH parallelization. Starting timer..."
        << std::endl;
    startTime = omp_get_wtime();
    gaussianImage = applyGaussianBlur_omp(image);
    duration = omp_get_wtime() - startTime;
    std::cout << "\nFINISHED. Blurring with OpenMP took " << duration << " seconds.\n" << std::endl;
    imshow( "Display window", gaussianImage);                   // Show our image inside it.
    cv::imwrite("gaussian_omp.png", gaussianImage);
    cv::waitKey(0);

    image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    std::cout << "Fifth test: BGR->YCbCr blur with OpenCV. Starting timer..."
        << std::endl;
    startTime = omp_get_wtime();
    cv::cvtColor(image, image, cv::COLOR_BGR2YCrCb);
    duration = omp_get_wtime() - startTime;
    std::cout << "\nFINISHED. Conversion with OpenMP took " << duration << " seconds.\n" << std::endl;

    image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    std::cout << "Sixth test: Gaussian blur with OpenCV. Starting timer..."
        << std::endl;
    startTime = omp_get_wtime();
    cv::GaussianBlur(image, image, cv::Size(5,5), 1 );
    duration = omp_get_wtime() - startTime;
    std::cout << "\nFINISHED. Gaussian blur with OpenCV took " << duration << " seconds.\n" << std::endl;
    return  0;

}
