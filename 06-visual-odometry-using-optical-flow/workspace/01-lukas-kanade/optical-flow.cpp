#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace std;
using namespace cv;

// default input image paths:
string file_1 = "./1.png";  // first image
string file_2 = "./2.png";  // second image

/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return
 */
inline double GetPixelValue(const cv::Mat &img, double x, double y) {
    uchar i11 = img.at<uchar>(int(y) + 0, int(x) + 0);
    uchar i12 = img.at<uchar>(int(y) + 1, int(x) + 0);
    uchar i21 = img.at<uchar>(int(y) + 0, int(x) + 1);
    uchar i22 = img.at<uchar>(int(y) + 1, int(x) + 1);

    double xx = x - floor(x);
    double yy = y - floor(y);
    
    return double(
        (1.0 - xx)*(1.0 - yy)*i11 +
        (0.0 + xx)*(1.0 - yy)*i21 + 
        (1.0 - xx)*(0.0 + yy)*i12 + 
        (0.0 + xx)*(0.0 + yy)*i22
    );
    /*
    uchar *data = &img.data[int(y) * img.step + int(x)];
    double xx = x - floor(x);
    double yy = y - floor(y);
    return double(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
    */
}

/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return
 */
inline double GetGradValue(const cv::Mat &grad, double x, double y) {
    double g11 = grad.at<double>(int(y) + 0, int(x) + 0);
    double g12 = grad.at<double>(int(y) + 1, int(x) + 0);
    double g21 = grad.at<double>(int(y) + 0, int(x) + 1);
    double g22 = grad.at<double>(int(y) + 1, int(x) + 1);

    double xx = x - floor(x);
    double yy = y - floor(y);

    return double(
        (1.0 - xx)*(1.0 - yy)*g11 +
        (0.0 + xx)*(1.0 - yy)*g21 + 
        (1.0 - xx)*(0.0 + yy)*g12 + 
        (0.0 + xx)*(0.0 + yy)*g22
    );
}


int main(int argc, char **argv) {
    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(img1, kp1);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    vector<KeyPoint> kp2_single_forward, kp2_single_inverse;
    vector<bool> success_single_forward, success_single_inverse;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single_forward, success_single_forward, false);
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single_inverse, success_single_inverse, true);

    Mat img2_single_forward, img2_single_inverse;
    cv::cvtColor(img2, img2_single_forward, CV_GRAY2BGR);
    cv::cvtColor(img2, img2_single_inverse, CV_GRAY2BGR);
    for (int i = 0; i < kp2_single_forward.size(); i++) {
        if (success_single_forward[i]) {
            cv::circle(img2_single_forward, kp2_single_forward[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single_forward, kp1[i].pt, kp2_single_forward[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    for (int i = 0; i < kp2_single_inverse.size(); i++) {
        if (success_single_inverse[i]) {
            cv::circle(img2_single_inverse, kp2_single_inverse[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single_inverse, kp1[i].pt, kp2_single_inverse[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("tracked single level -- forward", img2_single_forward);
    cv::imwrite("single-level--forward.jpg", img2_single_forward);
    cv::imshow("tracked single level -- inverse", img2_single_inverse);
    cv::imwrite("single-level--inverse.jpg", img2_single_inverse);

    // then test multi-level LK
    vector<KeyPoint> kp2_multi_forward, kp2_multi_inverse;
    vector<bool> success_multi_forward, success_multi_inverse;
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi_forward, success_multi_forward, false);
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi_inverse, success_multi_inverse, true);

    Mat img2_multi_forward, img2_multi_inverse;
    cv::cvtColor(img2, img2_multi_forward, CV_GRAY2BGR);
    cv::cvtColor(img2, img2_multi_inverse, CV_GRAY2BGR);
    for (int i = 0; i < kp2_multi_forward.size(); i++) {
        if (success_multi_forward[i]) {
            cv::circle(img2_multi_forward, kp2_multi_forward[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi_forward, kp1[i].pt, kp2_multi_forward[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    for (int i = 0; i < kp2_multi_inverse.size(); i++) {
        if (success_multi_inverse[i]) {
            cv::circle(img2_multi_inverse, kp2_multi_inverse[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi_inverse, kp1[i].pt, kp2_multi_inverse[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("tracked multi level -- forward", img2_multi_forward);
    cv::imwrite("multi-level--forward.jpg", img2_multi_forward);
    cv::imshow("tracked multi level -- inverse", img2_multi_inverse);
    cv::imwrite("multi-level--inverse.jpg", img2_multi_inverse);

    // use opencv's flow for validation
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error, cv::Size(8, 8));

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("tracked by opencv", img2_CV);
    cv::imwrite("opencv.jpg", img2_CV);

    cv::waitKey(0);

    return 0;
}

/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse
) {
    // image patch params:
    const int HALF_PATCH_SIZE = 4;
    const int N = 4 * HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    // gauss-newton max iteration:
    const int ITERATIONS = 10;

    // image gradients:
    Mat grad_x, grad_y;

    vector<bool> is_hessian_initialized(kp1.size());
    vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> hessian(kp1.size());
    if (inverse) {
        Scharr(img1, grad_x, CV_64F, 1, 0);
        Scharr(img1, grad_y, CV_64F, 0, 1);
    } else {
        Scharr(img2, grad_x, CV_64F, 1, 0);
        Scharr(img2, grad_y, CV_64F, 0, 1);
    }

    bool have_initial = !kp2.empty();

    for (size_t i = 0; i < kp1.size(); i++) {
        // target key point:
        auto kp = kp1[i];

        // dx,dy need to be estimated
        double dx = 0, dy = 0; 
        if (have_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // init hessian
        if (inverse) {
            is_hessian_initialized[i] = false;
            hessian[i] = Eigen::Matrix2d::Identity();
        }

        // Gauss-Newton iterations:
        for (int iter = 0; iter < ITERATIONS; iter++) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (kp.pt.x + dx <= HALF_PATCH_SIZE || kp.pt.x + dx >= img1.cols - HALF_PATCH_SIZE ||
                kp.pt.y + dy <= HALF_PATCH_SIZE || kp.pt.y + dy >= img1.rows - HALF_PATCH_SIZE) {   // go outside
                succ = false;
                break;
            }

            // compute cost and Jacobian:
            for (int x = -HALF_PATCH_SIZE; x < HALF_PATCH_SIZE; x++)
                for (int y = -HALF_PATCH_SIZE; y < HALF_PATCH_SIZE; y++) {
                    // error:
                    double error = 0;
                    // Jacobian:
                    Eigen::Vector2d J;

                    // image gradient:
                    J = Eigen::Vector2d(
                        GetGradValue(grad_x, kp.pt.x + x + (inverse ? 0.0 : dx), kp.pt.y + y + (inverse ? 0.0 : dy)),
                        GetGradValue(grad_y, kp.pt.x + x + (inverse ? 0.0 : dx), kp.pt.y + y + (inverse ? 0.0 : dy))
                    );
                    // normalization for Scharr:
                    J /= 26.0;

                    if (inverse && is_hessian_initialized[i]) {
                        // inverse method & hessian pre-computed:
                        H = hessian[i];
                    } else {
                        // update Hessian:
                        H += J * J.transpose();
                    }

                    error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) - GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    
                    b += error * J;
                    cost += error * error;
                }

            // cache hessian:
            if (inverse && !is_hessian_initialized[i]) {
                is_hessian_initialized[i] = true;
                hessian[i] = H;
            }

            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                cout << "cost increased: " << cost << ", " << lastCost << " at "<< iter + 1 << endl;
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
        }

        success.push_back(succ);

        // set kp2
        if (have_initial) {
            kp2[i].pt = kp.pt + Point2f(dx, dy);
        } else {
            KeyPoint tracked = kp;
            tracked.pt += cv::Point2f(dx, dy);
            kp2.push_back(tracked);
        }
    }
}

void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse) 
{
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<Mat> pyr1, pyr2; // image pyramids
    pyr1.push_back(img1);
    pyr2.push_back(img2);
    for (int level = 0; level < pyramids; level++) {
        Mat down1, down2;
        pyrDown(pyr1[level], down1, Size(pyramid_scale*pyr1[level].cols, pyramid_scale*pyr1[level].rows));
        pyrDown(pyr2[level], down2, Size(pyramid_scale*pyr2[level].cols, pyramid_scale*pyr2[level].rows));

        pyr1.push_back(down1);
        pyr2.push_back(down2);        
    }

    // coarse-to-fine LK tracking in pyramids
    for (int level = pyramids - 1; level >= 0; --level) {
        // downscale ratio:
        double ratio = pow(pyramid_scale, level);

        // downscale template keypoints:
        vector<KeyPoint> kp1_scaled;
        for (int j = 0; j < kp1.size(); ++j) {
            auto kp = kp1[j];
            kp.pt.x *= ratio;
            kp.pt.y *= ratio;
            kp1_scaled.push_back(kp);
        }

        // track at current scale:
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_scaled, kp2, success, inverse);

        // only keep result for original scale:
        if (level > 0) {
            for (int j = 0; j < kp2.size(); ++j) {
                kp2[j].pt.x *= 1.0 / pyramid_scale;
                kp2[j].pt.y *= 1.0 / pyramid_scale;
            }
            success.clear();
        }
    }
}
