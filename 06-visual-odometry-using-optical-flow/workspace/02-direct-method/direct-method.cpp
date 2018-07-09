#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline:
double baseline = 0.573;
// default input file paths:
string left_file = "./left.png";
string disparity_file = "./disparity.png";
boost::format fmt_others("./%06d.png");    // other files

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3 &T21
);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3 &T21
);

// bilinear interpolation
inline double GetPixelValue(const cv::Mat &img, double x, double y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    double xx = x - floor(x);
    double yy = y - floor(y);
    return double(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

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
    // reference images:
    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 1000;
    int boarder = 40;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        // don't pick pixels close to boarder
        int x = rng.uniform(boarder, left_img.cols - boarder);  
        int y = rng.uniform(boarder, left_img.rows - boarder);
        // convert disparity to depth
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; 
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01~05.png's pose using this information
    Sophus::SE3 T_cur_ref;

    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        //DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }
}

void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3 &T21
) {
    // parameters
    const int HALF_PATCH_SIZE = 4;
    const int ITERATIONS = 100;
    
    Eigen::Matrix3d K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // image gradients:
    cv::Mat grad_x, grad_y;
    Scharr(img2, grad_x, CV_64F, 1, 0);
    Scharr(img2, grad_y, CV_64F, 0, 1);

    double cost = 0, lastCost = 0;
    int nGood = 0;  // good projections
    VecVector2d goodProjection;

    for (int iter = 0; iter < ITERATIONS; iter++) {
        nGood = 0;
        goodProjection.clear();

        // rigid transform defined by SE3:
        auto R = T21.rotation_matrix();
        auto t = T21.translation();

        // Define Hessian and bias
        Matrix6d H = Matrix6d::Zero();  // 6x6 Hessian
        Vector6d b = Vector6d::Zero();  // 6x1 bias

        for (size_t i = 0; i < px_ref.size(); i++) {
            //
            // compute the projection in the second image
            //
            // point in reference camera image plane:
            const auto &p_pixel = px_ref[i];
            // point in world frame:
            Eigen::Vector3d P(
                depth_ref[i] * (p_pixel.x() - cx) / fx, 
                depth_ref[i] * (p_pixel.y() - cy) / fy, 
                depth_ref[i]
            );
            // point in current camera frame:
            auto q_camera = R * P + t;

            // point in current camera image plane:
            auto q_pixel = K * q_camera;
            double u = q_pixel.x() / q_pixel.z();
            double v = q_pixel.y() / q_pixel.z();

            // check world point projection in current camera image plane:
            if (int(u) <= HALF_PATCH_SIZE || int(u) >= img2.cols - HALF_PATCH_SIZE ||
                int(v) <= HALF_PATCH_SIZE || int(v) >= img2.rows - HALF_PATCH_SIZE) {
                // skip outside projection:
                continue;
            } else {
                // update good projection stats:
                nGood++;
                goodProjection.push_back(Eigen::Vector2d(u, v));
            }

            // se3 gradients:
            Matrix26d J_pixel_xi;
            double X_prime = q_camera.x();
            double Y_prime = q_camera.y();
            double Z_prime = q_camera.z();
            J_pixel_xi << \
                                              fx/Z_prime,                                           0.0, \
                           -fx*X_prime/(Z_prime*Z_prime),         -fx*X_prime*Y_prime/(Z_prime*Z_prime), \
            fx*(1 + (X_prime*X_prime)/(Z_prime*Z_prime)),                           -fx*Y_prime/Z_prime, \
                                                     0.0,                                    fy/Z_prime, \
                           -fy*Y_prime/(Z_prime*Z_prime), -fy*(1 + (Y_prime*Y_prime)/(Z_prime*Z_prime)), \
                  fy*(X_prime*Y_prime)/(Z_prime*Z_prime),                            fy*X_prime/Z_prime
            ;

            // compute cost and Jacobian:
            for (int x = -HALF_PATCH_SIZE; x < HALF_PATCH_SIZE; x++) {
                for (int y = -HALF_PATCH_SIZE; y < HALF_PATCH_SIZE; y++) {
                    // pixel error:
                    double error = (
                        GetPixelValue(img1, p_pixel.x() + x, p_pixel.y() + y) -
                        GetPixelValue(img2, u + x, v + y)
                    );

                    // image gradients
                    Eigen::Vector2d J_img_pixel(
                        GetGradValue(grad_x, u + x, v + y),
                        GetGradValue(grad_y, u + y, v + y)
                    );
                    J_img_pixel /= 26.0;    

                    // total jacobian
                    Vector6d J = -J_img_pixel.transpose() * J_pixel_xi;

                    H += J * J.transpose();
                    b += -error * J;
                    cost += error * error;
                }
            }
        }

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3::exp(update) * T21;

        cost /= nGood;

        if (isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            // cost increased and it's time to stop iteration
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        lastCost = cost;
        cout << "cost = " << cost << ", good = " << nGood << " at " << iter << endl;
    }
    cout << "good projection: " << nGood << endl;
    cout << "T21 = \n" << T21.matrix() << endl;

    // in order to help you debug, we plot the projected pixels here
    cv::Mat img1_show, img2_show;
    cv::cvtColor(img1, img1_show, CV_GRAY2BGR);
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    for (auto &px: px_ref) {
        cv::rectangle(img1_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
                      cv::Scalar(0, 250, 0));
    }
    for (auto &px: goodProjection) {
        cv::rectangle(img2_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
                      cv::Scalar(0, 250, 0));
    }
    /*
    cv::imshow("reference", img1_show);
    cv::imshow("current", img2_show);
    cv::waitKey();
    */
}

void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3 &T21
) {
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    pyr1.push_back(img1);
    pyr2.push_back(img2);
    for (int level = 0; level < pyramids; level++) {
        cv::Mat down1, down2;
        cv::pyrDown(pyr1[level], down1, cv::Size(pyramid_scale*pyr1[level].cols, pyramid_scale*pyr1[level].rows));
        cv::pyrDown(pyr2[level], down2, cv::Size(pyramid_scale*pyr2[level].cols, pyramid_scale*pyr2[level].rows));

        pyr1.push_back(down1);
        pyr2.push_back(down2);        
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; --level) {
        // set the keypoints in this pyramid level:
        VecVector2d px_ref_pyr; 
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale:
        fx = fxG * scales[level]; fy = fyG * scales[level];
        cx = cxG * scales[level]; cy = cyG * scales[level]; 

        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }
    double fx = fxG, fy = fyG, cx = cxG, cy = cyG;  // restore the old values
}
