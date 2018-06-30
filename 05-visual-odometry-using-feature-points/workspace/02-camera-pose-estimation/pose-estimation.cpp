#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <sophus/so3.h>

using namespace Eigen;
using namespace std;

int main(int argc, char **argv) {
    // essential matrix definition:
    Matrix3d E;
    E << -0.020361855052347700, -0.40071100381184450, -0.03324074249824097,
         +0.393927077821636900, -0.03506401846698079, +0.58571103037210150,
         -0.006788487241438284, -0.58154342729156860, -0.01438258684486258;

    // camera pose:
    Matrix3d R;
    Vector3d t;

    // SVD decomposition:
    JacobiSVD<Matrix3d> svd(E, ComputeFullV | ComputeFullU);

    // sigma matrix:
    double scale = (svd.singularValues()[0] + svd.singularValues()[1]) / 2.0;
    DiagonalMatrix<double, 3> sigma(scale, scale, 0.0);

    // rotation matrix:
    Matrix3d Rz_pos = AngleAxisd(+M_PI/2.0, Vector3d(0, 0, 1)).toRotationMatrix(); 
    Matrix3d Rz_neg = AngleAxisd(-M_PI/2.0, Vector3d(0, 0, 1)).toRotationMatrix();

    // pose estimation:
    Matrix3d t_wedge1 = svd.matrixU() * Rz_pos * sigma * svd.matrixU().transpose();
    Matrix3d t_wedge2 = svd.matrixU() * Rz_neg * sigma * svd.matrixU().transpose();

    Matrix3d R1 = svd.matrixU() * Rz_pos * svd.matrixV().transpose();
    Matrix3d R2 = svd.matrixU() * Rz_neg * svd.matrixV().transpose();

    // pose 1:
    cout << "R1 = \n" << R1 << endl;
    cout << "t1 = \n" << +1.0 * Sophus::SO3::vee(t_wedge1) << endl;
    cout << endl;
    // pose 2:
    cout << "R2 = \n" << R2 << endl;
    cout << "t2 = \n" << +1.0 * Sophus::SO3::vee(t_wedge2) << endl;
    cout << endl;
    // pose 3:
    cout << "R3 = \n" << R1 << endl;
    cout << "t3 = \n" << -1.0 * Sophus::SO3::vee(t_wedge1) << endl;
    cout << endl;
    // pose 2:
    cout << "R4 = \n" << R2 << endl;
    cout << "t4 = \n" << -1.0 * Sophus::SO3::vee(t_wedge2) << endl;
    cout << endl;

    // check t^R=E up to scale
    Matrix3d tR = t_wedge1 * R1;
    cout << "t^R = \n" << tR << endl;

    return 0;
}