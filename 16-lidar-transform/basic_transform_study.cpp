#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

int main(int argc, char** argv)
{
    // robo B's pose in world coordinates O：
    Eigen::Vector3d B(3, 4, M_PI);

    // homogeneous transform matrix from robo B to world O：
    Eigen::Matrix3d TOB;
    TOB << cos(B(2)), -sin(B(2)), B(0),
           sin(B(2)),  cos(B(2)), B(1),
              0,          0,        1;

    // homogeneous transform matrix from world O to robo B:
    Eigen::Matrix3d TBO = TOB.inverse();

    // robo A's pose in world coordinates O：
    Eigen::Vector3d A(1, 3, -M_PI / 2);

    // calculate robo A's pose in robo B's coordinates：
    // a. build trans matrix:
    Eigen::Matrix3d TOA;
    TOA << cos(A(2)), -sin(A(2)), A(0),
           sin(A(2)),  cos(A(2)), A(1),
                   0,          0,    1;	   
    Eigen::Matrix3d TBA = TBO * TOA;
    // b. get pose:
    Eigen::Vector3d BA;
    BA(0) = TBA(0, 2);
    BA(1) = TBA(1, 2);
    BA(2) = atan2(TBA(1,0), TBA(0,0));

    cout << "The right answer is BA: 2 1 1.5708" << endl;
    cout << "Your answer is BA: " << BA.transpose() << endl;

    return 0;
}
