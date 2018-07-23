#pragma once

#include <cmath>
#include <limits>

namespace utils {
    namespace rotation {
        /**
            calculate dot product of x and y.

            @param x the input vector x.
            @param y the input vector y.
            @return result the output value.
        */
        double dot_product(const double x[3], const double y[3]) {
            return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
        }

        /**
            calculate cross product of x and y.

            @param x the input vector x.
            @param y the input vector y.
            @return result the output vector.
        */
        void cross_product(const double x[3], const double y[3], double result[3]){
            result[0] = x[1] * y[2] - x[2] * y[1];
            result[1] = x[2] * y[0] - x[0] * y[2];
            result[2] = x[0] * y[1] - x[1] * y[0];
        }

        /**
            convert Rodrigues' vector to quaternion.

            @param angle_axis the input Rodrigues' vector.
            @return quaternion the output quaternion.
        */
        void angle_axis_to_quaternion(const double *angle_axis, double *quaternion) {
            const double a0 = angle_axis[0];
            const double a1 = angle_axis[1];
            const double a2 = angle_axis[2];
            const double theta_squared = a0 * a0 + a1 * a1 + a2 * a2;

            if(theta_squared > std::numeric_limits<double>::epsilon()){
                const double theta = sqrt(theta_squared);
                const double half_theta = 1.0/2.0 * theta;
                const double k = sin(half_theta)/theta;

                quaternion[0] = cos(half_theta);
                quaternion[1] = a0 * k;
                quaternion[2] = a1 * k;
                quaternion[3] = a2 * k;
            }
            else{
                const double k(0.5);

                quaternion[0] = 1.0;
                quaternion[1] = a0 * k;
                quaternion[2] = a1 * k;
                quaternion[3] = a2 * k;
            }
        }

        /**
            convert quaternion to Rodrigues' vector.

            @param angle_axis the input quaternion.
            @return quaternion the output Rodrigues' vector.
        */
        void quaternion_to_angle_axis(const double* quaternion, double* angle_axis) {
            const double& q1 = quaternion[1];
            const double& q2 = quaternion[2];
            const double& q3 = quaternion[3];
            const double sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;
            
            // For quaternions representing non-zero rotation, the conversion
            // is numercially stable
            if(sin_squared_theta > std::numeric_limits<double>::epsilon()){
                const double sin_theta = sqrt(sin_squared_theta);
                const double& cos_theta = quaternion[0];
                
                // If cos_theta is negative, theta is greater than pi/2, which
                // means that angle for the angle_axis vector which is 2 * theta
                // would be greater than pi...
                const double two_theta = 2.0 * (
                    (cos_theta < 0.0) ? atan2(-sin_theta, -cos_theta) : atan2(sin_theta, cos_theta)
                );
                const double k = two_theta / sin_theta;
                
                angle_axis[0] = q1 * k;
                angle_axis[1] = q2 * k;
                angle_axis[2] = q3 * k;
            }
            else{
                // For zero rotation, sqrt() will produce NaN in derivative since
                // the argument is zero. By approximating with a Taylor series, 
                // and truncating at one term, the value and first derivatives will be 
                // computed correctly when Jets are used..
                const double k(2.0);
                angle_axis[0] = q1 * k;
                angle_axis[1] = q2 * k;
                angle_axis[2] = q3 * k;
            }
        }

        /**
            rotate point around angle axis specified by Rodrigues vector.

            @param angle_axis the input quaternion.
            @return quaternion the output Rodrigues' vector.
        */
        void rotate_point_around_angle_axis(const double angle_axis[3], const double point[3], double result[3]) {
            // magnitude of rotation:
            const double theta_squared = dot_product(angle_axis, angle_axis);
            
            if (theta_squared > std::numeric_limits<double>::epsilon()) {
                // away from zero, use the rodriguez formula
                // otherwise we get a division by zero
                // 
                //   result = cos_theta*point  +
                //            (1-cos_theta)*(n dot point)*n +
                //            sin_theta * (n x point)
                //
                const double theta = sqrt(theta_squared);
                const double cos_theta = cos(theta);
                const double sin_theta = sin(theta);
                const double theta_inverse = 1.0 / theta;

                // direction:
                const double n[3] = { 
                    angle_axis[0] * theta_inverse,
                    angle_axis[1] * theta_inverse,
                    angle_axis[2] * theta_inverse 
                };

                // dot product part:
                const double n_dot_point = (1.0 - cos_theta) * dot_product(n, point);

                // cross product part:
                double n_cross_point[3];
                cross_product(n, point, n_cross_point);                          

                result[0] = cos_theta * point[0] + n_dot_point * n[0] + sin_theta * n_cross_point[0];
                result[1] = cos_theta * point[1] + n_dot_point * n[1] + sin_theta * n_cross_point[1];
                result[2] = cos_theta * point[2] + n_dot_point * n[2] + sin_theta * n_cross_point[2];
            } else {
                // Near zero, the first order Taylor approximation of the rotation
                // matrix R corresponding to a vector n and angle w is
                //
                //   R = I + hat(n) * sin(theta)
                //
                // But sin_theta ~ theta and theta * n = angle_axis, which gives us
                //
                //  R = I + hat(n)
                //
                // and actually performing multiplication with the point point, gives us
                // R * point = point + w x point.
                //
                // Switching to the Taylor expansion near zero provides meaningful
                // derivatives when evaluated using Jets.
                //
                // Explicitly inlined evaluation of the cross product for
                // performance reasons.
                double n_cross_point[3];
                cross_product(angle_axis, point, n_cross_point); 

                result[0] = point[0] + n_cross_point[0];
                result[1] = point[1] + n_cross_point[1];
                result[2] = point[2] + n_cross_point[2];
            }
        }
    }
}