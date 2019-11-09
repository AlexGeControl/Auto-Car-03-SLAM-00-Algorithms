#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

using namespace std;

string scan_match_file = "./scan_match.txt";
string odom_file = "./odom.txt";

int main(int argc, char** argv)
{   
    /*
    	groud truth from lidar scan-match: 
	(timestamp, x, y, theta) -- (t_s, s_x, s_y, s_th)
    */
    vector<vector<double>> s_data;
    /*
    	observation: 
     	(timestamp, angular_speed_left, angular_speed_right) -- (t_r, w_L, w_R)
    */
    vector<vector<double>> r_data;

    ifstream fin_s(scan_match_file);
    ifstream fin_r(odom_file);
    if (!fin_s || !fin_r)
    {
        cerr << "请在有scan_match.txt和odom.txt的目录下运行此程序" << endl;
        return 1;
    }

    // load lidar scan-match:
    while (!fin_s.eof()) {
        double s_t, s_x, s_y, s_th;
        fin_s >> s_t >> s_x >> s_y >> s_th;
        s_data.push_back(vector<double>({s_t, s_x, s_y, s_th}));
    }
    fin_s.close();

    // load odometry:
    while (!fin_r.eof()) {
        double t_r, w_L, w_R;
        fin_r >> t_r >> w_L >> w_R;
        r_data.push_back(vector<double>({t_r, w_L, w_R}));
    }
    fin_r.close();

    // a. calculate J_21 & J_22:
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    // init matrix:
    A.conservativeResize(5000, 2);
    b.conservativeResize(5000);
    A.setZero();
    b.setZero();
    // build constraints:
    size_t id_r = 0;
    size_t id_s = 0;
    double last_rt = r_data[0][0];
    double w_Lt = 0;
    double w_Rt = 0;
    while (id_s < 5000)
    {
        // lidat scan-match:
        const double &s_t = s_data[id_s][0];
        const double &s_th = s_data[id_s][3];
        // odometry:
        const double &r_t = r_data[id_r][0];
        const double &w_L = r_data[id_r][1];
        const double &w_R = r_data[id_r][2];
        ++id_r;
        // add constraint:
        if (r_t < s_t)
        {
            double dt = r_t - last_rt;
            w_Lt += w_L * dt;
            w_Rt += w_R * dt;
            last_rt = r_t;
        }
        else
        {
            double dt = s_t - last_rt;
            w_Lt += w_L * dt;
            w_Rt += w_R * dt;
            last_rt = s_t;
	    
            A(id_s, 0) = w_Lt;
	    A(id_s, 1) = w_Rt;
            b(id_s) = s_th;

	    w_Lt = 0;
            w_Rt = 0;
            ++id_s;
        }
    }
    // solve using QR decomposition:
    Eigen::Vector2d J21J22;
    J21J22 = A.fullPivHouseholderQr().solve(b);
    
    const double &J21 = J21J22(0);
    const double &J22 = J21J22(1);

    // b. calculate b:
    Eigen::VectorXd C;
    Eigen::VectorXd S;
    // init matrix:
    C.conservativeResize(10000);
    S.conservativeResize(10000);
    C.setZero();
    S.setZero();
    // build constraints:
    id_r = 0;
    id_s = 0;
    last_rt = r_data[0][0];
    double th = 0;
    double cx = 0;
    double cy = 0;
    while (id_s < 5000)
    {
        // lidar scan-match:
        const double &s_t = s_data[id_s][0];
        const double &s_x = s_data[id_s][1];
        const double &s_y = s_data[id_s][2];
        // odometry:
        const double &r_t = r_data[id_r][0];
        const double &w_L = r_data[id_r][1];
        const double &w_R = r_data[id_r][2];
        ++id_r;
	// add constraints:
        if (r_t < s_t)
        {
            double dt = r_t - last_rt;
            cx += 0.5 * (-J21 * w_L * dt + J22 * w_R * dt) * cos(th);
            cy += 0.5 * (-J21 * w_L * dt + J22 * w_R * dt) * sin(th);
            th += (J21 * w_L + J22 * w_R) * dt;
            last_rt = r_t;
        }
        else
        {
            double dt = s_t - last_rt;
            cx += 0.5 * (-J21 * w_L * dt + J22 * w_R * dt) * cos(th);
            cy += 0.5 * (-J21 * w_L * dt + J22 * w_R * dt) * sin(th);
            th += (J21 * w_L + J22 * w_R) * dt;
            last_rt = s_t;

	    C(id_s << 1) = cx;
            S(id_s << 1) = s_x;
	    C((id_s << 1) + 1) = cy; 
            S((id_s << 1) + 1)= s_y;

            cx = 0;
            cy = 0;
            th = 0;
            ++id_s;
        }
    }
    // solve using QR decomposition:
    double b_wheel;
    b_wheel = C.fullPivHouseholderQr().solve(S)(0);

    double r_L = -b_wheel * J21;
    double r_R = +b_wheel * J22;
    
    cout << "[Solution from YaoGeFAD]:" << endl;
    cout << "\tJ21: " << J21 << endl;
    cout << "\tJ22: " << J22 << endl;
    cout << endl;
    cout << "\tb: " << b_wheel << endl;
    cout << "\tr_L: " << r_L << endl;
    cout << "\tr_R: " << r_R << endl;

    cout << "[参考答案]:" << endl;
    cout << "\t轮间距b为0.6m左右" << endl;
    cout << "\t两轮半径为0.1m左右" << endl;

    return 0;
}
