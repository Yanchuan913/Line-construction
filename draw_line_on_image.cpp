#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
using namespace std;
using namespace cv;


std::vector<std::vector<double >> readMatrixFile(const char* fileName) {
    std::vector<std::vector<double>> matrixALL{};
    int row = 0;
    std::ifstream fileStream;
    std::string tmp;
    int count = 0;// 行数计数器
    fileStream.open(fileName, std::ios::in);//ios::in 表示以只读的方式读取文件

    if (fileStream.fail())//文件打开失败:返回0
    {
        cout << "fail to open file" << endl;
    }
    else//文件存在
    {
        while (getline(fileStream, tmp, '\n'))//读取一行
        {
            //std::cout <<"tmp" << tmp << std::endl;
            row = 4;
            std::vector<double > tmpV{};
            std::istringstream is(tmp);
            for (int i = 0; i < row; i++) {
                std::string str_tmp;
                is >> str_tmp;
                tmpV.push_back(std::stod(str_tmp));
            }
            matrixALL.push_back(tmpV);
            count++;
        }
        fileStream.close();
    }
    return matrixALL;
}



int main() {
    std::vector<std::vector<double>> matrixALL = readMatrixFile("C:\\Users\\y50018302\\Downloads\\tea_room\\output\\sold_line_detect0.txt");
    Mat img_raw = imread("C:\\Users\\y50018302\\Downloads\\tea_room\\frames\\000000000001.jpg");
    if (img_raw.empty())
    {
        return -1;
    }
    cout << 0 << endl;
    //namedWindow("Example1", WINDOW_AUTOSIZE);

    cv::Point p1, p2;
    cout << matrixALL[0].size() << endl;
    for (auto it : matrixALL)
     //auto it = matrixALL[0];
    {
        p1.x = it[1];
        p1.y = it[0];
        p2.x = it[3];
        p2.y = it[2];
        cout << p1 << p2 << endl;
        //cv::imshow("img_raw", img_raw);
        cv::line(img_raw, p1, p2, Scalar(0, 0, 255), 2);
    }




    cv::imshow("img_raw", img_raw);

    //cout << 1 << endl;
    waitKey(0);

    return 0;
}
