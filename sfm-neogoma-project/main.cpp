#define CERES_FOUND 1
#include <opencv2/sfm.hpp>
//#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>

//For use imread
#include <opencv2/opencv.hpp>

//For reading all files in directory:
#include <filesystem>
namespace fs = std::filesystem;

using namespace std;
using namespace cv;
using namespace cv::sfm;
using namespace std::chrono;

static void help() {
 cout
     << "\n------------------------------------------------------------------------------------\n"
     << " This program shows the multiview reconstruction capabilities in the \n"
     << " OpenCV Structure From Motion (SFM) module.\n"
     << " It reconstruct a scene from a set of 2D images \n"
     << " Usage:\n"
     << "        sfm-neogoma <path_to_folder> <extension> <elem_firs> <elem_last> <f> <cx> <cy> <withColor>\n"
     << " where: path_to_folder is the file absolute path into your system which contains\n"
     << "        a folder with the images to use for reconstruction. \n"
     << "        ext  is the extension of the images without the dot '.', eg: jpg or png \n"
     << "        elem_first is the first image of the dataset (starting at 1) to consider. \n"
     << "        elem_last is the last image of the dataset to consider. \n"
     << "        f  is the focal length in pixels. \n"
     << "        cx is the image principal point x coordinates in pixels. \n"
     << "        cy is the image principal point y coordinates in pixels. \n"
     << "        withColor, if 0 don't compute the color of the points, if different (eg: 1) it does. \n"
     << "------------------------------------------------------------------------------------\n\n"
     << endl;
}

//Function to read images from the path
static int getdir(const string _filename, vector<String> &files)
{
 ifstream myfile(_filename.c_str());
 if (!myfile.is_open()) {
   cout << "Unable to read file: " << _filename << endl;
   exit(0);
 } else {;
   size_t found = _filename.find_last_of("/\\");
   string line_str, path_to_file = _filename.substr(0, found);
   while ( getline(myfile, line_str) )
     files.push_back(path_to_file+string("/")+line_str);
 }
 return 1;
}

//Functions to write (init and add element) to point cloud files; formats: ply, csv and xyz
void ply_init(std::ofstream &myfile, int vertex_count, bool withColor)
{
   myfile << "ply\n";
   myfile << "format ascii 1.0\n";
   myfile << "comment This point cloud has been generated with the software sfm-negoma." << "\n";
   myfile << "element vertex " + to_string(vertex_count) + "\n";
   myfile << "property float x\n";
   myfile << "property float y\n";
   myfile << "property float z\n";
   if (withColor)
   {
       myfile << "property uchar blue\n";
       myfile << "property uchar green\n";
       myfile << "property uchar red\n";
   }
   myfile << "end_header\n";
}

void csv_init(std::ofstream &myfile, bool withColor)
{
   if (withColor)
   {
       myfile << "x,y,z,b,g,r\n";
   }
   else
   {
       myfile << "x,y,z\n";
   }
}

void ply_add(std::ofstream &myfile, Vec3f tempPoint)
{
   myfile << to_string(tempPoint[0]) + " " + to_string(tempPoint[1]) + " " + to_string(tempPoint[2]) + "\n";
}

void ply_add(std::ofstream &myfile, Vec3f tempPoint, Vec3b tempColor)
{
   myfile << to_string(tempPoint[0]) + " " + to_string(tempPoint[1]) + " " + to_string(tempPoint[2]) + " " + to_string(tempColor[0])  + " " + to_string(tempColor[1])  + " " + to_string(tempColor[2]) + "\n";
}

void csv_add(std::ofstream &myfile, Vec3f tempPoint)
{
   myfile << to_string(tempPoint[0]) + "," + to_string(tempPoint[1]) + "," + to_string(tempPoint[2]) + "\n";
}

void csv_add(std::ofstream &myfile, Vec3f tempPoint, Vec3b tempColor)
{
   myfile << to_string(tempPoint[0]) + "," + to_string(tempPoint[1]) + "," + to_string(tempPoint[2]) + "," + to_string(tempColor[0])  + "," + to_string(tempColor[1])  + "," + to_string(tempColor[2]) + "\n";
}

//Path sort function
bool PathSort(const fs::path &first, const fs::path &second)
{
   return first.filename().string() < second.filename().string();
}

//Main
int main(int argc, char* argv[])
{
   //Start measuring time
   auto start = high_resolution_clock::now();

 // Read input parameters
 if ( argc != 9 )
 {
   help();
   exit(0);
 }
 // Parse the image paths
 string string_folder(argv[1]);
 std::string ext("." + string(argv[2]));

 //Get all the files in the directory with the extension
 vector<fs::path> list_paths;
 for (auto &p : fs::recursive_directory_iterator(string_folder))
 {
     if (p.path().extension() == ext)
         list_paths.push_back(p.path());
 }

 //Sort the list by alphabetical order using our path comparing function
 std::sort(list_paths.begin(), list_paths.end(), PathSort);

 //Write the first files within the specified subrange
 string string_file = string_folder + "/image_paths_file.txt";
 std::ofstream myfile_txt;
 myfile_txt.open (string_file);
 int elem_first = atoi(argv[3])-1;
 int elem_last = atoi(argv[4])-1;
 elem_last = elem_last < (list_paths.size()-1)? elem_last : list_paths.size()-1;
 for (int iii = elem_first; iii <= elem_last; iii++)
 {
     myfile_txt << list_paths[iii].stem().string() << ext << '\n';
 }

 myfile_txt.close();

 //return 0;

 vector<String> images_paths;
 getdir(string_file, images_paths );

 //Load first image to estimate camera params
 Mat image1 = imread(images_paths[0],IMREAD_COLOR);
 int width = image1.cols;
 int height = image1.rows;

 // Read f, cx, cy
 float f  = atof(argv[5]),
       cx = atof(argv[6]), cy = atof(argv[7]);

 //Estimate them if they are 0
 if (cx == 0)
     cx = width/2;

 if (cy == 0)
     cy = height/2;

 if (f == 0)
     f = 800;

 // Build intrinsics
 Matx33d K = Matx33d( f, 0, cx,
                      0, f, cy,
                      0, 0,  1);
 bool is_projective = true;
 vector<Mat> Rs_est, ts_est, points3d_estimated;
 reconstruct(images_paths, Rs_est, ts_est, K, points3d_estimated, is_projective);

 // Print output
 cout << "\n----------------------------\n" << endl;
 cout << "Reconstruction: " << endl;
 cout << "============================" << endl;
 cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
 cout << "Estimated cameras: " << Rs_est.size() << endl;
 cout << "Refined intrinsics: " << endl << K << endl << endl;
 cout << "Writing 3D Point Cloud to files: " << endl;
 cout << "============================" << endl;

 // Create the pointcloud
 cout << "Recovering points  ... ";
 // recover estimated points3d

 bool withColor = !(argv[8] == 0);

 //Init PLY and CSV files
 std::ofstream myfile_ply;
 myfile_ply.open (string_folder + "/point_cloud.ply");
 ply_init(myfile_ply, points3d_estimated.size(), withColor);

 std::ofstream myfile_csv;
 myfile_csv.open (string_folder + "/point_cloud.csv");
 csv_init(myfile_csv, withColor);

 //std::ofstream myfile_xyz;
 //myfile_xyz.open (string_folder + "/point_cloud.xyz");

 // The computed 3d points are in world coordinates

 //Write to files the 3D points (and compute the color of each one and write also the color)
 ///vector<Vec3f> point_cloud_est;
 ///vector<Vec3b> point_cloud_est_color;
 for (int i = 0; i < points3d_estimated.size(); ++i)
 {
   Vec3f point_3dworld = Vec3f(points3d_estimated[i]);
   ///point_cloud_est.push_back(point_3dworld);

   if (!withColor)
   {
       ply_add(myfile_ply, point_3dworld);
       csv_add(myfile_csv, point_3dworld);
       continue;
   }

   //To get the color of the point

   // - Find the camera position in 3d world nearest to the point

   // - Camera position is computed taking into account the relation between a point in camera and world coordinates:
   //   point_camera = R * point_world + t
   //   R.inv() * ( point_camera - t ) = point_world
   //   The position of the camera is point_camera = (0,0,0)
   //   point_camera_world = - R.inv() * t

   float minimumDistance = INFINITY;
   int minimumDistance_idx = -1;
   for (size_t jj = 0; jj < Rs_est.size(); ++jj)
   {
      Mat cameraPosition_3dworld_Mat = -Rs_est[jj].inv()*ts_est[jj];
      Vec3f cameraPosition_3dworld = Vec3f(cameraPosition_3dworld_Mat);
      float distance = cv::norm(cameraPosition_3dworld - point_3dworld);
      if (distance < minimumDistance){
          minimumDistance_idx = jj;
      }
   }

   // - Project the 3d point to a 2d point in that camera (using projectPoints)
   Mat t_nearest = ts_est[minimumDistance_idx];
   Mat R_nearest = Rs_est[minimumDistance_idx];
   Mat rvec_nearest;
   Rodrigues(R_nearest,rvec_nearest);

   vector<Point3f> inputPoints = {point_3dworld};
   vector<Point2f> outputPoints;
   Mat distCoeffs;
   projectPoints(inputPoints, rvec_nearest, t_nearest, K, distCoeffs, outputPoints);

   Point2f point_2dimage = outputPoints[0];

   // - Get the coordinate in the image and access the color of the pixel
   Mat image_nearest = imread(images_paths[minimumDistance_idx], IMREAD_COLOR);

   // - Point should be inside the image, but for security redundance:
   Vec3b tempColor;
   if (point_2dimage.x>0 && point_2dimage.x<image_nearest.cols && point_2dimage.y>0 && point_2dimage.y<image_nearest.rows)
   {
       tempColor = image_nearest.at<Vec3b>(Point(point_2dimage.x,point_2dimage.y));
   }
   else
   {
       tempColor = {0,0,0};
   }

   ///point_cloud_est_color.push_back(tempColor);

   //Write to files
   ply_add(myfile_ply, point_3dworld, tempColor);
   csv_add(myfile_csv, point_3dworld, tempColor);
 }

 //Close files
 myfile_ply.close();
 myfile_csv.close();

 //Print end
 if ( points3d_estimated.size() > 0 )
 {
   cout << "Point cloud has " << points3d_estimated.size() << " points";
 }
 else
 {
   cout << "Empty pointcloud" << endl;
 }

 cout << endl << "[DONE] Program finished successfully" << endl;

 //Stop and measure time
 auto stop = high_resolution_clock::now();
 auto duration = stop - start;

 //Display time
 auto hh = duration_cast<hours>(duration);
 duration -= hh;
 auto mm = duration_cast<minutes>(duration);
 duration -= mm;
 auto ss = duration_cast<seconds>(duration);
 cout << "Time elapsed: "
      << hh.count() << "h: "
      << mm.count() << "m: "
      << ss.count() << "s"
      << endl;

 //Return
 return 0;
}

/*

//#include <QCoreApplication>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
   //QCoreApplication a(argc, argv);

   Mat image;
   char keyPress;

   cout << "Hello\n";

   //OpenCV Code
   image = imread("/home/juanjose/Pictures/screenshot.png", IMREAD_COLOR);
   imshow("Opp",image);
   while(true)
   {
       keyPress = waitKey();
       if (keyPress == 'q'){
           destroyAllWindows();
           break;
       }
   }

   return 0;
   //return a.exec();

}
*/
