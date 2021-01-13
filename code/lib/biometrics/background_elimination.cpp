#include <iostream>
#include <queue>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//If you have libopencv-dev in version 3.x, compile with
//g++ -std=c++11 background_elimination.cpp -o background_elimination `pkg-config --cflags --libs opencv`

//Else if you have libopencv-dev in version 4.x, compile with
//g++ -std=c++11 background_elimination.cpp -o background_elimination `pkg-config --cflags --libs opencv4`

int component_iterative(cv::Mat &point_value, int i, int j, int mark, int frame_xmin, int frame_xmax, int frame_ymin, int frame_ymax, int to_visit) {
    std::vector<std::vector<int> > caneva_for_adjascency{ {-1,0} , {0,-1} , {0,1} , {1,0} };
    int ctr = 0;
    std::queue<std::vector<int>> q;
    q.push(std::vector<int>{i,j});
    while (!q.empty()) {
        std::vector<int> v;
        v = q.front();
        q.pop();
        int i = v[0];
        int j = v[1];
        if (point_value.at<int>(i,j) == to_visit) {
            point_value.at<int>(i,j) = mark;
            ctr++;
            for (int k = 0; k < caneva_for_adjascency.size(); k++) {
                int i2 = i + caneva_for_adjascency[k][0];
                int j2 = j + caneva_for_adjascency[k][1];
                if (i2 >= frame_xmin && i2 < frame_xmax && j2 >= frame_ymin && j2 < frame_ymax) {
                    if (point_value.at<int>(i2,j2) == to_visit) {
                        q.push(std::vector<int>{i2, j2});
                    }
                }
            }

        }
    }
    return ctr;
}

int main(int argc, char** argv) {
	auto timer_0 = std::chrono::high_resolution_clock::now();
	int edge_color_gap = 20;
	int min_color_to_consider = 3;
	int max_slope = 3;
	float minimum_content_ratio = 0.19;
	int maximum_angle = 7;
	int min_edge_component = 200;
	int frame_xmin;
	int frame_xmax;
	int frame_ymin;
	int frame_ymax;
	std::vector<std::vector<int> >caneva_for_edge { {0,-1} , {0,0} , {0,1} };
	// 2: will show timing
	int verbose = 2;

	// storing different frames for different cameras
	std::map<int, std::vector<int>> camera_frames;
	camera_frames[1] = std::vector<int>{40, 270, 50, 190};
	camera_frames[0] = std::vector<int>{40, 270, 40, 140};
	camera_frames[2] = std::vector<int>{40, 270, 100, 200};

	if (argc != 3){
		std::cout << "Usage : ./background_elimination <file> <camera>" << std::endl;
		return -1;
	}

	std::string filename(argv[1]);
	int camera = std::atoi(argv[2]);
	std::string delimiter = ".";
	std::string name = filename.substr(0, filename.find(delimiter));
	std::stringstream tmp;
	std::stringstream tmp_2;
	tmp << name << "_mod.png";
	tmp_2 << name << "_mod_mod.png";
	std::string target = tmp.str();
	std::string target_2 = tmp_2.str();

	if (camera_frames.find(camera) != camera_frames.end()) {
		frame_xmin = camera_frames[camera][0];
		frame_xmax = camera_frames[camera][1];
		frame_ymin = camera_frames[camera][2];
		frame_ymax = camera_frames[camera][3];
	} else {
		std::cout << "Camera must be 0, 1 or 2" << std::endl;
		return -1;
	}


	// read image as int
	cv::Mat img;
	img = cv::imread(filename, cv::IMREAD_GRAYSCALE); // in Serge script he loads as an RGB
	if(! img.data )                              // Check for invalid input
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    int width = img.cols;
 	int height = img.rows;
 	auto timer_1 = std::chrono::high_resolution_clock::now();
 	std::chrono::duration<double> read_image =  timer_1 - timer_0;
 	if (verbose > 1)
 		std::cout << read_image.count() <<  " s to read image" << std::endl;

 	//convert pixels to int
    img.convertTo(img, CV_32S);
    auto timer_2 = std::chrono::high_resolution_clock::now();
 	std::chrono::duration<double> to_int_pixels =  timer_2 - timer_1;
 	if (verbose > 1)
 		std::cout << to_int_pixels.count() <<  " s to int pixels" << std::endl;

 	cv::Mat is_not_edge(width, height, CV_32S, cv::Scalar(0));
	std::vector<std::vector<int> > edge_points;
	std::vector<std::vector<int> > non_edge_points;

	for (int x = frame_xmin; x < frame_xmax; x++) {
		for (int y = frame_ymin; y < frame_ymax; y++) {
			int mincol = img.at<int>(y,x);
			int flag = 1;
			if (mincol >= min_color_to_consider) {
				is_not_edge.at<int>(x,y) = 1;
				// optimization for caneva_for_edge
				int maxcol = mincol;
				if (y > frame_ymin) {
					int pix = img.at<int>(y-1, x);
					mincol = std::min(mincol, pix);
					maxcol = std::max(maxcol, pix);
				} else {
					mincol = 0;
				}
				if (y < frame_ymax - 1) {
					int pix = img.at<int>(y+1,x);
					mincol = std::min(mincol, pix);
					maxcol = std::max(maxcol, pix);
				} else {
					mincol = 0;
				}
				flag = (maxcol - mincol) > edge_color_gap;
			}

			if (flag) {
				is_not_edge.at<int>(x,y) = 0;
				edge_points.push_back(std::vector<int>{x,y});
			} else {
				non_edge_points.push_back(std::vector<int>{x,y});
			}
		}
	}

	auto timer_21 = std::chrono::high_resolution_clock::now();
 	std::chrono::duration<double> get_edges =  timer_21 - timer_2;
 	if (verbose > 1)
 		std::cout << get_edges.count() <<  " s to get edges" << std::endl;

	// look at connected edge pixels
	// small connected components are unreliable edge pixels
	int mark = -1;
	std::map<int, int> size_component_edges;
	for (auto pt : edge_points) {
		int x = pt[0];
		int y = pt[1];
		if (is_not_edge.at<int>(x,y) == 0) {
			size_component_edges[mark] = component_iterative(is_not_edge, x, y, mark, frame_xmin, frame_xmax, frame_ymin, frame_ymax, 0);
			mark--;
		}
	}

	for (auto pt : edge_points) {
		int x = pt[0];
		int y = pt[1];
		if (size_component_edges[is_not_edge.at<int>(x,y)] < min_edge_component) {
			is_not_edge.at<int>(x,y) = 0;
		}
	}
	auto timer_22 = std::chrono::high_resolution_clock::now();
 	std::chrono::duration<double> connected_components_of_edges =  timer_22 - timer_21;
 	if (verbose > 1)
 		std::cout << connected_components_of_edges.count() <<  " s to get connected components of edges" << std::endl;

	// identify connected components
	// (exclude components vertically connected to border; see below)
	mark = 2;
	std::map<int, int> size_component;
	for (auto pt : non_edge_points) {
		int x = pt[0];
		int y = pt[1];
		if (is_not_edge.at<int>(x,y) == 1)
			size_component[mark] = component_iterative(is_not_edge, x, y, mark, frame_xmin, frame_xmax, frame_ymin, frame_ymax, 1);
		++mark;
	}
	auto timer_23 = std::chrono::high_resolution_clock::now();
 	std::chrono::duration<double> connected_components =  timer_23 - timer_22;
 	if (verbose > 1)
 		std::cout << connected_components.count() <<  " s to get connected components" << std::endl;

	// uncount from the components pixels which are vertically adjascent to borders
	// because it is certainly the background and may look like a big component
	// normally, this is not necessary because border pixels are too dark
	// TODO: can be paralellized
	for (int x = frame_xmin; x < frame_xmax; x++) {
		int y = frame_ymin;
		mark = is_not_edge.at<int>(x,y);
		if (mark > 0) {
			while (is_not_edge.at<int>(x,y) == mark && (y < frame_ymax)) {
				is_not_edge.at<int>(x,y) = 0;
				size_component[mark] -= 1;
				y++;
			}
		}
		y = frame_ymax - 1;
		mark = is_not_edge.at<int>(x,y);
		if (mark > 0) {
			while (is_not_edge.at<int>(x,y) == mark && (y >= frame_ymin)) {
				is_not_edge.at<int>(x,y) = 0;
				size_component[mark] -= 1;
				y--;
			}
		}
	}

	// find the largest connected component
	int largest_m = 0;
	int largest_ct = 0;
	for (auto const& m : size_component) {
		if (m.second > largest_ct) {
			largest_ct = m.second;
			largest_m = m.first;
		}
	}
	auto timer_24 = std::chrono::high_resolution_clock::now();
 	std::chrono::duration<double> postprocess_connected_components =  timer_24 - timer_23;
 	if (verbose > 1)
 		std::cout << postprocess_connected_components.count() <<  " s to postprocess connected components" << std::endl;

	// largest vertical segments containing pixels which are either
	// in the largest connected component or unreliable edges
	// take a segment as reliable if is has no unreliable edge inside
	std::map<int, int> segment_ymin;
	std::map<int, int> segment_ymax;
	std::map<int, int> reliable;
	// TODO: can be paralellized
	int ymin_max = 0;
	int ymax_max = 0;
	for (int x = frame_xmin; x < frame_xmax; x++) {
		int length_max = -1;
		int ymin = -1;
		int ymax = 0;
		reliable[x] = 1;
		for (int y = frame_ymin; y < frame_ymax; y++) {
			if (is_not_edge.at<int>(x,y) == largest_m || is_not_edge.at<int>(x,y) == 0) {
				if (is_not_edge.at<int>(x,y) == 0)
					reliable[x] = 0;
				if (ymin < 0) {
					ymin = y;
					ymax = y;
				} else {
					if (y != (ymax + 1)) {
						if ((ymax - ymin) > length_max) {
							length_max = ymax - ymin;
							ymin_max = ymin;
							ymax_max = ymax;
						}
						ymin = y;
					}
					ymax = y;
				}
			}
		}
		if (!(ymin < 0)) {
			if ((ymax - ymin) > length_max) {
				length_max = ymax - ymin;
				ymin_max = ymin;
				ymax_max = ymax;
			}
			segment_ymin[x] = ymin_max;
			segment_ymax[x] = ymax_max;
		}
	}

	// interpolate ymin and ymax from reliable ones
	int last_x = 0;
	std::map<int, int> ymin;
	for (int x = frame_xmin; x < frame_xmax; x++) {
		if (segment_ymin.find(x) != segment_ymin.end()) {
			if (last_x >= 0) {
				int slope = (segment_ymin[x] - segment_ymin[last_x]) / (x - last_x);
				if (std::abs(slope) < max_slope) {
					// slope is reasonable
					if (last_x < (x -1)) {
						// interpolate on missing x's
						for (int x2 = last_x + 1; x2 < x; x2++) {
							ymin[x2] = std::floor(slope * (x2 - x) + segment_ymin[x]);
						}
					}
					last_x = x;
					ymin[x] = segment_ymin[x];
				}
			} else {
				last_x = x;
				ymin[x] = segment_ymin[x];
			}
		}
	}

	last_x = 0;
	std::map<int, int> ymax;
	for (int x = frame_xmin; x < frame_xmax; x++) {
		if (segment_ymax.find(x) != segment_ymax.end()) {
			if (last_x >= 0) {
				int slope = (segment_ymax[x] - segment_ymax[last_x]) / (x - last_x);
				if (std::abs(slope) < max_slope) {
					// slope is reasonable
					if (last_x < (x - 1)) {
						// interpolate on missing x's
						for (int x2 = last_x + 1; x2 < x; x2++) {
							ymax[x2] = std::floor(slope * (x2 - x) + segment_ymax[x]);
						}
					}
					last_x = x;
					ymax[x] = segment_ymax[x];
				}
			} else {
				last_x = x;
				ymax[x] = segment_ymax[x];
			}
		}
	}
	auto timer_25 = std::chrono::high_resolution_clock::now();
 	std::chrono::duration<double> get_segments =  timer_25 - timer_24;
 	if (verbose > 1)
 		std::cout << get_segments.count() <<  " s to get segments" << std::endl;

	std::map<std::string, int> sigmas = {{"n", 0}, {"x", 0}, {"y", 0}, {"xx", 0}, {"xy", 0}, {"yy", 0}};
	for (auto const& d : ymin) {
		int x = d.first;
		int y = d.second;
		if (x>= frame_xmin && x < frame_xmax && y >= frame_ymin && y < frame_ymax) {
			sigmas["n"] += 1;
			sigmas["x"] += x;
			sigmas["y"] += y;
			sigmas["xx"] += x * x;
			sigmas["xy"] += x * y;
			sigmas["yy"] += y * y;
		}
	}

	float var_x = (float)(sigmas["xx"] - sigmas["x"] * sigmas["x"] / sigmas["n"]) / sigmas["n"];
	float var_y = (float)(sigmas["yy"] - sigmas["y"] * sigmas["y"] / sigmas["n"]) / sigmas["n"];
	float covar = (float)(sigmas["xy"] - sigmas["x"] * sigmas["y"] / sigmas["n"]) / sigmas["n"];
	float lambd = (float)covar / var_x;
	float mu = (float)sigmas["y"] / sigmas["n"] - lambd * sigmas["x"] / sigmas["n"];
	float angle_ymin = std::atan(lambd) / M_PI * 180;

	sigmas = {{"n", 0}, {"x", 0}, {"y", 0}, {"xx", 0}, {"xy", 0}, {"yy", 0}};
	for (auto const& d : ymax) {
		int x = d.first;
		int y = d.second;
		if (x>= frame_xmin && x < frame_xmax && y >= frame_ymin && y < frame_ymax) {
			sigmas["n"] += 1;
			sigmas["x"] += x;
			sigmas["y"] += y;
			sigmas["xx"] += x * x;
			sigmas["xy"] += x * y;
			sigmas["yy"] += y * y;
		}
	}

	var_x = (float)(sigmas["xx"] - sigmas["x"] * sigmas["x"] / sigmas["n"]) / sigmas["n"];
	var_y = (float)(sigmas["yy"] - sigmas["y"] * sigmas["y"] / sigmas["n"]) / sigmas["n"];
	covar = (float)(sigmas["xy"] - sigmas["x"] * sigmas["y"] / sigmas["n"]) / sigmas["n"];
	lambd = (float)covar / var_x;
	mu = (float)sigmas["y"] / sigmas["n"] - lambd * sigmas["x"] / sigmas["n"];
	float angle_ymax = std::atan(lambd) / M_PI * 180;

	float angle = (angle_ymin + angle_ymax) / 2.0;
	if (camera == 1) {
		std::cout << "angle: " << angle << std::endl;
	}

	// erease out pixels
	cv::Mat mask(height, width, CV_32S, cv::Scalar(0));
	int ct = 0;
	for (int x = frame_xmin; x < frame_xmax; x++) {
		if (ymin.find(x) != ymin.end() && ymax.find(x) != ymax.end()) {
			for (int y = ymin[x]; y < ymax[x]; y++) {
				mask.at<int>(y,x) = 255;
				ct++;
			}
		}
	}

	// masking the image
	cv::bitwise_and(img, mask, img);
	// ratio of kept content
	float content = 1 - (float)ct / (frame_xmax - frame_xmin) / (frame_ymax - frame_ymin);
	cv::imwrite(target, img);

	// rotation by angle
 	// opencv only accepts float matrixes
 	cv::Point2f center(width/2, height/2);
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1);
    cv::Mat dst(height, width, CV_32FC1, cv::Scalar(0.0));
    cv::Mat img2;
    img.convertTo(img2, CV_32FC1);
    cv::warpAffine(img2, dst, M, dst.size());
	cv::imwrite(target_2, dst);

	// check that the result is good enough
	if (camera == 1) {
		if (std::abs(angle) > maximum_angle) {
			std::cout << "Finger needs to be aligned!" << std::endl;
			return -1;
		}


		if (content < minimum_content_ratio) {
			std::cout << "Too much finger was cut!" << std::endl;
			return -1;
		}
	}
	auto timer_3 = std::chrono::high_resolution_clock::now();
 	std::chrono::duration<double> earease_pixels =  timer_3 - timer_25;
 	std::chrono::duration<double> eliminate_background =  timer_3 - timer_2;
 	if (verbose > 1){
 		std::cout << earease_pixels.count() <<  " s to earease pixels" << std::endl;
 		std::cout << eliminate_background.count() <<  " s to eliminate background" << std::endl;
 	}

    auto timer_4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> rotation = timer_4 - timer_3;
    if (verbose > 1){
 		std::cout << rotation.count() <<  " s to rotate the finger" << std::endl;
 	}

    std::chrono::duration<double> elapsed = timer_4 - timer_0;
    std::cout << "Total elapsed time: " << elapsed.count() << " s" << std::endl;

	return 0;
}
