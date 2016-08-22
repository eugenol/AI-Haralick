#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <Windows.h>
#include <cmath>
#include <ctime>

using namespace cv;
using namespace std;

void GLCM_calc(Mat& I, int distance, int direction, ofstream &ofile);
int openImage(string &imageName, ofstream &ofile);
vector<string> get_all_files_names_within_folder(string folder, string format);


int main(int argc, char** argv)
{
	string path;
	string format;

	if (argc == 1) // no arguments given
	{
		cout << "Usage is opencv1 \"directory path\" [extension]" << endl;
		return -1;
	}
	else if (argc == 2) //only file path
	{
		path = argv[1];
		format = "*";
	}
	else //file path and extension
	{
		path = argv[1];
		format = argv[2];
	}

	ofstream outfile("Results.txt");
	vector<string> fileNames;
	string filename;

	//Get all files in given folder
	fileNames = get_all_files_names_within_folder(path, format);

	if (fileNames.empty())
	{
		cout << "No files in given directory." << endl;
		return -1;
	}

	clock_t starttime = clock();

	//process each image
	for each(string file in fileNames)
	{
		cout << "Processing Image: " << file;
		filename = path + "\\" + file;
		if (openImage(filename, outfile)==0)
			cout << " - Succesful" << endl;
		else
			cout << " - Could not open Image" << endl;
		outfile << endl;
	}

	clock_t endtime = clock();

	double runtime = (endtime - starttime) / (double)CLOCKS_PER_SEC;
	outfile.close();

	cout << "All images processed." << endl;
	cout << fileNames.size() <<" images processed in " << runtime<< " seconds."<< endl;

	return 0;
}

int openImage(string &imageName, ofstream &ofile)
{
	Mat image;
	image = imread(imageName.c_str(), IMREAD_GRAYSCALE); // Read the file

	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	CV_Assert(image.depth() == CV_8U);  // accept only uchar images

	// Reduce gray levels to have 16 levels only
	uchar divideWith = 16;
	uchar table[256];

	//scale to 16 gray levels
	for (int i = 0; i < 256; ++i)
		table[i] = (uchar)((i / divideWith));

	// make look up table
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.data;
	for (int i = 0; i < 256; ++i)
		p[i] = table[i];

	// apply to image
	Mat newImage;
	LUT(image, lookUpTable, newImage);

	int distances[2] = { 1, 3 }; 

	for (int i = 0; i < 2;++i) // 2 distances
	{
		for (int j = 0;j < 4;++j) // 4 directions
		{
			GLCM_calc(newImage, distances[i], j, ofile);
		}
	}

	return 0; // success
}

void GLCM_calc(Mat& I, int distance, int direction, ofstream &ofile)
{
	// accept only char type matrices
	//CV_Assert(I.depth() == CV_8U);

	int channels = I.channels();

	int nRows = I.rows;
	int nCols = I.cols * channels;

	int cfactor = 16; //factor for fixing levels- reduced to 16 to make matrix smaller

	//set up matrix
	int GLCM[16][16] = { 0 };
	int d0, d1;

	switch (direction)
	{
		// 0 degrees
	case 0:		d0 = 0;
		d1 = distance;
		break;
		//45 degrees
	case 1:	d0 = -distance;
		d1 = distance;
		break;
		//90 degrees
	case 2:	d0 = -distance;
		d1 = 0;
		break;
		//135 degrees
	case 3:	d0 = -distance;
		d1 = -distance;
		break;
	}

	// To do: Try to speed this up by using pointers
	for (int i = 0; i < nRows; ++i)
	{
		for (int j = 0; j < nCols; ++j)
		{
			if ((i + d0 < nRows) && (i + d0 >= 0) && (j + d1 < nCols) && (j + d1 >= 0))
				GLCM[(size_t)I.at<uchar>(i, j)][(size_t)I.at<uchar>(i + d0, j + d1)]++;

			if ((i - d0 >= 0) && (i - d0 < nRows) && (j - d1 >= 0) && (j - d1 < nCols))
				GLCM[(size_t)I.at<uchar>(i, j)][(size_t)I.at<uchar>(i - d0, j - d1)]++;
		}
	}

	int sum = 0;
	float P[16][16] = { 0 };

	for (int i = 0; i < 16; ++i)
	{
		for (int j = 0; j < 16; ++j)
		{
			sum += GLCM[i][j];
			P[i][j] =(float) GLCM[i][j];
		}
	}

	//Probability
	for (int i = 0; i < 16; ++i)
	{
		for (int j = 0; j < 16; ++j)
		{
			P[i][j] /= (float)sum;
		}
	}

	// mean
	float mu_i = 0, mu_j = 0;
	for (int i = 0; i < 16; ++i)
	{
		for (int j = 0; j < 16; ++j)
		{
			mu_i += cfactor*i*P[i][j];
			mu_j += cfactor*j*P[i][j];
		}
	}

	// standard deviation
	float sd_i = 0, sd_j = 0;
	for (int i = 0; i < 16; ++i)
	{
		for (int j = 0; j < 16; ++j)
		{
			sd_i += P[i][j] * (cfactor*i - mu_i)*(cfactor*i - mu_i);
			sd_j += P[i][j] * (cfactor*j - mu_j)*(cfactor*j - mu_j);
		}
	}
	sd_i = sqrtf(sd_i);
	sd_j = sqrtf(sd_j);


	float max_probability = P[0][0];
	float energy = 0;
	float homogeneity = 0;
	float contrast = 0;
	float correlation = 0;
	float entropy = 0;

	for (int i = 0; i < 16; ++i)
	{
		for (int j = 0; j < 16; ++j)
		{
			//probability
			if (P[i][j] > max_probability)
				max_probability = P[i][j];
			//energy
			energy += P[i][j] * P[i][j];
			//homogeneity
			homogeneity += P[i][j] / (1 + abs(cfactor*i - cfactor*j));
			//contrast
			contrast += P[i][j] * abs(cfactor*i - cfactor*j)*abs(cfactor*i - cfactor*j);
			//correlation
			correlation += (P[i][j] * (cfactor*i - mu_i)*(cfactor*j - mu_j) / (sd_i*sd_j));
			//entropy
			if (GLCM[i][j] != 0)
				entropy += P[i][j] * -log(P[i][j]);

		}
	}

	// Output features to file, delimited by a space
	ofile << max_probability;
	ofile << " " << energy;
	ofile << " " << homogeneity;
	ofile << " " << contrast;
	ofile << " " << correlation;
	ofile << " " << entropy <<" ";
}

/*
Find all files in a given folder
http://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
*/
vector<string> get_all_files_names_within_folder(string folder, string format)
{
	vector<string> names;
	string search_path = folder + "/*." + format;
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);

	if (hFind != INVALID_HANDLE_VALUE)
	{
		do 
		{
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}