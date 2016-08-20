#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <Windows.h>

using namespace cv;
using namespace std;

void GLCM_calc(Mat& I, int distance, int direction, ofstream &ofile);
int openImage(string &imageName, ofstream &ofile);
vector<string> get_all_files_names_within_folder(string folder, string format);


int main(int argc, char** argv)
{
	if (argc != 3)
	{
		cout << "Usage is opencv1 \"directory path\" extension" << endl;
		return -1;
	}

	ofstream outfile("Results.txt");
	vector<string> fileNames;
	string filename;
	string path = argv[1];
	string format = argv[2];

	fileNames = get_all_files_names_within_folder(path, format);

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

	outfile.close();

	cout << "All images processed" << endl;

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
	uchar divideWith = 16;//16;
	uchar table[256];
	//for (int i = 0; i < 256; ++i)
	//	table[i] = (uchar)(divideWith * (i / divideWith));

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

	for (int i = 0; i<2;++i)
		for (int j = 0;j<4;++j)
			GLCM_calc(newImage, distances[i], j, ofile);

	return 0;
}

void GLCM_calc(Mat& I, int distance, int direction, ofstream &ofile)
{
	// accept only char type matrices
	//CV_Assert(I.depth() == CV_8U);

	int channels = I.channels();

	int nRows = I.rows;
	int nCols = I.cols * channels;

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


	int i, j;
	for (i = 0; i < nRows; ++i)
	{
		for (j = 0; j < nCols; ++j)
		{
			if ((i + d0 < nRows) && (i + d0 >= 0) && (j + d1 < nCols) && (j + d1 >= 0))
				GLCM[(size_t)I.at<uchar>(i, j)][(size_t)I.at<uchar>(i + d0, j + d1)]++;

			if ((i - d0 >= 0) && (i - d0 < nRows) && (j - d1 >= 0) && (j - d1 < nCols))
				GLCM[(size_t)I.at<uchar>(i, j)][(size_t)I.at<uchar>(i - d0, j - d1)]++;
		}
	}

	int sum = 0;
	float P[16][16] = { 0 };

	for (int k = 0; k < 16; k++)
	{
		for (int l = 0; l < 16; l++)
		{
			sum += GLCM[k][l];
			P[k][l] =(float) GLCM[k][l];
		}
	}

	//Probability
	for (int k = 0; k < 16; k++)
	{
		for (int l = 0; l < 16; l++)
		{
			P[k][l] /= (float)sum;
		}
	}

	// mean
	float mu_i = 0, mu_j = 0;
	for (int k = 0; k < 16; k++)
	{
		for (int l = 0; l < 16; l++)
		{
			mu_i += k*P[k][l];
			mu_j += k*P[k][l];
		}
	}

	// standard deviation
	float sd_i = 0, sd_j = 0;
	for (int k = 0; k < 16; k++)
	{
		for (int l = 0; l < 16; l++)
		{
			sd_i += P[k][l] * (k - mu_i)*(k - mu_i);
			sd_j += P[k][l] * (l - mu_j)*(l - mu_j);
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

	for (int k = 0; k < 16; k++)
	{
		for (int l = 0; l < 16; l++)
		{
			//probability
			if (P[k][l] > max_probability)
				max_probability = P[k][l];
			//energy
			energy += P[k][l] * P[k][l];
			//homogeneity
			homogeneity += P[k][l] / (1 + abs(k - l));
			//contrast
			contrast += P[k][l] * (k - l)*(k - l);
			//correlation
			correlation += (P[k][l] * (k - mu_i)*(l - mu_j) / (sd_i*sd_j));
			//entropy
			if (GLCM[k][l] != 0)
				entropy += P[k][l] * -log(P[k][l]);

		}
	}

	ofile << max_probability;
	ofile << " " << energy;
	ofile << " " << homogeneity;
	ofile << " " << contrast;
	ofile << " " << correlation;
	ofile << " " << entropy;


}

vector<string> get_all_files_names_within_folder(string folder, string format)
{
	vector<string> names;
	string search_path = folder + "/*." + format;
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}