#ifndef _anngpm_h
#define _annpgm_h
#include <iostream>
#include <random>
#include <math.h>
#include <iomanip>
#include <pthread.h>
#include <sys/stat.h>
#include "pgmreader.h"


using namespace arma;
using namespace std;



vec_data *get_vec_data_ppm(pm_img *img, int numdata);

std::fstream startHistory(const char *filename, nn *thisnn, vec_data *D, int numGenerations);
bool appendNNToHistory(nn *thisnn, fstream *fp);

void write_nn_to_img(nn *thisnn, const char filename[], int height, int width, int func);

void write_nn_regions_to_img(nn *thisnn, const char filename[], int height, int width, int func);
void write_nn_inter_to_img(nn *thisnn, const char filename[], int height, int width, int func);

void write_all_nn_to_image(nn *thisnn,vec_data *data, const char filename[], int height, int width);
void write_all_nn_to_image_parallel(nn *thisnn,vec_data *data, const char filename[], int height, int width);

void write_data_to_img(vec_data *data,const char filename[]);

void write_hperrs_to_imgs(nn *thisnn, const char filename[], const char dirname[]);

#endif