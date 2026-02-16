#ifndef CSV_H
#define CSV_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../cfg/cfg.h"

#define LABEL_SIZE 5

int ReadCSV(const char *filename);
int ReadIrisCSV(const char *filename, double feature[DIMENSION][SAMPLE_SIZE], int label[LABEL_SIZE], int SampleSize);

#endif
