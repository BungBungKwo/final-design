#include "csv.h"

int ReadCSV(const char *filename)
{
	FILE *fp = fopen(filename, "r");
	if (!fp)
	{
		printf("Failed to open file\n");
		return -1;
	}
	
	char buf[1024];
	while(fgets(buf, sizeof(buf), fp))
	{
		char *token = strtok(buf, ",");
		while(token)
		{
			printf("%s\n", token);
			token = strtok(NULL, ",");
		}
	}

	fclose(fp);

	return 0;
}

int ReadIrisCSV(const char *filename, double feature[DIMENSION][SAMPLE_SIZE], int label[LABEL_SIZE], int SampleSize)
{
	FILE *fp = fopen(filename, "r");

	if(!fp) return -1; // Error handling

	char buf[1024];
	int samp_idx = 0;	// Sample Index

	while(fgets(buf, 1024, fp) && samp_idx < SAMPLE_SIZE)
	{
		if(strlen(buf) <= 1)
		{
			continue;
		}
		char *token = strtok(buf, ",");

		for(int i = 0; i < DIMENSION; i++)
		{
			if(!token) break;
			feature[i][samp_idx] = atof(token);
			token = strtok(NULL, ",");
		}

		if(token)
		{
			token[strcspn(token, "\r\n")] = 0;

			if(strcmp(token, "Iris-setosa") == 0)
				label[samp_idx] = 0;
			else if(strcmp(token, "Iris-versicolor") == 0)
				label[samp_idx] = 1;
			else label[samp_idx] = 2;
		}

		samp_idx++;
	}

	fclose(fp);

	return samp_idx;
}