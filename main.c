#include "main.h"

int main(void)
{
	double feature[DIMENSION][SAMPLE_SIZE];
	int label[SAMPLE_SIZE];
	
	int result = ReadCSV("iris/iris.data");
	int count = ReadIrisCSV("iris/iris.data", feature, label, SAMPLE_SIZE);

	if (count == 0)
	{
		printf("Iris CSV loading failed...\n");
		return -1;
	}

	printf("Loaded %d samples\n\n", count);

	for (int i = 0; i < 5; i++)
    {
        printf("%f %f %f %f -> %d\n",
               feature[0][i],
               feature[1][i],
               feature[2][i],
               feature[3][i],
               label[i]);
    }

	return 0;
}
