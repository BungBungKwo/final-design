#include <stdio.h>
#include <math.h>

double FitnessFunc(double x1, double x2);
int FitnessFunc_Datafile_Gen(FILE *fp);

/* Fitness Function: Schaffer f6 function (Minimalized Version) */
double FitnessFunc(double x1, double x2)
{
	double sum_sq = x1 * x1 + x2 * x2;
	double sqrt_sumsq = sqrt(sum_sq);
	double sin_sq = sin(sqrt_sumsq) * sin(sqrt_sumsq);
	double denominator = 1.0 + 0.001 * sum_sq;
	return (0.5 + (sin_sq - 0.5) / (denominator * denominator));
}

/* Generate Schaffer F6 Datafile */
int FitnessFunc_Datafile_Gen(FILE *fp)
{
	double x1, x2;
	const double MIN_X = -100.0;
	const double MAX_X = 100.0;
	const double STEP = 0.1;

	/* Fitness Function Generation */
	if (!fp)
		return 1;
	fprintf(fp, "# x1 x2 fitness\n");
	for (x1 = MIN_X; x1 <= MAX_X; x1 += STEP)
	{
		for (x2 = MIN_X; x2 <= MAX_X; x2 += STEP)
		{
			double fitness = FitnessFunc(x1,x2);
			fprintf(fp, "%f %f %f\n", x1, x2, fitness);
		}
		fprintf(fp, "\n");
	}

	return 0; // Error Code: No error
}

int main(void)
{
	FILE *fp = NULL;

	printf("====== Datafile of Schaffer F6 Functin Generation ======\n");
	fp = fopen("Schaffer_F6_Data.dat", "w");
	if(!FitnessFunc_Datafile_Gen(fp))
		printf("Datafile Generation Successed!\n\n");
	else
		printf("Datafile Generation Failed!\n\n");
	fclose(fp);

	return 0;
}
