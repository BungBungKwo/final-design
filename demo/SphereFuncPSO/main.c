#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

/* PSO Parameters Setting */
#ifndef PSO
#define DIMENSION 2	// Dimensions of Problem
#define SWARM_SIZE 20	// Typical: 15-30
#define MAX_ITERATION 300	// Maximum iteration
#define INTERVAL 50	// Iteration Interval (for visiualization)
#define W 0.5	// Weight of Velocity
#define C1 1.5	// Weight of Pbest
#define C2 1.5	// Weight of Gbest
#define MIN_X -10.0
#define MAX_X 10.0
#define MIN_V -0.5
#define MAX_V 0.5

typedef struct {
	double x[DIMENSION];	// Position of Particle
	double v[DIMENSION];	// Velocity of Particle
	double fitness;		// Fitness of Particle
	double pbestfit;		// Best Fitness of Particle
	double pbestx[DIMENSION]; // Position correspond to cornfield vector
} Particle;

double FitnessFunc(double x1, double x2);

/* Fitness Function: Sphere function (Minimalized Version) */
double FitnessFunc(double x1, double x2)
{
	double Max_Fitness = 2.0 * MAX_X * MAX_X;
	return ((x1*x1 + x2*x2) / Max_Fitness);
}
#endif

int main(void)
{
	FILE *fp = NULL;
	srand(time(NULL));

	/* Search Space Parameters */
	const double STEP = 0.1;

	/* PSO Paradigm Testing */
	printf("====== PSO Algorithm Test: Sphere Function ======\n");
	printf("Test starting...\n");
	
	/* Search Space Parameters */

	/* Initialization of Swarm */
	printf("Swarm initialization starting...\n");
	Particle swarm[SWARM_SIZE];
	int gbest = 0;
	int i, j, iter;
	for (i = 0; i < SWARM_SIZE; i++)
	{
		for (j = 0; j < DIMENSION; j++)
		{
			swarm[i].x[j] = MIN_X + (double)rand() / RAND_MAX * (MAX_X - MIN_X);	// Particle Position Initialization
			swarm[i].v[j] = MIN_V + (double)rand() / RAND_MAX * (MAX_V - MIN_V);	// Particle Velocity Initialization
			swarm[i].pbestx[j] = swarm[i].x[j];				// Particle Best Position Initialization
		}
		swarm[i].fitness = FitnessFunc(swarm[i].x[0], swarm[i].x[1]);	// Particle Fitness Initialization	
		swarm[i].pbestfit = swarm[i].fitness;	// Particcle Best Fitness Initialization
		
		// Global Best Particle Index Initialization
		if (swarm[i].pbestfit < swarm[gbest].pbestfit)
			gbest = i;
	}
	printf("Swarm initialization completed.\n");
	printf("Initial best fitness: %.10f at (%.4f, %.4f)\n", swarm[gbest].pbestfit, swarm[gbest].pbestx[0], swarm[gbest].pbestx[1]);

	/* Particles Update */
	fp = fopen("Fitness.dat", "w");
	for (iter = 0; iter < MAX_ITERATION; iter++)
	{
		for(i = 0; i < SWARM_SIZE; i++)
		{
			swarm[i].fitness = FitnessFunc(swarm[i].x[0], swarm[i].x[1]);// Particle Fitness Calculation
			if (swarm[i].fitness < swarm[i].pbestfit)
			{
				swarm[i].pbestfit = swarm[i].fitness;	// Particle Best Fitness Update
				for (j = 0; j < DIMENSION; j++)
					swarm[i].pbestx[j] = swarm[i].x[j];	// Particle Best Position Update
			}
			if (swarm[i].fitness < swarm[gbest].pbestfit)	// Gobal Best Index Update
				gbest = i;
			for (j = 0; j < DIMENSION; j++)
			{
				double r1 = (double)rand()/RAND_MAX;
				double r2 = (double)rand()/RAND_MAX;

				/* Particle Velocity Update */
				swarm[i].v[j] = W * swarm[i].v[j] + C1 * r1 * (swarm[i].pbestx[j] - swarm[i].x[j]) + C2 * r2 * (swarm[gbest].pbestx[j] - swarm[i].x[j]);
				/* Velocity Limitation */
				if (swarm[i].v[j] > MAX_V)
					swarm[i].v[j] = MAX_V;
				if (swarm[i].v[j] < MIN_V)
					swarm[i].v[j] = MIN_V;

				/* Particle Position Update */
				swarm[i].x[j] += swarm[i].v[j];
				// Position Limitation
				if (swarm[i].x[j] > MAX_X)
					swarm[i].x[j] = MAX_X;
				if (swarm[i].x[j] < MIN_X)
					swarm[i].x[j] = MIN_X;
			}
		}
		/* Iteration Output */
		if ((iter + 1) % 10 == 0)
		{
			printf("Iteration %3d: Bestfitness = %.10e at (%.4f, %.4f)\n", iter + 1, swarm[gbest].pbestfit, swarm[gbest].pbestx[0], swarm[gbest].pbestx[1]);
		}
		fprintf(fp, "%d %.10e\n", iter, swarm[gbest].pbestfit);	// Global best fitness of each iteration (convergence speed)
		// Particles' position of iteration interval
		char snapshot_filename[100];
		if (iter == 0 || (iter+1) % INTERVAL == 0)
		{
			sprintf(snapshot_filename, "snapshoot%03d.dat", iter+1);
			FILE *fp_snap = fopen(snapshot_filename, "w");
			for (int i = 0; i < SWARM_SIZE; i++)
			{
				fprintf(fp_snap, "%.6f %.6f\n", swarm[i].x[0], swarm[i].x[1]);
			}
			fclose(fp_snap);
		}
	}
	fclose(fp);

	/* Optimization Output */
	return 0;
}