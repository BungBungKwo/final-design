#include "main.h"

int main(void)
{
	FILE *fp = NULL;

	FILE *rand_seed_fp = fopen("RandNumSeed.txt", "r");
	unsigned int rand_seed;
	if(!rand_seed_fp)
	{
		rand_seed_fp = fopen("RandNumSeed.txt", "w");
		rand_seed = (unsigned int)time(NULL);
		fprintf(rand_seed_fp,"%u\n", rand_seed);
	}
	else
	{
		fscanf(rand_seed_fp, "%u", &rand_seed);
	}
	srand(rand_seed);

	/* PSO Paradigm Testing */
	printf("====== PSO Algorithm Test: Sphere Function ======\n");
	printf("Test starting...\n");

	/* Initialization of Swarm */
	printf("Swarm initialization starting...\n");

	Particle swarm[SWARM_SIZE];	// Swarm (Particles)
	int gbest = 0;	//  Index of global best
	int i, j, iter;
	
	for (i = 0; i < SWARM_SIZE; i++)
	{
		for (j = 0; j < DIMENSION; j++)
		{
			swarm[i].x[j] = MIN_X + (double)rand() / RAND_MAX * (MAX_X - MIN_X);	// Particle Position Initialization
			swarm[i].v[j] = MIN_V + (double)rand() / RAND_MAX * (MAX_V - MIN_V);	// Particle Velocity Initialization
			swarm[i].pbestx[j] = swarm[i].x[j];				// Particle Best Position Initialization
		}
		swarm[i].fitness = FitnessFunc(swarm[i].x, DIMENSION);	// Particle Fitness Initialization	
		swarm[i].pbestfit = swarm[i].fitness;	// Particcle Best Fitness Initialization
		
		// Global Best Particle Index Initialization
		if (swarm[i].pbestfit < swarm[gbest].pbestfit)
			gbest = i;
	}

	printf("Swarm initialization completed.\n");
	
	printf("Initial best fitness: %.10f at (%.4f, %.4f)\n", swarm[gbest].pbestfit, swarm[gbest].pbestx[0], swarm[gbest].pbestx[1]);	// Initialization Prompt

	/* Particles Update */
	for (iter = 0; iter < MAX_ITERATION; iter++)
	{
		for(i = 0; i < SWARM_SIZE; i++)
		{
			swarm[i].fitness = FitnessFunc(swarm[i].x, DIMENSION);// Particle Fitness Calculation
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

		/* (Interval) Iteration Output */
		if ((iter + 1) % 10 == 0)
		{
			printf("Iteration %3d: Bestfitness = %.10e at (%.4f, %.4f)\n", iter + 1, swarm[gbest].pbestfit, swarm[gbest].pbestx[0], swarm[gbest].pbestx[1]);
		}

		/* Iteration Output */
		fp = fopen("Fitness.data", "a");
		fprintf(fp, "%d %.10e\n", iter, swarm[gbest].pbestfit);	// Global best fitness of each iteration (convergence speed): Fitness.data
		fclose(fp);
	}

	/* Optimization Output */

	return 0;
}
