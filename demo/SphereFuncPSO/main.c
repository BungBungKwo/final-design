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

	/* PSO Parameter Settings */
	const int SWARM_SIZE[] = {20, 50, 70, 100, 120};
	/*
	const int MAX_ITERATION[];
	const double W[];
	const double C1[];
	const double C2[];
	*/

	int para_loop = sizeof(SWARM_SIZE)/sizeof(SWARM_SIZE[0]);	// Parameters-var Loop

	double fitness_record[MAX_ITERATION][para_loop];

	for (int loop = 0; loop < para_loop; loop++)
	{
		int current_swarm_size = SWARM_SIZE[loop];

		/* PSO Paradigm Testing Prompt */
		printf("====== PSO Algorithm Test: Sphere Function ======\n");
		printf("Test starting...\n");

		/* Initialization of Swarm */
		printf("Swarm initialization starting...\n");
		Particle swarm[current_swarm_size];	// Swarm (Particles)
		int i, j, iter;
		int gbest = 0;	// Index of global best
		for (i = 0; i < current_swarm_size; i++)
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
			for(i = 0; i < current_swarm_size; i++)
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

			fitness_record[iter][loop] = swarm[gbest].pbestfit;	// Gbest fitness of each iteration

			/* (Interval) Iteration Output
			if ((iter + 1) % 10 == 0)
			{
				printf("Iteration %3d: Bestfitness = %.10e at (%.4f, %.4f)\n", iter + 1, swarm[gbest].pbestfit, swarm[gbest].pbestx[0], swarm[gbest].pbestx[1]);
			}
			*/

			/* Iteration Output 
			fp = fopen("Fitness.data", "a");
			fprintf(fp, "%d %.10e\n", iter, swarm[gbest].pbestfit);	// Global best fitness of each iteration (convergence speed): Fitness.data
			fclose(fp);
			*/
		}
	}

	/* Optimization Output */
	fp = fopen("Fitness.data", "w");
	// Header
	fprintf(fp, "#Iteration");
	for(int i = 0; i < para_loop; i++)
	{
		fprintf(fp,"\t%d", SWARM_SIZE[i]);
	}
	fprintf(fp,"\n");
	// Data
	for (int i = 0; i < MAX_ITERATION; i++)
	{
		fprintf(fp, "%d", i);
		for (int j = 0; j < para_loop; j++)
			fprintf(fp, " %.10e", fitness_record[i][j]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	return 0;
}
