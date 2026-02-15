/* Fitness Function: Schaffer f6 function (Minimalized Version) */
double FitnessFunc(double x1, double x2)
{
	double sum_sq = x1 * x1 + x2 * x2;
	double sqrt_sumsq = sqrt(sum_sq);
	double sin_sq = sin(sqrt_sumsq) * sin(sqrt_sumsq);
	double denominator = 1.0 + 0.001 * sum_sq;
	return (0.5 + (sin_sq - 0.5) / (denominator * denominator));
}

	FILE *fp = NULL;
	srand(time(NULL));

	/* Search Space Parameters */
	const double MIN_X = -100.0;
	const double MAX_X = 100.0;
	const double STEP = 0.1;
	const double MIN_V = -5;
	const double MAX_V = 5;

	/* PSO Paradigm Testing */
	printf("====== PSO Algorithm Benchmark Test: Schaffer f6 ======\n");
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
			printf("Iteration %3d: Bestfitness = %.10f at (%.4f, %.4f)\n", iter + 1, swarm[gbest].pbestfit, swarm[gbest].pbestx[0], swarm[gbest].pbestx[1]);
		}
		fprintf(fp, "%d %f\n", iter, swarm[gbest].pbestfit);
	}
	fclose(fp);

	/* Optimization Output */