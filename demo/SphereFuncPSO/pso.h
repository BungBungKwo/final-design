#ifndef PSO
#define DIMENSION 2	// Dimensions of Problem
#define SWARM_SIZE 20	// Typical: 15-30
#define MAX_ITERATION 300	// Maximum iteration
#define INTERVAL 50	// Iteration Interval (for visiualization)

#define W 1.7	// Weight of Velocity
#define MAX_W 1.2	// Minimum weight of velocity
#define MIN_W 0.4	// Maximum weight of velocity
#define C1 2.8	// Weight of Pbest
#define C2 1.3	// Weight of Gbest

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

/* Fitness Function: Sphere function (Minimalized Version) */
double FitnessFunc(double *x, int dim)
{
	double sum = 0.0;
	for (int i = 0; i < dim; i++)
		sum += x[i] * x[i];
	return sum;
}

#endif