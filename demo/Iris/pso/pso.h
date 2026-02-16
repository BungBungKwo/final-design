/* PSO Parameters Setting */
#ifndef PSO_H
#define PSO_H

#include "../cfg/cfg.h"

#define SWARM_SIZE 30	// Typical: 15-30
#define MAX_ITERATION 300	// Maximum iteration

#define W 0.5	// Weight of Velocity
#define C1 1.5	// Weight of Pbest
#define C2 1.5	// Weight of Gbest

typedef struct {
	double x[DIMENSION];	// Position of Particle
	double v[DIMENSION];	// Velocity of Particle
	double fitness;		// Fitness of Particle
	double pbestfit;		// Best Fitness of Particle
	double pbestx[DIMENSION]; // Position correspond to cornfield vector
} Particle;

double FitnessFunc(double x1, double x2);

#endif