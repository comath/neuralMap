#include "linearAlgebra.h"

int main(int argc, char* argv[])
{
	float v[5];
	float u[5];
	v[0] = 3; v[1] = 4; v[2] = 1.5; v[3] = 2; v[4] = 6;
	u[0] = 3; u[1] = 4; u[2] = 1.5; u[3] = 2; u[4] = 6;

	printf("%f\n", c_dot(v, u, 5));
}