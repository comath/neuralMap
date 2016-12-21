#include <stdio.h>
#include <stdlib.h>
#include "../utils/linearAlgebra.h"



void c_printMatrix(double * arr, int m, int n){
	int i = 0,j =0;
	int index = 0;
	for(i=0;i<m;i++){
		printf("[");
		for(j=0;j<n-1;j++){
			printf("%f,",arr[index]);
			index++;
		}
		printf("%f", arr[index]);
		index++;
		printf("]\n");
	}
	return ;
}

float c_dot(float *v, float *u, int n)
{
	return dot(v,u,n);
}

