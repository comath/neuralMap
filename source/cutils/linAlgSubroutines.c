#include "structDefs.h"
#include "linAlgSubroutines.h"
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
/*
 This can only handle neural networks whose input dimension is greater than the number of nodes in the first layer
*/
struct ipCacheData * solve(ipMemory *mb)
{	
	MKL_INT outDim = mb->numHps;
	MKL_INT inDim = mb->inDim;
	#ifdef DEBUG
		printf("------------------Solve--------------\n");
		printf("The dimensions are inDim: %d, outDim: %d \n", inDim, outDim);
	#endif
	MKL_INT info;
	MKL_INT minMN = ((inDim)>(outDim)?(outDim):(inDim));
	/* Local arrays */
	
	struct ipCacheData *output = malloc(sizeof(struct ipCacheData));
	if(outDim<inDim){
		output->solution = malloc(inDim*(inDim+1)*sizeof(float));
		output->projection = output->projection + inDim;
	} else {
		output->solution = malloc(inDim*sizeof(float));
		output->projection = NULL;
	}
	/* Standard SVD, not the new type */
	/*
	info = LAPACKE_sgesvd( LAPACK_ROW_MAJOR, 'A', 'A',
						  outDim, inDim, A, inDim,
		    			  s, u, outDim,
		    			  vt, inDim,
		    			  superb);
	*/
	info = LAPACKE_sgesdd( LAPACK_ROW_MAJOR, 'A', outDim, inDim, mb->subA, inDim, 
							mb->s, 
							mb->u, outDim, 
							mb->vt, inDim );
	// Incase of memory leaks:
	//mkl_thread_free_buffers();
	if( info ) {
		if(info > 0){
			printf( "The algorithm computing SVD failed to converge.\n" );
		} else {
			printf("There was an illegal value.\n");
		}
		exit( 1 );
	}


	// Multiplying Sigma+t with u
	int i = 0;
	for(i=0;i<outDim;i++){
		cblas_sscal(outDim,(1/mb->s[i]),mb->u+i*outDim,1);
	}
		 
	
	#ifdef DEBUG
		if(inDim < 10 && outDim < 20){
			printf("U:\n");
			printMatrix(mb->u,outDim,outDim);
			printf("Diagonal entries of S:\n");
			printFloatArr(mb->s,minMN);
			printf("V^T:\n");
			printMatrix(mb->vt,inDim,inDim);
		}
		printf("Multiplying u sigma+t with vt\n");
	#endif
	// Multiplying u sigma+t with vt
	cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans,
				outDim, inDim, outDim, 1, mb->u,
				outDim, mb->vt, inDim,
				0,mb->c, inDim);
	#ifdef DEBUG
		if(inDim < 10 && outDim < 20){
			printf("Result of the previous multiplication:\n");
			printMatrix(mb->c,inDim,outDim);
			printf("b:");
			printFloatArr(mb->subB,outDim);
		}
		printf("Multiplying v sigma+ u with b for the solution \n");
		
	#endif
	// Multiplying v sigma+ u with b for the solution				\/ param 7
	cblas_sgemv (CblasRowMajor, CblasTrans, outDim, inDim,1, mb->c, inDim, mb->subB, 1, 0, output->solution, 1);
	#ifdef DEBUG
		printf("Result of the previous multiplication:\n");
		if(inDim < 10){
			printFloatArr(output->solution,inDim);
		}
	#endif
	// Saving the kernel basis from vt

	if(outDim<inDim){
		#ifdef DEBUG
			printf("Multiplying the first %d rows of vt for the projection\n", outDim);
		#endif

		cblas_sgemm (CblasRowMajor, CblasTrans, CblasNoTrans,
					outDim, outDim, (inDim-outDim), 1, mb->vt+inDim*outDim,outDim, 
					mb->vt+inDim*outDim, inDim,
					0, output->projection, outDim);
	} 
	// Incase of memory leaks:
	// mkl_thread_free_buffers();
	#ifdef DEBUG
		printf("-----------------/Solve--------------\n");
	#endif
	return output;
}







void printFloatArr(float *arr, uint length){
	uint i = 0;
	printf("[");
	for(i=0;i<length-1;i++){
		if(arr[i] == FLT_MAX){
			printf("---,");
		} else {
			printf("%f,",arr[i]);
		}
		
	}	
	if(arr[length-1] == FLT_MAX){
		printf("---,");
	} else {
		printf("%f",arr[length-1]);
	}
	printf("]\n");
}

void printMatrix(float *arr, uint inDim, uint outDim){
	uint i = 0;
	for(i=0;i<outDim;i++){
		printFloatArr(arr + inDim*i,inDim);
	}
}