#include "ipCalculator.h"
#include <float.h>

typedef struct ipCacheInput {
	//This should be a constant
	const ipCache * info;
	const nnLayer *layer0;
	//This shouldn't be
	uint *key;
} ipCacheInput;

typedef struct ipCacheData {
	float *solution;
	float *projection;
} ipCacheData;

/*
 This can only handle neural networks whose input dimension is greater than the number of nodes in the first layer
*/
struct ipCacheData * solve(float *A, MKL_INT m, MKL_INT n, float *b)
{
	CBLAS_LAYOUT    layout = CblasRowMajor;
	CBLAS_TRANSPOSE noTrans = CblasNoTrans;
	CBLAS_TRANSPOSE trans = CblasTrans;
	MKL_INT lda = n;
	MKL_INT ldu = m;
	MKL_INT ldvt = n;
	MKL_INT info;
	MKL_INT minMN = ((n)>(m)?(m):(n));
	float superb[minMN-1]; 
	/* Local arrays */
	float *s = calloc(n,sizeof(float));
	float *u = calloc(ldu*m , sizeof(float));
	float *vt = calloc(ldvt*n, sizeof(float));


	/* Executable statements */
	printf( "LAPACKE_sgesvd (row-major, high-level) Example Program Results\n" );
	info = LAPACKE_sgesvd( LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda,
		    s, u, ldu, vt, ldvt, superb );
	// Incase of memory leaks:
	//mkl_thread_free_buffers();
	if( info > 0 ) {
		printf( "The algorithm computing SVD failed to converge.\n" );
		exit( 1 );
	}


	// Multiplying Sigma+t with u
	int i = 0;
	for(i=0;i<m;i++){
		cblas_sscal(m,(1/s[i]),u+i*m,1);
	}
	float * projection;
	if(n == m){
		output->projection = NULL;
	} else {
		output->projection = calloc(n*n,sizeof(float));
	}
	 
	float * solution = calloc(n, sizeof(float));
	float *c = calloc(n*m , sizeof(float));
	// Multiplying u sigma+t with vt
	cblas_sgemm (layout, noTrans, noTrans, m, n, m, 1, u, ldu, vt, ldvt, 0,c, n);
	// Multiplying v sigma+ u with b for the solution
	cblas_sgemv (layout, trans, m, n,1, c, n, b, 1, 0, solution, 1);
	// Saving the kernel basis from vt
	if(m<n){
		void cblas_sgemm (layout, trans, noTrans, m, m, n-m, 1, vt+n*m, n-m, vt+n*m, n, 0, projection, m);
	}
	free(s);
	free(u);
	free(vt);
	free(c);
	// Incase of memory leaks:
	//mkl_thread_free_buffers();

	struct ipCacheData *output = malloc(sizeof(struct ipCacheData));
	output->solution = solution;
	output->projection = projection;
	return output;
}

void * ipCacheDataCreator(void * input)
{
	struct ipCacheInput *myInput;
	myInput = (struct ipCacheInput *) input;
	nnLayer *layer0 = input->info->layer0;

	uint inDim = layer0->inDim;
	uint outDim = layer0->outDim;

	float *subA = calloc(inDim*outDim, sizeof(float));
	float *subB = calloc(outDim,sizeof(float));
	uint i = 0;
	uint numHps = 0;
	for(i=0;i<inDim;i++){
		if(checkIndex(myInput->key,i)){
			cblas_scopy (inDim, layer0->A +i*inDim, 1, subA + numHps*inDim, 1);
			subB[numHps] = layer0->b[i];
			numHps++;
		}
	}
	//If there's less included hyperplanes there will be a kernel
	if(numHps <= inDim){
		MKL_INT m = numHps;
		MKL_INT n = inDim;
		return solve(subA, m, n, subB, kernelBasis, solution);
	} else {
		return NULL;
	}
}
void ipCacheDataDestroy(void * data)
{
	struct ipCacheData *myData;
	myData = (struct ipCacheData *) data;
	if(myData){
		if(myData->kernelBasis){
			free(myData->kernelBasis);
		}
		if(myData->solution){
			free(solution);
		}
		free(myData);
	}
}

void createHPCache(ipCache *ipCache, nnLayer *layer0)
{
	
}

ipCache * allocateCache(nnLayer *layer0)
{
	ipCache *cache = malloc(sizeof(ipCache));
	cache->bases = createTree(8, ipCacheDataCreator, NULL, ipCacheDataDestroy);

	createHPCache(ipCache *ipCache, layer0);
}

void freeCache(ipCache * cache)
{

}

float computeDist(float * p, ipCacheInput * myInput, Tree * ipCache)
{
	CBLAS_LAYOUT    layout 	= CblasRowMajor;
	CBLAS_TRANSPOSE noTrans = CblasNoTrans;
	CBLAS_TRANSPOSE trans 	= CblasTrans;
	struct ipCacheData *myBasis = (struct *ipCacheData) addData(ipCache, myInput->key, myInput);;
	
	if(myData){
		MKL_INT inDim = myInput->layer->inDim;
		MKL_INT dimKernel = myBasis->dimKernel;
		float * px = malloc(myInput->layer->inDim * sizeof(float));
		cblas_scopy (inDim, p, 1, px, 1);
		cblas_saxpy (inDim,1,myBasis->solution,1,px,1);
		cblas_sgemv (layout, noTrans, inDim, inDim,-1,myData->projection, inDim, px, 1, 1, px, 1);
		float norm = cblas_snrm2 (inDim, px, 1);
		free(px);
		return norm;
	} else {
		return -1;
	}	
}

void computeDistToHPS(float *p, ipCache *cache, float *distances)
{
	uint outDim = cache->outDim;
	uint inDim = cache->inDim;
	float * localCopy = calloc(outDim*inDim,sizeof(float));
	cblas_scopy (inDim*outDim, cache->hpOffsetVecs, 1, localCopy, 1);


	for(uint i =0;i<outDim;++i){
		cblas_saxpy (inDim,1,p,1,localCopy + i*inDim,1);
		distances[i] = cblas_sdot (inDim, localCopy + i*inDim, 1, cache->hpNormals + i*inDim, 1);
	}
	free(localCopy);
}


void getInterSig(float *p, uint *ipSignature, ipCache * cache, nnLayer *layer0)
{
	#ifdef DEBUG
		cout << "Getting the intersection signature for " << endl << v;
	#endif
	uint outDim = cache->outDim;
	uint inDim = cache->inDim;
	ipCacheInput myInput = {info = info, key = ipSignature};

	float *distances = calloc(outDim,sizeof(float));
	computeDistToHPS(p, cache, distances);
	uint j = 1, k = 1;
	
	clearKey(ipSignature, uint dataLen);

	// Get the distance to the closest hyperplane and blank it from the distance array
	uint curSmallestIndex = cblas_isamin (outDim, distances, 1);
	float curDist = distances[curSmallestIndex];
	distances[curSmallestIndex] = FLT_MAX;
	addIndexToKey(ipSignature, currentSmallestIndex);

	// Get the distance to the second closest hyperplane and blank it
	curSmallestIndex = cblas_isamin (outDim, distances, 1);
	float nextDist = distances[curSmallestIndex];
	distances[curSmallestIndex] = FLT_MAX;

	while(curDist>0 && nextDist < SCALEDTUBETHRESHOLD*curDist && j < inDim)
	{	

		addIndexToKey(ipSignature, curSmallestIndex);
		// Prepare for next loop
		curSmallestIndex = cblas_isamin (outDim, distances, 1);
		// Current distance should be the distance to the current IP set
		curDist = computeDist(p, &myInput, cache);
		// Next distance should either be to the next hyperplane or the next closest IP of the same rank.
		nextDist = distances[curSmallestIndex];
		distances[curSmallestIndex] = FLT_MAX;
		j++;
	}
	free(distances);
}