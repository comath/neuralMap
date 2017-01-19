#include "ipCalculator.h"
#include <float.h>

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>



typedef struct ipCacheInput {
	//This should be a constant
	const ipCache * info;
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
	struct ipCacheData *output = malloc(sizeof(struct ipCacheData));


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

	 
	float * solution = calloc(n, sizeof(float));
	float *c = calloc(n*m , sizeof(float));
	// Multiplying u sigma+t with vt
	cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, m, 1, u, ldu, vt, ldvt, 0,c, n);
	// Multiplying v sigma+ u with b for the solution
	cblas_sgemv (CblasRowMajor, CblasTrans, m, n,1, c, n, b, 1, 0, solution, 1);
	// Saving the kernel basis from vt
	if(m<n){
		output->projection = calloc(n*n,sizeof(float));
		cblas_sgemm (CblasRowMajor, CblasTrans, CblasNoTrans, m, m, (n-m), 1, vt+n*m, (n-m), vt+n*m, n, 0, output->projection, m);
	} else {
		output->projection = NULL;
	}
	free(s);
	free(u);
	free(vt);
	free(c);
	// Incase of memory leaks:
	//mkl_thread_free_buffers();

	
	output->solution = solution;
	return output;
}

void * ipCacheDataCreator(void * input)
{
	struct ipCacheInput *myInput;
	myInput = (struct ipCacheInput *) input;
	nnLayer *layer0 = myInput->info->layer0;

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
		return solve(subA, m, n, subB);
	} else {
		return NULL;
	}
}
void ipCacheDataDestroy(void * data)
{
	struct ipCacheData *myData;
	myData = (struct ipCacheData *) data;
	if(myData){
		if(myData->projection){
			free(myData->projection);
		}
		if(myData->solution){
			free(myData->solution);
		}
		free(myData);
	}
}

void createHPCache(ipCache *cache)
{
	#ifdef DEBUG
		printf("Setting up hyperplanes\n");
	#endif
	uint i = 0;
	nnLayer *layer0 = cache->layer0;
	uint inDim = layer0->inDim;
	uint outDim = layer0->outDim;

	cache->hpNormals = calloc(outDim*inDim,sizeof(float));
	cache->hpOffsetVecs = calloc(outDim*inDim,sizeof(float));

	float scaling = 1;
	for(i=0;i<outDim;i++){
		scaling = cblas_snrm2 (inDim, layer0->A + inDim*i, 1);
		cblas_saxpy (inDim,1/scaling,layer0->A+inDim*i,0,cache->hpNormals + inDim*i,1);
		cblas_saxpy (inDim,layer0->b[i]/scaling,cache->hpNormals+inDim*i,0,cache->hpOffsetVecs + inDim*i,1);
	}
}

ipCache * allocateCache(nnLayer *layer0, float threshhold)
{
	ipCache *cache = malloc(sizeof(ipCache));
	uint keyLen = calcKeyLen(layer0->outDim);
	cache->bases = createTree(8,keyLen , ipCacheDataCreator, NULL, ipCacheDataDestroy);

	printf("Creating cache of inDim %d and outDim %d with threshhold %f \n", layer0->inDim, layer0->outDim, threshhold);
	cache->layer0 = layer0;
	createHPCache(cache);
	cache->threshold = threshhold;
	return cache;
}

void freeCache(ipCache * cache)
{
	if(cache){
		printf("Cache exists, deallocating\n");
		freeTree(cache->bases);
		if(cache->hpOffsetVecs){
			free(cache->hpOffsetVecs);
		}
		if(cache->hpNormals){
			free(cache->hpNormals);
		}
		free(cache);
	}
}


float computeDist(float * p, uint *ipSignature, ipCache *cache)
{

	ipCacheInput myInput = {.info = cache, .key = ipSignature};
	struct ipCacheData *myBasis = addData(cache->bases, ipSignature, &myInput);;
	
	if(myBasis){
		MKL_INT inDim = cache->layer0->inDim;
		float * px = malloc(inDim * sizeof(float));
		cblas_scopy (inDim, p, 1, px, 1);
		cblas_saxpy (inDim,1,myBasis->solution,1,px,1);
		cblas_sgemv (CblasRowMajor, CblasNoTrans, inDim, inDim,-1,myBasis->projection, inDim, px, 1, 1, px, 1);
		float norm = cblas_snrm2 (inDim, px, 1);
		free(px);
		return norm;
	} else {
		return -1;
	}	
}

void computeDistToHPS(float *p, ipCache *cache, float *distances)
{
	uint outDim = cache->layer0->outDim;
	uint inDim = cache->layer0->inDim;
	float * localCopy = calloc(outDim*inDim,sizeof(float));
	cblas_scopy (inDim*outDim, cache->hpOffsetVecs, 1, localCopy, 1);


	for(uint i =0;i<outDim;++i){
		cblas_saxpy (inDim,1,p,1,localCopy + i*inDim,1);
		distances[i] = cblas_sdot (inDim, localCopy + i*inDim, 1, cache->hpNormals + i*inDim, 1);
	}
	free(localCopy);
}


void getInterSig(ipCache * cache, float *p, uint *ipSignature)
{
	uint outDim = cache->layer0->outDim;
	uint inDim = cache->layer0->inDim;
	
	float *distances = calloc(outDim,sizeof(float));
	computeDistToHPS(p, cache, distances);
	uint j = 1;
	
	clearKey(ipSignature,cache->bases->keyLength);

	// Get the distance to the closest hyperplane and blank it from the distance array
	uint curSmallestIndex = cblas_isamin (outDim, distances, 1);
	float curDist = distances[curSmallestIndex];
	distances[curSmallestIndex] = FLT_MAX;
	addIndexToKey(ipSignature, curSmallestIndex);

	// Get the distance to the second closest hyperplane and blank it
	curSmallestIndex = cblas_isamin (outDim, distances, 1);
	float nextDist = distances[curSmallestIndex];
	distances[curSmallestIndex] = FLT_MAX;

	while(curDist>0 && nextDist < cache->threshold*curDist && j < inDim)
	{	

		addIndexToKey(ipSignature, curSmallestIndex);
		// Prepare for next loop
		curSmallestIndex = cblas_isamin (outDim, distances, 1);
		// Current distance should be the distance to the current IP set
		curDist = computeDist(p, ipSignature, cache);
		// Next distance should either be to the next hyperplane or the next closest IP of the same rank.
		nextDist = distances[curSmallestIndex];
		distances[curSmallestIndex] = FLT_MAX;
		j++;
	}
	free(distances);
}

struct dataAddThreadArgs {
	int tid;
	int numThreads;

	uint numData;
	float * data;
	uint *ipSignature;

	ipCache *cache;
};

void * addBatch_thread(void *thread_args)
{
	struct dataAddThreadArgs *myargs;
	myargs = (struct dataAddThreadArgs *) thread_args;

	int tid = myargs->tid;	
	int numThreads = myargs->numThreads;

	uint numData = myargs->numData;
	float *data = myargs->data;
	uint *ipSignature = myargs->ipSignature;

	ipCache *cache = myargs->cache;

	uint dim = cache->layer0->inDim;
	uint keySize = calcKeyLen(cache->layer0->outDim);

	int i = 0;
	for(i=tid;i<numData;i=i+numThreads){
		getInterSig(data+i*dim, ipSignature+i*keySize, cache);
		
	}
	pthread_exit(NULL);
}

void getInterSigBatch(ipCache *cache, float *data, uint *ipSignature, uint numData, uint numProc)
{
	int maxThreads = numProc;
	int rc =0;
	int i =0;

	//Add one data to the first node so that we can avoid the race condition.
	

	struct dataAddThreadArgs *thread_args = malloc(maxThreads*sizeof(struct dataAddThreadArgs));

	pthread_t threads[maxThreads];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	for(i=0;i<maxThreads;i++){
		thread_args[i].cache = cache;
		thread_args[i].numData = numData;
		thread_args[i].ipSignature = ipSignature;
		thread_args[i].data = data;
		thread_args[i].numThreads = maxThreads;
		thread_args[i].tid = i;
		rc = pthread_create(&threads[i], NULL, addBatch_thread, (void *)&thread_args[i]);
		if (rc){
			printf("Error, unable to create thread\n");
			exit(-1);
		}
	}

	for( i=0; i < maxThreads; i++ ){
		rc = pthread_join(threads[i], &status);
		if (rc){
			printf("Error, unable to join: %d \n", rc);
			exit(-1);
     	}
	}

	free(thread_args);
}