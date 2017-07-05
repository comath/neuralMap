#include "ipTrace.h"


/*
Creates the data needed for the hp distance computation

Needs two float arrays of size n*m and m
*/
void fillHPCache(nnLayer *layer, float * hpNormals, float * hpOffsetVals)
{
	int i = 0;
	int n = layer->inDim;
	int m = layer->outDim;

	#ifdef DEBUG
		printf("Filling HPs\n");
	#endif
	float scaling = 1;
	for(i=0;i<m;i++){
		scaling = cblas_snrm2 (n, layer->A + n*i, 1);
		if(scaling){
			cblas_saxpy (n, 1/scaling, layer->A + n*i, 1, hpNormals + n*i,1);
			hpOffsetVals[i] = layer->b[i]/scaling;
		} else {
			// HP normals has length 0, thus it is 0. The offset vector should also be 0
			memset(hpNormals+i*n,0, n*sizeof(float));
			hpOffsetVals[i]=0;
		}
	}
}

// Orders the distWithIndex struct by the distance
int distOrderingCmp (const void * a, const void * b)
{
	distanceWithIndex *myA = (distanceWithIndex *)a;
	distanceWithIndex *myB = (distanceWithIndex *)b;
	if( ( (myB)->dist - (myA)->dist ) > 0){
		return -1;
	} else {
		return 1;
	}
}

/*
Computes the distance between a point and the hyperplanes.

Inputs: p, and the result of fillHPCache
Outputs: distances, ordered by distance (close to far)
*/
void computeDistToHPS(float *p, distanceWithIndex *distances,
						float * hpNormals, float * hpOffsetVals, int m, int n)
{	
	#ifdef DEBUG
		printf("Getting distances to local HPs\n");
	#endif
	float curDist = 0;
	for(int i = 0; i<m;i++){
		distances[i].index = i;
		curDist = cblas_sdot(n, p, 1, hpNormals + i*n, 1);
		if(curDist == 0){
			// Sometimes the normal vector is 0, ie, this hyperplane is not used
			// We check for this, and if so, essentially remove this distance from the situation.
			float norm = cblas_snrm2 (n, hpNormals + i*n, 1);
			if(norm == 0){
				distances[i].dist = FLT_MAX;
			} else {
				distances[i].dist = 0;
			}
		}
		curDist += hpOffsetVals[i];
		if(curDist < 0){
			distances[i].dist = -curDist;
		} else {
			distances[i].dist = curDist;
		}
	}
	qsort(distances, m, sizeof(distanceWithIndex), distOrderingCmp);	
}

void solveSquareSystem(int n, int m, float *A, float *b, float *x, int *pivot)
{
	memcpy(x,b,m*sizeof(float));
	int rc = LAPACKE_sgetrs (CblasRowMajor , 'N' , n, m , A , 1, pivot , x, 1);
}

/*
Creates a traceCache, creates a solution and calls fillHPCache
*/
traceCache * allocateTraceCache(nnLayer * layer)
{
	int m = layer->outDim;
	int n = layer->inDim;
	int i = 0;
	#ifdef DEBUG
		printf("Allocating traceCache, inDim: %d, rank: %d\n", n,m);
	#endif
	if(m>n){
		printf("Can Only handle layers with the number of HP is less than or equal to the dim\n");
		exit(-1);
	}
	traceCache * tc = malloc(sizeof(traceCache));
	tc->layer = layer;
	printf("In trace layer->A[0]: %f pointer %p\n", layer->A[0], layer->A);

	tc->hpNormals = calloc(m*n,sizeof(float));
	tc->hpOffsetVals = calloc(m,sizeof(float));
	fillHPCache(layer,tc->hpNormals, tc->hpOffsetVals);
	printf("after hpC layer->A[0]: %f pointer %p\n", layer->A[0], layer->A);

	float *Acopy = malloc(n*m*sizeof(float));
	memcpy(Acopy,layer->A,n*m*sizeof(float));
	float *u = malloc(m*m*sizeof(float)); //n^2
	float *vt = malloc(n*n*sizeof(float)); //m^2
	float *c = malloc(n*m*sizeof(float)); //m*n
	float *s = malloc(m*sizeof(float)); //m
	
	tc->solution = malloc(n * sizeof(float));
	if(!tc->solution){
		printf("Memory Error\n");
		exit(-1);	
	}
	const float oneF = 1.0;
	const float zeroF = 0.0;
	const char N = 'N';

	// Solves Ax=b, in many steps

	printf("just before svm layer->A[0]: %f pointer %p\n", layer->A[0], layer->A);


	// SVD first
	int info = LAPACKE_sgesdd( LAPACK_ROW_MAJOR, 'A', m, n, Acopy, n, 
							s, 
							u, m, 
							vt, n );
	// Incase of memory leaks:
	// mkl_thread_free_buffers(); Not a real leak, it's the MKL buffer.
	if( info ) {
		if(info > 0){
			printf( "The algorithm computing SVD failed to converge.\n" );
		} else {
			printf("There was an illegal value.\n");
		}
		exit( 1 );
	}
	printf("just after svm layer->A[0]: %f pointer %p\n", layer->A[0], layer->A);

	// Multiplying Sigma+t with u
	i = 0;
	while(i<m && s[i] !=0){
		cblas_sscal(m,(1/s[i]),u+i,m);
		i++;
	}
	/* Multiplies v with (sigma+*u)
	sgemm (&N,&N, 
		&n, &m, &m, 
		&oneF, vt, &n, 
		u, &m,
		&zeroF,c,&n);
	*/
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, n, m, m, 1, vt, n, u, m, 0, c, m);

	#ifdef DEBUG
		printf("U:\n");
		printMatrix(u,m,m);
		printf("Vt:\n");
		printMatrix(vt,n,n);
		printf("C:\n");
		printMatrix(c,m,n);
	#endif
	printf("just before gemv layer->A[0]: %f pointer %p\n", layer->A[0], layer->A);

	// Multiplying v sigma+ u with b for the solution				\/ param 7
	cblas_sgemv (CblasRowMajor, CblasTrans, m, n,1, c, n, layer->b, 1, 0, tc->solution, 1);

	free(u);
	free(vt);
	free(c);
	free(Acopy);

	free(s);

	

	tc->keyLen = calcKeyLen(n);
	#ifdef DEBUG
		printf("Trace Cache Allocated\n");
	#endif
	printf("End of trace layer->A[0]: %f pointer %p\n", layer->A[0], layer->A);

	return tc;
}

void freeTraceCache(traceCache * tc)
{
	if(tc){
		if(tc->solution){
			free(tc->solution);
		}
		free(tc->hpNormals);
		free(tc->hpOffsetVals);
		free(tc);
	}
}

traceMemory * allocateTraceMB(int m, int n)
{
	traceMemory * tm = malloc(sizeof(traceMemory));
	tm->tau = malloc(m*sizeof(float));
	tm->distances = malloc(m*sizeof(distanceWithIndex));
	tm->interDists = malloc(m*sizeof(float));
	tm->permA = malloc(m*n*sizeof(float));
	tm->pointBuff = malloc(n*sizeof(float));
	return tm;
}

void freeTraceMB(traceMemory * tm)
{
	if(tm){
		if(tm->tau){free(tm->tau);}
		if(tm->distances){free(tm->distances);}
		if(tm->interDists){free(tm->interDists);}
		if(tm->permA){free(tm->permA);}
		if(tm->pointBuff){free(tm->pointBuff);}
		free(tm);
	}
}



void fillPermMatrix(distanceWithIndex *distances, nnLayer *layer, float *permA)
{
	int n = layer->inDim;
	int m = layer->outDim;

	for(int i=0;i<m;i++){
		memcpy(permA + i*n, layer->A+(distances[i].index)*n, n*sizeof(float));
	}
}


/* 
Workhorse of this file. All external computational calls, eventually call this and repack the return.
We compute Q with Gram Schmidt, which is orthogonal, so we can do this sequential projection onto Q
This will produce the projection onto each subspace in the flag of the rearanged matrix
The flag of that rearanged matrix is the trace of the point through the intersection poset
Thus we get the distances for the trace in ~ O(mn^2-(1/3)m^3)

The interDists are the distances to the closest rank i intersection for i in (1,... ,n)

Input: point, tc, tm,
Output: distances, interDists
*/
void fullTraceWithDist(traceCache * tc, traceMemory * tm, float * point, distanceWithIndex * distances, float * interDists)
{
	int m = tc->layer->outDim;
	int n = tc->layer->inDim;
	
	#ifdef DEBUG
		printf("--------------------------------------------------\n");
		printf("Taking distace to point with pointer %p\n", point);
		if(n<10){
			printf("Point:");
			printFloatArr(point,n);
		}
	#endif

	computeDistToHPS(point,distances,tc->hpNormals,tc->hpOffsetVals,m,n);
	fillPermMatrix(distances,tc->layer,tm->permA);
	#ifdef DEBUG
		if(n<10 && m<10){
			printf("Distances to HPs: [");
			for(int i=0;i<m-1;i++){
				printf("%d:%f, ", distances[i].index,distances[i].dist);
			}
			printf("%d:%f]\n", distances[m-1].index,distances[m-1].dist);
			printf("HPs rearranged:\n");
			printMatrix(tm->permA,n,m);
		}
	#endif
	// Does QR decomp. Creates R and the pivot indexes for Q. 
	int rc = LAPACKE_sgeqrf (LAPACK_COL_MAJOR, n, m, tm->permA, n, tm->tau);
	if(rc){
		printf("QR failed, maybe inDim=%d and outDim=%d? illegal parameter at index: %d with val of %f\n",n,m, -rc, tm->permA[-rc]);
		//printMatrix(tc->layer->A,n,m);
		exit(-1);
	}
	// Unpacks Q from R and the pivot indexes
	rc = LAPACKE_sorgqr(LAPACK_COL_MAJOR, n, m, m, tm->permA, n, tm->tau);
	if(rc){
		printf("Getting Q failed\n");
		exit(-1);
	}
	#ifdef DEBUG
		if(n<10 && m<10){
			printf("Orthogonalized HPs (for the flag):\n");
			printMatrix(tm->permA,n,m);
		}
	#endif
	memcpy(tm->pointBuff,point,n*sizeof(float));
	cblas_saxpy (n,-1,tc->solution,1,tm->pointBuff,1);
	#ifdef DEBUG
		if(n<10){
			printf("Point minus solution:");
			printFloatArr(tm->pointBuff,n);
			printf("Norm:%f\n",cblas_snrm2(n,tm->pointBuff,1));
		}
	#endif

	// Final computation of trace distances
	float curProj = 0;
	float dot = 0;
	for(int i = 0; i< m;i++){
		dot = cblas_sdot (n, tm->permA + i*n, 1, tm->pointBuff, 1);
		curProj += dot*dot;
		interDists[i] = sqrtf(curProj);
		
		#ifdef DEBUG
			if(n<10){
				printf("Distance to the closest rank %d intersection: %f\n",i+1,interDists[i]);
			}
		#endif
	}
	#ifdef DEBUG
		printf("------------------------------------------------\n");
	#endif
}

void fullTrace(traceCache * tc, traceMemory * tm, float * point, float * dists, kint * ipSigTrace){
	int m = tc->layer->outDim;
	uint keyLen = tc->keyLen;
	fullTraceWithDist(tc, tm, point, tm->distances, dists);
	memset(ipSigTrace,0,m*keyLen*sizeof(kint));
	#ifdef DEBUG
		printf("====================================\n");
	#endif
	for(int i = 0; i< m-1;i++){
		addIndexToKey(ipSigTrace + i*keyLen, tm->distances[i].index);
		memcpy(ipSigTrace+i*keyLen,ipSigTrace+(i+1)*keyLen,keyLen);
		#ifdef DEBUG
			printf("=======%d=======\n",i);
			printf("Current ip");
			printKey(ipSigTrace + i*keyLen,m);
		#endif
	}
	addIndexToKey(ipSigTrace + (m-1)*keyLen,tm->distances[m-1].index);
	#ifdef DEBUG
		printf("=======%d=======\n",m-1);
		printf("Current ip");
		printKey(ipSigTrace + (m-1)*keyLen,m);
		printf("====================================\n");
	#endif
}


/* 
Computes the associated intersection from the outputs fullTraceWithDist with the given threshold. See paper for explanation of workings. 

Used in the other packing functions, internal. 
*/
void getIntersection(float *interDists, distanceWithIndex *dists, int m, kint * ipSig, float threshold)
{
	int i = 0;
	int keyLen = calcKeyLen(m);
	memset(ipSig,0,keyLen*sizeof(kint));
	for(i = 0; i< m-1;i++){
		addIndexToKey(ipSig, dists[i].index);
	}
	i = m-1;
	#ifdef DEBUG
		printf("====================================\n");
		printf("Current ip");
		printKey(ipSig,m);
		printf("Distance to closest rank %d intersection %f\n",i, interDists[i-1]);
		printf("Distance to %d closest hyperplane: %d:%f\n",i,dists[i].index, dists[i].dist);
		if(threshold*interDists[i-1] - dists[i].dist > 0){
			printf("Removing %d from key\n",dists[i-1].index);
		} 
	#endif
	
	while(i > 0 && threshold*interDists[i-1] - dists[i].dist > 0){
		removeIndexFromKey(ipSig,dists[i-1].index);
		#ifdef DEBUG
			printf("=======%d=======\n",i);
			printf("Current ip");
			printKey(ipSig,m);
			printf("Distance to closest rank %d intersection %f\n",i, interDists[i-1]);
			printf("Distance to %d closest hyperplane: %f\n",i, dists[i].dist);
			if(threshold*interDists[i-1] - dists[i].dist > 0){
				printf("Removing %d from key\n",dists[i-1].index);
			} 
		#endif
		i--;
		
	}
	#ifdef DEBUG
		printf("=======%d=======\n",i);
		printf("Final ip");
		printKey(ipSig,m);
		printf("====================================\n");
	#endif
}


void ipCalc(traceCache * tc, traceMemory * tm, float * point, float threshold, kint *ipSig)
{
	fullTraceWithDist(tc, tm, point, tm->distances, tm->interDists);
	getIntersection(tm->interDists,tm->distances, tc->layer->outDim, ipSig, threshold);
}

void bothIPCalcTrace(traceCache *tc, traceMemory *tm, float *point,float threshold, kint *ipSig, float *traceDists, int * traceRaw)
{
	fullTraceWithDist(tc, tm, point, tm->distances, traceDists);
	getIntersection(traceDists,tm->distances, tc->layer->outDim, ipSig, threshold);
	for(uint i = 0; i < tc->layer->outDim; i++){
		traceRaw[i] = tm->distances[i].index;
	}
}

struct traceThreadArgs {
	uint tid;
	uint numThreads;

	uint numData;
	float * data;
	kint *ipSigTraces;
	float *dists;

	traceCache *tc;
	traceMemory *tm;
};

void * fullTraceBatch_thread(void *thread_args)
{
	struct traceThreadArgs *myargs;
	myargs = (struct traceThreadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;

	uint numData = myargs->numData;

	traceCache *tc = myargs->tc;

	uint n = tc->layer->inDim;
	uint m = tc->layer->outDim;
	uint keySize = calcKeyLen(m);
	traceMemory * tm = allocateTraceMB(m,n);
	uint i = 0;
	for(i=tid;i<numData;i=i+numThreads){
		fullTrace(myargs->tc, tm, myargs->data+i*n, myargs->dists + i*m,  myargs->ipSigTraces+i*m*keySize);
		//printf("Thread %d at mutex with %u nodes \n",tid,tc->bases->numNodes);	
	}
	freeTraceMB(tm);
	pthread_exit(NULL);
}

void batchFullTrace(traceCache * tc, float * data, float * dists, kint * ipSigTraces, int numData, int numProc){

	int maxThreads = numProc;
	int rc =0;
	int i =0;
	printf("ipCalc working on %u data points, with %u threads\n",numData,maxThreads);
	//Add one data to the first node so that we can avoid the race condition.
	

	struct traceThreadArgs *thread_args = malloc(maxThreads*sizeof(struct traceThreadArgs));
	
	pthread_t threads[maxThreads];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	for(i=0;i<maxThreads;i++){
		thread_args[i].tc = tc;
		thread_args[i].numData = numData;
		thread_args[i].ipSigTraces = ipSigTraces;
		thread_args[i].data = data;
		thread_args[i].dists = dists;
		thread_args[i].numThreads = maxThreads;
		thread_args[i].tid = i;
		rc = pthread_create(&threads[i], NULL, fullTraceBatch_thread, (void *)&thread_args[i]);
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

struct ipCalcThreadArgs {
	uint tid;
	uint numThreads;

	uint numData;
	float * data;
	kint *ipSigs;
	float threshold;

	traceCache *tc;
	traceMemory *tm;
};

void * ipCalcBatch_thread(void *thread_args)
{
	struct ipCalcThreadArgs *myargs;
	myargs = (struct ipCalcThreadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;

	uint numData = myargs->numData;

	traceCache *tc = myargs->tc;

	uint n = tc->layer->inDim;
	uint m = tc->layer->outDim;
	uint keySize = calcKeyLen(m);
	traceMemory * tm = allocateTraceMB(m,n);
	uint i = 0;
	for(i=tid;i<numData;i=i+numThreads){
		ipCalc(myargs->tc, tm, myargs->data+i*n, myargs->threshold, myargs->ipSigs+i*keySize);
		//printf("Thread %d at mutex with %u nodes \n",tid,tc->bases->numNodes);	
	}
	freeTraceMB(tm);
	pthread_exit(NULL);
}

void batchIpCalc(traceCache * tc, float * data, float threshold, kint * ipSigs, int numData, int numProc){

	int maxThreads = numProc;
	int rc =0;
	int i =0;
	printf("ipCalc working on %u data points, with %u threads\n",numData,maxThreads);
	//Add one data to the first node so that we can avoid the race condition.
	

	struct ipCalcThreadArgs *thread_args = malloc(maxThreads*sizeof(struct ipCalcThreadArgs));
	
	pthread_t threads[maxThreads];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	for(i=0;i<maxThreads;i++){
		thread_args[i].tc = tc;
		thread_args[i].numData = numData;
		thread_args[i].ipSigs = ipSigs;
		thread_args[i].data = data;
		thread_args[i].numThreads = maxThreads;
		thread_args[i].tid = i;
		thread_args[i].threshold = threshold;
		rc = pthread_create(&threads[i], NULL, ipCalcBatch_thread, (void *)&thread_args[i]);
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