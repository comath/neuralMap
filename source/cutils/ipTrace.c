#include "ipTrace.h"

void fillHPCache(nnLayer *layer, float * hpNormals, float * hpOffsetVals)
{
	int i = 0;
	int n = layer->inDim;
	int m = layer->outDim;

	printMatrix(layer->A,layer->inDim,layer->outDim);
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

void computeDistToHPS(float *p, 
						float * hpNormals, float * hpOffsetVals, int m, int n, 
						distanceWithIndex *distances)
{	
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
	
traceCache * allocateTraceCache(nnLayer * layer)
{
	int m = layer->outDim;
	int n = layer->inDim;
	int i = 0;

	if(m>n){
		printf("Can Only handle layers with the number of HP is less than or equal to the dim\n");
		exit(-1);
	}
	traceCache * tc = malloc(sizeof(traceCache));
	if(!tc){
		printf("Memory Error\n");
		exit(-1);
	}
	tc->layer = layer;

	tc->hpNormals = calloc(m*n,sizeof(float));
	tc->hpOffsetVals = calloc(m,sizeof(float));
	fillHPCache(layer,tc->hpNormals, tc->hpOffsetVals);
	
	float *u = malloc(m*m*sizeof(float)); //n^2
	if(!u){
		printf("Memory Error\n");
		exit(-1);	
	}
	float *vt = malloc(n*n*sizeof(float)); //m^2
	if(!vt){
		printf("Memory Error\n");
		exit(-1);	
	}
	float *c = malloc(n*m*sizeof(float)); //m*n
	if(!c){
		printf("Memory Error\n");
		exit(-1);	
	}
	float *s = malloc(m*sizeof(float)); //m
	if(!s){
		printf("Memory Error\n");
		exit(-1);	
	}
	
	
	tc->solution = malloc(n * sizeof(float));
	if(!tc->solution){
		printf("Memory Error\n");
		exit(-1);	
	}
	const float oneF = 1.0;
	const float zeroF = 0.0;
	const char N = 'N';

	MKL_INT info;
	info = LAPACKE_sgesdd( LAPACK_ROW_MAJOR, 'A', m, n, layer->A, n, 
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
	// Multiplying Sigma+t with u
	i = 0;
	while(i<m && s[i] !=0){
		cblas_sscal(m,(1/s[i]),u+i,m);
		i++;
	}
	sgemm (&N,&N, 
		&n, &m, &m, 
		&oneF, vt, &n, 
		u, &m,
		&zeroF,c,&n);
	
	// Multiplying v sigma+ u with b for the solution				\/ param 7
	cblas_sgemv (CblasRowMajor, CblasTrans, m, n,1, c, n, layer->b, 1, 0, tc->solution, 1);
	free(u);
	free(vt);
	free(c);
	free(s);

	

	tc->keyLen = calcKeyLen(n);
	
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
	if(!tm){
		printf("Memory Error\n");
		exit(-1);
	}
	tm->tau = malloc(m*sizeof(float));
	if(!tm->tau){
		printf("Memory Error\n");
		exit(-1);
	}
	tm->distances = malloc(m*sizeof(distanceWithIndex));
	if(!tm->distances){
		printf("Memory Error\n");
		exit(-1);
	}
	tm->interDists = malloc(m*sizeof(float));
	if(!tm->interDists){
		printf("Memory Error\n");
		exit(-1);
	}
	tm->permA = malloc(m*n*sizeof(float));
	if(!tm->permA){
		printf("Memory Error\n");
		exit(-1);
	}
	tm->pointBuff = malloc(n*sizeof(float));
	if(!tm->pointBuff){
		printf("Memory Error\n");
		exit(-1);
	}
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

void fullTraceWithDist(traceCache * tc, traceMemory * tm, float * point, distanceWithIndex * distances, float * interDists){
	int m = tc->layer->outDim;
	int n = tc->layer->inDim;
	
	computeDistToHPS(point,tc->hpNormals,tc->hpOffsetVals,m,n,distances);
	fillPermMatrix(distances,tc->layer,tm->permA);
	int rc = LAPACKE_sgeqrf (LAPACK_COL_MAJOR, m, m, tm->permA, n, tm->tau);
	if(rc){
		printf("QR failed\n");
		exit(-1);
	}
	rc = LAPACKE_sorgqr(LAPACK_COL_MAJOR, m, m, m, tm->permA, n, tm->tau);
	if(rc){
		printf("Getting Q failed\n");
		exit(-1);
	}
	memcpy(tm->pointBuff,point,n*sizeof(float));
	cblas_saxpy (n,-1,tc->solution,1,tm->pointBuff,1);
	float curProj = 0;
	for(int i = 0; i< m;i++){
		curProj += cblas_sdot (n, tm->permA + i*n, 1, tm->pointBuff, 1);
		if(curProj < 0){
			interDists[i] = -curProj;
		} else {
			interDists[i] = curProj;
		}
	}
}

void fullTrace(traceCache * tc, traceMemory * tm, float * point, float * dists, kint * ipSigTrace){
	int m = tc->layer->outDim;
	uint keyLen = tc->keyLen;
	fullTraceWithDist(tc, tm, point, tm->distances, dists);
	memset(ipSigTrace,0,m*keyLen*sizeof(kint));
	for(int i = 0; i< m-1;i++){
		addIndexToKey(ipSigTrace + i*keyLen, tm->distances[i].index);
		memcpy(ipSigTrace+i*keyLen,ipSigTrace+(i+1)*keyLen,keyLen);
	}
	addIndexToKey(ipSigTrace + (m-1)*keyLen,tm->distances[m-1].index);
}

void ipCalc(traceCache * tc, traceMemory * tm, float * point, kint *ipSig, float threshold)
{
	int i =0;
	int m = tc->layer->outDim;
	uint keyLen = tc->keyLen;
	fullTraceWithDist(tc, tm, point, tm->distances, tm->interDists);
	memset(ipSig,0,m*keyLen*sizeof(kint));
	for(i = 0; i< m-1;i++){
		addIndexToKey(ipSig, tm->distances[i].index);
	}
	i = m-1;
	while(threshold*tm->interDists[i-1] - tm->distances[i].dist > 0 && i > 1){
		i--;
		removeIndexFromKey(ipSig,tm->distances[i].index);
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
		ipCalc(myargs->tc, tm, myargs->data+i*n, myargs->ipSigs+i*keySize, myargs->threshold);
		//printf("Thread %d at mutex with %u nodes \n",tid,tc->bases->numNodes);	
	}
	freeTraceMB(tm);
	pthread_exit(NULL);
}

void batchIpCalc(traceCache * tc, float * data, kint * ipSigs, float threshold, int numData, int numProc){

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