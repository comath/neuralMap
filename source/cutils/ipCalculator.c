#include "ipCalculator.h"
#include <float.h>


typedef struct ipMemory {
	float *s;  // Diagonal entries
	float *u; 
	float *vt; 
	float *c; 
	float *distances; 
	float *distancesForSorting; 
	float *localCopy;
	float *superb; 
	float *px; 
	float *subA;
	float * subB;
	MKL_INT info;
	MKL_INT minMN;
	int numHps;
	int *hpDistIndexList;
	
} ipMemory;

ipMemory * allocateIPMemory(int inDim, int outDim)
{
	int m = inDim;
	int n = outDim;
	int numRelevantHP = 0;
	int minMN = 0;
	if(m < n){
		/* 
		If the spacial dim is less than the number of hyperplanes then we want to 
		take the distance to the closest instersection (which will be of full rank).
		*/
		numRelevantHP = m + 1;
		minMN = m;
	} else {
		numRelevantHP = n;
		minMN = n;
	}
	ipMemory *mb = malloc(sizeof(ipMemory));
	// Each pointer is followed by the number of floats it has all to itself.
	float * mainBuffer = malloc((2*m+2*n+n*n+m*m+2*n*m+minMN+(m+1)*numRelevantHP)*sizeof(float));
	mb->s = mainBuffer+0; //m
	mb->px = mainBuffer+ m; //m
	mb->distances = mainBuffer+ m; // n
	mb->distancesForSorting = mainBuffer + n; //n
	mb->u = mainBuffer+ n; //n^2
	mb->vt = mainBuffer+ n*n; //m^2
	mb->c = mainBuffer+ m*m; //m*n
	mb->localCopy =  mainBuffer+ m*n; //m*n
	mb->superb = mainBuffer+ m*n; // minMN
	mb->subA = mainBuffer+ minMN; // m*numRelevantHP
	mb->subB = mainBuffer+ m*numRelevantHP; // numRelevantHP
	mb->hpDistIndexList = malloc(numRelevantHP*sizeof(uint));
	return mb;
}

void freeIPMemory(ipMemory *mb)
{
	free(mb->s);
	free(mb->hpDistIndexList);
}

typedef struct ipCacheInput {
	//This should be a constant
	const ipCache * cache;
	ipMemory *mb;
	//This shouldn't be
	kint *key;
} ipCacheInput;

typedef struct ipCacheData {
	float *solution;
	float *projection;
} ipCacheData;

/*
 This can only handle neural networks whose input dimension is greater than the number of nodes in the first layer
*/
struct ipCacheData * solve(float *A, MKL_INT outDim, MKL_INT inDim, float *b, ipMemory *mb)
{	
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
	info = LAPACKE_sgesdd( LAPACK_ROW_MAJOR, 'A', outDim, inDim, A, inDim, 
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
			printFloatArr(b,outDim);
		}
		printf("Multiplying v sigma+ u with b for the solution \n");
		
	#endif
	// Multiplying v sigma+ u with b for the solution				\/ param 7
	cblas_sgemv (CblasRowMajor, CblasTrans, outDim, inDim,1, mb->c, inDim, b, 1, 0, output->solution, 1);
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



void * ipCacheDataCreator(void * input)
{
	#ifdef DEBUG 
		printf("-------------------ipCacheDataCreator--------------------------------\n");
	#endif
	struct ipCacheInput *myInput;
	myInput = (struct ipCacheInput *) input;
	nnLayer *layer = myInput->cache->layer;
	ipMemory *mb = myInput->mb;

	

	//If there's less included hyperplanes there will be a kernel
	if(mb->numHps <= layer->inDim){
		MKL_INT m = mb->numHps;
		MKL_INT n = layer->inDim;
		ipCacheData *ret = solve(mb->subA, m, n, mb->subB,mb);
		#ifdef DEBUG 
			printf("-------------------/ipCacheDataCreator--------------------------------\n");
		#endif
		return ret;
	} else {
		#ifdef DEBUG 
			printf("-------------------/ipCacheDataCreator--------------------------------\n");
		#endif
		return NULL;
	}
	
}
void ipCacheDataDestroy(void * data)
{
	struct ipCacheData *myData;
	myData = (struct ipCacheData *) data;
	if(myData){
		if(myData->solution){
			free(myData->solution);
		}
		free(myData);
	}
}

void createHPCache(ipCache *cache)
{
	
	int i = 0;
	nnLayer *layer = cache->layer;
	MKL_INT inDim = layer->inDim;
	MKL_INT outDim = layer->outDim;

	cache->hpNormals = malloc(outDim*inDim*2*sizeof(float));
	cache->hpOffsetVecs =  cache->hpNormals+ outDim*inDim;

	float scaling = 1;
	#ifdef DEBUG
		printf("--------createHPCache-------------\n");
		printf("inDim: %u\n", inDim);
		printf("outDim: %u\n", outDim);
	#endif
	for(i=0;i<outDim;i++){
		#ifdef DEBUG
			printf("----%u----\n",i);
			if(inDim < 10){
				printf("Hyperplane vector: ");
				printFloatArr(layer->A + inDim*i,inDim);
			}
			printf("Offset Value: %f\n",layer->b[i]);			
		#endif	
		scaling = cblas_snrm2 (inDim, layer->A + inDim*i, 1);
		if(scaling){
			cblas_saxpy (inDim,1/scaling,
						layer->A + inDim*i,1,
						cache->hpNormals + inDim*i,1);
			cblas_saxpy (inDim,layer->b[i]/scaling,
						cache->hpNormals+inDim*i,1,
						cache->hpOffsetVecs + inDim*i,1);
			#ifdef DEBUG
				printf("Scaling factor (norm of hyperplane vector): %f\n",scaling);
				
				if(inDim < 10){
					printf("Normalized normal vector: ");
					printFloatArr(cache->hpNormals + i*inDim,inDim);
					printf("Offset vector: ");
					printFloatArr(cache->hpOffsetVecs+ i*inDim,inDim);
				}
				
			#endif
		} else {
			// HP normals has length 0, thus it is 0. The offset vector should also be 0
			cblas_scopy(inDim,cache->hpNormals+i*inDim,1,cache->hpOffsetVecs+i*inDim,1);
		}
	}
	#ifdef DEBUG
		printf("---------------------------------\n");
	#endif
}

ipCache * allocateCache(nnLayer *layer, float threshhold, int depthRestriction, long long int freeMemory)
{
	long long int inDim = layer->inDim;
	long long int outDim = layer->outDim;
	ipCache *cache = malloc(sizeof(ipCache));
	uint keyLen = calcKeyLen(layer->outDim);
	cache->bases = createTree(6,keyLen, ipCacheDataCreator, NULL, ipCacheDataDestroy);

	int sizeOfAProjection =  layer->inDim*layer->inDim*sizeof(float);
	cache->maxNodesBeforeTrim = 8*(freeMemory)/(sizeOfAProjection);
	cache->maxNodesBeforeTrim = cache->maxNodesBeforeTrim/100;
	cache->maxNodesAfterTrim = (4*cache->maxNodesBeforeTrim);
	cache->maxNodesAfterTrim = cache->maxNodesAfterTrim/5;
	int rc = pthread_mutex_init(&(cache->balanceLock), 0);
	if (rc != 0) {
        printf("balanceLock Initialization failed at");
    }
	printf("Creating cache of inDim %lld and outDim %lld with threshhold %f and depth restriction %d \n", layer->inDim, layer->outDim, threshhold,depthRestriction);
	printf("Maximum number of nodes in the tree %lld, number of nodes after trim: %lld\n", cache->maxNodesBeforeTrim,cache->maxNodesAfterTrim );
	#ifdef DEBUG
		printf("-----------------allocateCache--------------------\n");
		if(layer->inDim < 10 && layer->outDim <20){
			printf("The matrix is: \n");
			printMatrix(layer->A,layer->inDim,layer->outDim);
			printf("The offset vector is: ");
			printFloatArr(layer->b, layer->outDim);
		}
	#endif
	cache->layer = layer;
	createHPCache(cache);
	cache->threshold = threshhold;
	cache->depthRestriction = depthRestriction;

	if(inDim < outDim){
		/* 
		If the spacial dim is less than the number of hyperplanes then we want to 
		take the distance to the closest instersection (which will be of full rank).
		*/
		cache->numRelevantHP = inDim + 1;
		cache->minOutIn = inDim;
	} else {
		cache->numRelevantHP = outDim;
		cache->minOutIn = outDim;
	}
	
	#ifdef DEBUG
		printf("--------------------------------------------------\n");
	#endif
	return cache;
}

void freeCache(ipCache * cache)
{
	if(cache){
		printf("Cache exists, deallocating\n");
		freeTree(cache->bases);
		if(cache->hpNormals){
			free(cache->hpNormals);
		}
		pthread_mutex_destroy(&(cache->balanceLock));
		free(cache);
	}
}


float computeDist(float * p, kint *ipSignature, ipCache *cache, ipMemory *mb)
{
	struct ipCacheData *myBasis;
	ipCacheInput myInput = {.cache = cache, .key = ipSignature, .mb=mb};
	char cacheUse =0;
	#ifdef DEBUG
		printf("Using Cache\n");
	#endif
	myBasis = addData(cache->bases, ipSignature, &myInput);
	cacheUse = 1;
	

	if(myBasis){
		MKL_INT inDim = cache->layer->inDim;
		cblas_scopy (inDim, p, 1, mb->px, 1);
		cblas_saxpy (inDim,1,myBasis->solution,1,mb->px,1);
		if(myBasis->projection){
			cblas_sgemv (CblasRowMajor, CblasNoTrans, inDim, inDim,-1,myBasis->projection, inDim, mb->px, 1, 1, mb->px, 1);
		} 
		float norm = cblas_snrm2 (inDim, mb->px, 1);
		if(norm < 0){
			norm = -norm;
		}
		if(!cacheUse){
			ipCacheDataDestroy((void *) myBasis);
		}
		return norm;
	} else {
		return -1;
	}
}



void computeDistToHPS(float *p, ipCache *cache, float *distances, ipMemory *mb)
{
	MKL_INT outDim = cache->layer->outDim;
	MKL_INT inDim = cache->layer->inDim;
	cblas_scopy (inDim*outDim, cache->hpOffsetVecs, 1, mb->localCopy, 1);
	
	#ifdef DEBUG
		printf("--------computeDistToHPS---------\n");
	#endif
	for(int i =0;i<outDim;++i){
		#ifdef DEBUG
			printf("----%d----\n",i);
			if(inDim < 10){
				printf("Point were taking the distance to: ");
				printFloatArr(p,inDim);
				printf("Copied over the offset vectors: ");
				printFloatArr(mb->localCopy + i*inDim,inDim);
				printf("Original offset vectors: ");
				printFloatArr(cache->hpOffsetVecs+ i*inDim,inDim);
				printf("Normal Vectors: ");
				printFloatArr(cache->hpNormals + i*inDim,inDim);
			}
		#endif
		cblas_saxpy (inDim,1,p,1,mb->localCopy + i*inDim,1);
		distances[i] = cblas_sdot (inDim, mb->localCopy + i*inDim, 1, cache->hpNormals + i*inDim, 1);
		if(distances[i] < 0){
			distances[i] = -distances[i];
		}
		if(distances[i] == 0){
			// Sometimes the normal vector is 0, ie, this hyperplane is not used
			// We check for this, and if so, essentially remove this distance from the situation.
			float norm = cblas_snrm2 (inDim, cache->hpNormals + i*inDim, 1);
			if(norm == 0){
				distances[i] = FLT_MAX;
			} else {
				distances[i] = 0;
			}
			
		}
		#ifdef DEBUG
			printf("p + offset vector: ");
			if(inDim < 10){
				printFloatArr(mb->localCopy+i*inDim,inDim);
			}
			if(mb->distances[i] == FLT_MAX){
				printf("Final distance is massive as the hp does not exist\n");
			} else {
				printf("Final distance (normal dotted with p+offset): %f \n",distances[i]);
			}
			printf("---------\n");
		#endif
	}
	#ifdef DEBUG
		printf("---------------------------------\n");
	#endif
}

void fillIntersection(int * hpIndexList, int numRelevantHP, nnLayer *layer, ipMemory *mb)
{
	uint inDim = layer->inDim;
	int i = 0;
	int numHps = 0;
	for(i=0;i<numRelevantHP;i++){
		cblas_scopy (inDim, layer->A+hpIndexList[i]*inDim, 1, mb->subA + numHps*inDim, 1);
		mb->subB[numHps] = layer->b[i];
		numHps++;
	}
}

void getInterSig(ipCache * cache, float *p, kint * ipSignature, ipMemory *mb)
{
	int i = 0;
	// Prepare all the internal values and place commonly referenced ones on the stack
	int outDim = cache->layer->outDim;
	uint keyLength = cache->bases->keyLength;
	clearKey(ipSignature, keyLength);

	

	computeDistToHPS(p, cache, mb->distances,mb);
	cblas_scopy (outDim, mb->distances, 1, mb->distancesForSorting, 1);
	for(i=0;i<cache->numRelevantHP;i++){
		mb->hpDistIndexList[i] = cblas_isamin (outDim, mb->distancesForSorting, 1);
		mb->distancesForSorting[mb->hpDistIndexList[i]] = FLT_MAX;
		addIndexToKey(ipSignature,mb->hpDistIndexList[i]);
	}
	fillIntersection(mb->hpDistIndexList,cache->numRelevantHP, cache->layer,mb);

	float posetDist = 0;
	float hpDist;
	i = cache->numRelevantHP-1;
	do{
		removeIndexFromKey(ipSignature,mb->hpDistIndexList[i]);
		hpDist = mb->distances[mb->hpDistIndexList[i]];
		posetDist = computeDist(p, ipSignature, cache, mb);
		#ifdef DEBUG
			printf("Loop %d \n",i);
			printf("Checking xu{i}(dist:%f) vs %f*{j}(dist:%f)",posetDist,cache->threshold,hpDist);
			printf("xu{i}:");
			printKey(ipSignature,outDim);
			printf("i: %u \n", mb->hpDistIndexList[i-1]);
			printf("j: %u \n", mb->hpDistIndexList[i]);
		#endif
		mb->numHps = i;
		i--;
		#ifdef DEBUG
			if(i > -1){
				if(cache->threshold*posetDist - hpDist > 0){
					printf("[%d]Passed, decrease while loop will run. Difference between poset and hp distance:%f\n",i,cache->threshold*posetDist - hpDist);
				} else {
					printf("[%d]Failed, decrease while loop will not run. Difference between poset and hp distance:%f\n",i,cache->threshold*posetDist - hpDist);
				}
			}			
		#endif
	} while(!(posetDist < 0) && cache->threshold*posetDist - hpDist > 0 && i > 0);
	if(i == 0){
		posetDist = mb->distances[mb->hpDistIndexList[0]];
		hpDist = mb->distances[mb->hpDistIndexList[1]];
		if(cache->threshold*posetDist - hpDist > 0){
			removeIndexFromKey(ipSignature,mb->hpDistIndexList[0]);;
		}
	}
	
	#ifdef DEBUG
		printf("Main While Loop Completed\n");
		printf("The raw interSig is (in key form)");
		printKeyArr(ipSignature,keyLength);
		printf("The unfurled interSig is ");
		printKey(ipSignature,outDim);
		printf("--------------------------/getInterSig-----------------------------------------------\n");;
	#endif
}

struct IPAddThreadArgs {
	uint tid;
	uint numThreads;

	uint numData;
	float * data;
	kint *ipSignature;

	int * joinCondition;

	ipCache *cache;
};

int checkJoinCondition(int * joinConditionArr, int numThreads)
{
	int i = 0;
	int j = 1;
	while(j){
		j = 0;
		for(i=0;i<numThreads;i++){
			if(joinConditionArr[i]==0){
				j = 1;
			}
		}
	}
	return 1;
}

void * addIPBatch_thread(void *thread_args)
{
	struct IPAddThreadArgs *myargs;
	myargs = (struct IPAddThreadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;

	uint numData = myargs->numData;
	float *data = myargs->data;
	kint *ipSignature = myargs->ipSignature;

	int *joinCondition = myargs->joinCondition;

	ipCache *cache = myargs->cache;

	uint dim = cache->layer->inDim;
	uint keySize = calcKeyLen(cache->layer->outDim);
	ipMemory *mb = allocateIPMemory(cache->layer->inDim,cache->layer->outDim);
	uint i = 0;
	for(i=tid;i<numData;i=i+numThreads){		
		getInterSig(cache,data+i*dim, ipSignature+i*keySize,mb);
		joinCondition[tid] = 1;
		printf("Thread %d at mutex with %u nodes \n",tid,cache->bases->numNodes);
		pthread_mutex_lock(&(cache->balanceLock));
			if(cache->bases->numNodes > cache->maxNodesBeforeTrim){
				printf("Balancing Tree in thread %d. Current nodes: %d\n",tid,cache->bases->numNodes);
				checkJoinCondition(joinCondition,numThreads);
				balanceAndTrimTree(cache->bases, cache->maxNodesAfterTrim);
			}
		pthread_mutex_unlock(&(cache->balanceLock));
		joinCondition[tid] = 0;
	}
	pthread_exit(NULL);
}

void getInterSigBatch(ipCache *cache, float *data, kint *ipSignature, uint numData, uint numProc)
{
	int maxThreads = numProc;
	int rc =0;
	int i =0;
	struct IPAddThreadArgs *thread_args = malloc(maxThreads*sizeof(struct IPAddThreadArgs));
	int *joinCondition = malloc(maxThreads*sizeof(int));
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
		thread_args[i].joinCondition = joinCondition;
		rc = pthread_create(&threads[i], NULL, addIPBatch_thread, (void *)&thread_args[i]);
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