#include "ipCalculator.h"
#include <float.h>

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

typedef struct intersection {
	uint numHps;
	float *subA;
	float *subB;

	uint maxNumHps;
} intersection;

typedef struct ipCacheInput {
	//This should be a constant
	const ipCache * info;
	intersection *I;
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
struct ipCacheData * solve(float *A, MKL_INT outDim, MKL_INT inDim, float *b)
{	
	#ifdef DEBUG
		printf("------------------Solve--------------\n");
		printf("The dimensions are inDim: %lld, outDim: %lld \n", inDim, outDim);
	#endif
	MKL_INT info;
	MKL_INT minMN = ((inDim)>(outDim)?(outDim):(inDim));
	float *superb = calloc(minMN, sizeof(float)); 
	/* Local arrays */
	float *s = calloc(inDim,sizeof(float)); // Diagonal entries
	float *u = calloc(outDim*outDim , sizeof(float));
	float *vt = calloc(inDim*inDim, sizeof(float));
	struct ipCacheData *output = malloc(sizeof(struct ipCacheData));


	/* Standard SVD, not the new type */
	/*
	info = LAPACKE_sgesvd( LAPACK_ROW_MAJOR, 'A', 'A',
						  outDim, inDim, A, inDim,
		    			  s, u, outDim,
		    			  vt, inDim,
		    			  superb);
	*/
	info = LAPACKE_sgesdd( LAPACK_ROW_MAJOR, 'A', outDim, inDim, A, inDim, 
							s, 
							u, outDim, 
							vt, inDim );
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
		cblas_sscal(outDim,(1/s[i]),u+i*outDim,1);
	}
		 
	float *solution = calloc(inDim, sizeof(float));
	float *c = calloc(inDim*outDim, sizeof(float));
	#ifdef DEBUG
		if(inDim < 10 && outDim < 20){
			printf("U:\n");
			printMatrix(u,outDim,outDim);
			printf("Diagonal entries of S:\n");
			printFloatArr(s,minMN);
			printf("V^T:\n");
			printMatrix(vt,inDim,inDim);
		}
		printf("Multiplying u sigma+t with vt\n");
	#endif
	// Multiplying u sigma+t with vt
	cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans,
				outDim, inDim, outDim, 1, u,
				outDim, vt, inDim,
				0,c, inDim);
	#ifdef DEBUG
		if(inDim < 10 && outDim < 20){
			printf("Result of the previous multiplication:\n");
			printMatrix(c,inDim,outDim);
			printf("b:");
			printFloatArr(b,outDim);
		}
		printf("Multiplying v sigma+ u with b for the solution \n");
		
	#endif
	// Multiplying v sigma+ u with b for the solution				\/ param 7
	cblas_sgemv (CblasRowMajor, CblasTrans, outDim, inDim,1, c, inDim, b, 1, 0, solution, 1);
	#ifdef DEBUG
		printf("Result of the previous multiplication:\n");
		if(inDim < 10){
			printFloatArr(solution,inDim);
		}
		
	#endif
	// Saving the kernel basis from vt
	if(outDim<inDim){
		output->projection = calloc(inDim*inDim,sizeof(float));
		#ifdef DEBUG
			printf("Multiplying the first %lld rows of vt for the projection\n", outDim);
		#endif

		cblas_sgemm (CblasRowMajor, CblasTrans, CblasNoTrans,
					outDim, outDim, (inDim-outDim), 1, vt+inDim*outDim,outDim, 
					vt+inDim*outDim, inDim,
					0, output->projection, outDim);
	} else {
		output->projection = NULL;
	}
	free(s);
	free(u);
	free(vt);
	free(c);
	free(superb);
	// Incase of memory leaks:
	//mkl_thread_free_buffers();

	output->solution = solution;
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
	nnLayer *layer = myInput->info->layer;
	intersection * I = myInput->I;

	

	//If there's less included hyperplanes there will be a kernel
	if(I->numHps <= layer->inDim){
		MKL_INT m = I->numHps;
		MKL_INT n = layer->inDim;
		ipCacheData *ret = solve(I->subA, m, n, I->subB);
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
	
	uint i = 0;
	nnLayer *layer = cache->layer;
	uint inDim = layer->inDim;
	uint outDim = layer->outDim;

	cache->hpNormals = calloc(outDim*inDim,sizeof(float));
	cache->hpOffsetVecs = calloc(outDim*inDim,sizeof(float));

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
			cblas_saxpy (inDim,1/scaling,layer->A + inDim*i,1,cache->hpNormals + inDim*i,1);
			cblas_saxpy (inDim,layer->b[i]/scaling,cache->hpNormals+inDim*i,1,cache->hpOffsetVecs + inDim*i,1);
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

ipCache * allocateCache(nnLayer *layer, float threshhold, int depthRestriction)
{
	ipCache *cache = malloc(sizeof(ipCache));
	uint keyLen = calcKeyLen(layer->outDim);
	cache->bases = createTree(8,keyLen , ipCacheDataCreator, NULL, ipCacheDataDestroy);

	printf("Creating cache of inDim %d and outDim %d with threshhold %f and depth restriction %d \n", layer->inDim, layer->outDim, threshhold,depthRestriction);
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
		if(cache->hpOffsetVecs){
			free(cache->hpOffsetVecs);
		}
		if(cache->hpNormals){
			free(cache->hpNormals);
		}
		free(cache);
	}
}


float computeDist(float * p, kint *ipSignature, ipCache *cache, intersection *I)
{
	struct ipCacheData *myBasis;
	ipCacheInput myInput = {.info = cache, .key = ipSignature, .I=I};
	char cacheUse =0;
	if(I->maxNumHps < I->numHps + cache->depthRestriction){
		#ifdef DEBUG
			printf("Using Cache\n");
		#endif
		myBasis = addData(cache->bases, ipSignature, &myInput);
		cacheUse = 1;
	} else {
		#ifdef DEBUG
			printf("Computing without cache.\n");
		#endif
		myBasis = ipCacheDataCreator(&myInput);
		cacheUse =0;
	}

	if(myBasis){
		MKL_INT inDim = cache->layer->inDim;
		float * px = malloc(inDim * sizeof(float));
		cblas_scopy (inDim, p, 1, px, 1);
		cblas_saxpy (inDim,1,myBasis->solution,1,px,1);
		if(myBasis->projection){
			cblas_sgemv (CblasRowMajor, CblasNoTrans, inDim, inDim,-1,myBasis->projection, inDim, px, 1, 1, px, 1);
		} 
		float norm = cblas_snrm2 (inDim, px, 1);
		if(norm < 0){
			norm = -norm;
		}
		if(!cacheUse){
			ipCacheDataDestroy((void *) myBasis);
		}
		free(px);
		return norm;
	} else {
		return -1;
	}
}



void computeDistToHPS(float *p, ipCache *cache, float *distances)
{
	uint outDim = cache->layer->outDim;
	uint inDim = cache->layer->inDim;
	float * localCopy = calloc(outDim*inDim,sizeof(float));
	cblas_scopy (inDim*outDim, cache->hpOffsetVecs, 1, localCopy, 1);
	
	#ifdef DEBUG
		printf("--------computeDistToHPS---------\n");
	#endif
	for(uint i =0;i<outDim;++i){
		#ifdef DEBUG
			printf("----%d----\n",i);
			if(inDim < 10){
				printf("Point were taking the distance to: ");
				printFloatArr(p,inDim);
				printf("Copied over the offset vectors: ");
				printFloatArr(localCopy + i*inDim,inDim);
				printf("Original offset vectors: ");
				printFloatArr(cache->hpOffsetVecs+ i*inDim,inDim);
				printf("Normal Vectors: ");
				printFloatArr(cache->hpNormals + i*inDim,inDim);
			}
		#endif
		cblas_saxpy (inDim,1,p,1,localCopy + i*inDim,1);
		distances[i] = cblas_sdot (inDim, localCopy + i*inDim, 1, cache->hpNormals + i*inDim, 1);
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
				printFloatArr(localCopy+i*inDim,inDim);
			}
			if(distances[i] == FLT_MAX){
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
	free(localCopy);
}

intersection * fillIntersection(uint * hpIndexList, int numRelevantHP, nnLayer *layer)
{
	uint inDim = layer->inDim;
	struct intersection * I = malloc(sizeof(struct intersection)); 
	I->subA = calloc((inDim+1)*numRelevantHP, sizeof(float));
	I->subB = I->subA + inDim*numRelevantHP;
	I->maxNumHps = numRelevantHP;
	int i = 0;
	int numHps = 0;
	for(i=0;i<numRelevantHP;i++){
		cblas_scopy (inDim, layer->A +hpIndexList[i]*inDim, 1, I->subA + numHps*inDim, 1);
		I->subB[numHps] = layer->b[i];
		numHps++;
	}
	return I;
}

void freeIntersection(intersection *I)
{
	free(I->subA);
	free(I);
}

void getInterSig(ipCache * cache, float *p, kint * ipSignature)
{
	int i = 0;
	// Prepare all the internal values and place commonly referenced ones on the stack
	uint outDim = cache->layer->outDim;
	uint inDim = cache->layer->inDim;
	uint keyLength = cache->bases->keyLength;
	clearKey(ipSignature, keyLength);

	int numRelevantHP;
	if(inDim < outDim){
		/* 
		If the spacial dim is less than the number of hyperplanes then we want to 
		take the distance to the closest instersection (which will be of full rank).
		
		*/
		numRelevantHP = inDim + 1;
	} else {
		numRelevantHP = outDim;
	}

	float *posetDistToHP = malloc(outDim*sizeof(float));
	uint *hpDistIndexList = malloc(numRelevantHP*sizeof(uint));
	
	float *distances = malloc(outDim*sizeof(float));
	float *distancesForSorting = malloc(outDim*sizeof(float));
	computeDistToHPS(p, cache, distances);
	cblas_scopy (outDim, distances, 1, distancesForSorting, 1);
	for(i=0;i<numRelevantHP;i++){
		hpDistIndexList[i] = cblas_isamin (outDim, distancesForSorting, 1);
		distancesForSorting[hpDistIndexList[i]] = FLT_MAX;
		addIndexToKey(ipSignature,hpDistIndexList[i]);
	}
	free(distancesForSorting);
	
	intersection *I = fillIntersection(hpDistIndexList,numRelevantHP, cache->layer);

	float posetDist;
	float hpDist;
	i = numRelevantHP-1;
	do{
		removeIndexFromKey(ipSignature,hpDistIndexList[i]);
		I->numHps = i;
		posetDist = computeDist(p, ipSignature, cache, I);
		hpDist = distances[hpDistIndexList[i]];
		i--;
		#ifdef DEBUG
			if(i > -1){
				if(posetDistToHP[i] > 0){
					printf("Passed, decrease while loop will run. Value of [%u]:%f\n",i,cache->threshold*posetDist - hpDist);
				} else {
					printf("Failed, decrease while loop will not run. Value of [%u]:%f\n",i,cache->threshold*posetDist - hpDist);
				}
			}			
		#endif
	} while(!(posetDist < 0) && cache->threshold*posetDist - hpDist > 0 && i > 0);
	if(i == 0){
		posetDist = distances[hpDistIndexList[0]];
		hpDist = distances[hpDistIndexList[1]];
		if(cache->threshold*posetDist - hpDist > 0){
			removeIndexFromKey(ipSignature,hpDistIndexList[0]);;
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
	free(distances);
	free(posetDistToHP);
	free(hpDistIndexList);
	freeIntersection(I);
}

struct IPAddThreadArgs {
	uint tid;
	uint numThreads;

	uint numData;
	float * data;
	kint *ipSignature;

	ipCache *cache;
};

void * addIPBatch_thread(void *thread_args)
{
	struct IPAddThreadArgs *myargs;
	myargs = (struct IPAddThreadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;

	uint numData = myargs->numData;
	float *data = myargs->data;
	kint *ipSignature = myargs->ipSignature;

	ipCache *cache = myargs->cache;

	uint dim = cache->layer->inDim;
	uint keySize = calcKeyLen(cache->layer->outDim);

	uint i = 0;
	for(i=tid;i<numData;i=i+numThreads){		
		getInterSig(cache,data+i*dim, ipSignature+i*keySize);
	}
	pthread_exit(NULL);
}

void getInterSigBatch(ipCache *cache, float *data, kint *ipSignature, uint numData, uint numProc)
{
	int maxThreads = numProc;
	int rc =0;
	int i =0;
	//printf("Number of processors: %d\n",maxThreads);
	//Add one data to the first node so that we can avoid the race condition.
	

	struct IPAddThreadArgs *thread_args = malloc(maxThreads*sizeof(struct IPAddThreadArgs));

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