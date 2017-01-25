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
struct ipCacheData * solve(float *A, MKL_INT outDim, MKL_INT inDim, float *b)
{	
	#ifdef DEBUG
		printf("------------------Solve--------------\n");
		printf("The dimensions are inDim: %u, outDim: %d \n", inDim, outDim);
	#endif
	MKL_INT info;
	MKL_INT minMN = ((inDim)>(outDim)?(outDim):(inDim));
	float *superb = calloc(minMN, sizeof(float)); 
	/* Local arrays */
	float *s = calloc(inDim,sizeof(float)); // Diagonal entries
	float *u = calloc(outDim*outDim , sizeof(float));
	float *vt = calloc(inDim*inDim, sizeof(float));
	struct ipCacheData *output = malloc(sizeof(struct ipCacheData));


	/* Executable statements */
	info = LAPACKE_sgesvd( LAPACK_ROW_MAJOR, 'A', 'A',
						  outDim, inDim, A, inDim,
		    			  s, u, outDim,
		    			  vt, inDim,
		    			  superb);
	// Incase of memory leaks:
	//mkl_thread_free_buffers();
	if( info > 0 ) {
		printf( "The algorithm computing SVD failed to converge.\n" );
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
		printf("U:\n");
		printMatrix(u,outDim,outDim);
		printf("Diagonal entries of S:\n");
		printFloatArr(s,minMN);
		printf("V^T:\n");
		printMatrix(vt,inDim,inDim);
		printf("Multiplying u sigma+t with vt\n");
	#endif
	// Multiplying u sigma+t with vt
	cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans,
				outDim, inDim, outDim, 1, u,
				outDim, vt, inDim,
				0,c, inDim);
	#ifdef DEBUG
		printf("Result of the previous multiplication:\n");
		printMatrix(c,inDim,outDim);
		printf("Multiplying v sigma+ u with b for the solution \n");
		printf("b:");
		printFloatArr(b,outDim);
	#endif
	// Multiplying v sigma+ u with b for the solution				\/ param 7
	cblas_sgemv (CblasRowMajor, CblasTrans, inDim, outDim,1, c, outDim, b, 1, 0, solution, 1);
	#ifdef DEBUG
		printf("Result of the previous multiplication:\n");
		printFloatArr(solution,inDim);
		
	#endif
	// Saving the kernel basis from vt
	if(outDim<inDim){
		output->projection = calloc(inDim*inDim,sizeof(float));
		#ifdef DEBUG
			printf("Multiplying the first %u rows of vt for the projection\n", outDim);
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
		ipCacheData *ret = solve(subA, m, n, subB);
		free(subA);
		free(subB);
		return ret;
	} else {
		free(subA);
		free(subB);
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
	nnLayer *layer0 = cache->layer0;
	uint inDim = layer0->inDim;
	uint outDim = layer0->outDim;

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
			printf("Hyperplane vector: ");
			printFloatArr(layer0->A + inDim*i,inDim);
			printf("Offset Value: %f\n",layer0->b[i]);			
		#endif	
		scaling = cblas_snrm2 (inDim, layer0->A + inDim*i, 1);
		if(scaling){
			cblas_saxpy (inDim,1/scaling,layer0->A + inDim*i,1,cache->hpNormals + inDim*i,1);
			cblas_saxpy (inDim,layer0->b[i]/scaling,cache->hpNormals+inDim*i,1,cache->hpOffsetVecs + inDim*i,1);
			#ifdef DEBUG
				printf("Scaling factor (norm of hyperplane vector): %f\n",scaling);
				printf("Normalized normal vector: ");
				printFloatArr(cache->hpNormals + i*inDim,inDim);
				printf("Offset vector: ");
				printFloatArr(cache->hpOffsetVecs+ i*inDim,inDim);
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

ipCache * allocateCache(nnLayer *layer0, float threshhold)
{
	ipCache *cache = malloc(sizeof(ipCache));
	uint keyLen = calcKeyLen(layer0->outDim);
	cache->bases = createTree(8,keyLen , ipCacheDataCreator, NULL, ipCacheDataDestroy);

	printf("Creating cache of inDim %d and outDim %d with threshhold %f \n", layer0->inDim, layer0->outDim, threshhold);
	#ifdef DEBUG
		printf("-----------------allocateCache--------------------\n");
		printf("The matrix is: \n");
		printMatrix(layer0->A,layer0->inDim,layer0->outDim);
		printf("The offset vector is: ");
		printFloatArr(layer0->b, layer0->outDim);
	#endif
	cache->layer0 = layer0;
	createHPCache(cache);
	cache->threshold = threshhold;
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
		if(norm < 0){
			norm = -norm;
		}
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
	
	#ifdef DEBUG
		printf("--------computeDistToHPS---------\n");
	#endif
	for(uint i =0;i<outDim;++i){
		#ifdef DEBUG
			printf("----%d----\n",i);
			printf("Point were taking the distance to: ");
			printFloatArr(p,inDim);
			printf("Copied over the offset vectors: ");
			printFloatArr(localCopy + i*inDim,inDim);
			printf("Original offset vectors: ");
			printFloatArr(cache->hpOffsetVecs+ i*inDim,inDim);
			printf("Normal Vectors: ");
			printFloatArr(cache->hpNormals + i*inDim,inDim);
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
			printFloatArr(localCopy+i*inDim,inDim);
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


void getInterSig(ipCache * cache, float *p, uint *ipSignature)
{
	uint outDim = cache->layer0->outDim;
	uint inDim = cache->layer0->inDim;
	uint keyLength = cache->bases->keyLength;
	#ifdef DEBUG
		printf("--------------------------getInterSig-----------------------------------------------\n");
		printf("Aquiring the ipSignature of ");
		printFloatArr(p,inDim);
	#endif
	float *distances = calloc(outDim,sizeof(float));
	computeDistToHPS(p, cache, distances);
	#ifdef DEBUG
		printf("The distances to the hyperplanes are ");
		printFloatArr(distances,outDim);
	#endif
	uint j = 1;
	
	clearKey(ipSignature,keyLength);
	

	// Get the distance to the closest hyperplane and blank it from the distance array
	uint curSmallestIndex = cblas_isamin (outDim, distances, 1);
	float curDist = distances[curSmallestIndex];
	#ifdef DEBUG
		printf("The closest hyperplane is %u and is %f away.\n", curSmallestIndex,curDist);
	#endif
	distances[curSmallestIndex] = FLT_MAX;
	addIndexToKey(ipSignature, curSmallestIndex);
	#ifdef DEBUG
		printf("After adding index %u the key is now: ", curSmallestIndex);
		printKey(ipSignature,outDim);
	#endif	
	// Get the distance to the second closest hyperplane and blank it
	curSmallestIndex = cblas_isamin (outDim, distances, 1);
	float nextDist = distances[curSmallestIndex];
	distances[curSmallestIndex] = FLT_MAX;
	#ifdef DEBUG
		printf("The second closest hyperplane is %u and is %f away.\n", curSmallestIndex,nextDist);
		if(!(curDist>0 && nextDist < cache->threshold*curDist && j < inDim)){
			printf("\t This does not satisfy the condition.\n");
		}
	#endif

	while(curDist>0 && nextDist < cache->threshold*curDist && j < inDim)
	{	
		#ifdef DEBUG
			printf("----While Loop %u----\n", j);
			printf("Current Distance is %f, nextDist is %f\n", curDist, nextDist);
			printf("Adding %u to the ipSignature and finding the distance to the associated intersection\n",curSmallestIndex);
		#endif
		addIndexToKey(ipSignature, curSmallestIndex);
		#ifdef DEBUG
			printf("The raw interSig is (in key form)");
			printKeyArr(ipSignature,keyLength);
			printf("The unfurled interSig is ");
			printKey(ipSignature,outDim);
		#endif
		
		// Prepare for next loop
		curSmallestIndex = cblas_isamin (outDim, distances, 1);
		// Current distance should be the distance to the current IP set
		curDist = computeDist(p, ipSignature, cache);
		// Next distance should either be to the next hyperplane or the next closest IP of the same rank.
		nextDist = distances[curSmallestIndex];
		distances[curSmallestIndex] = FLT_MAX;
		j++;
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
}

struct dataAddThreadArgs {
	uint tid;
	uint numThreads;

	uint numData;
	float * data;
	uint *ipSignature;

	ipCache *cache;
};

void * addBatch_thread(void *thread_args)
{
	struct dataAddThreadArgs *myargs;
	myargs = (struct dataAddThreadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;

	uint numData = myargs->numData;
	float *data = myargs->data;
	uint *ipSignature = myargs->ipSignature;

	ipCache *cache = myargs->cache;

	uint dim = cache->layer0->inDim;
	uint keySize = calcKeyLen(cache->layer0->outDim);

	uint i = 0;
	for(i=tid;i<numData;i=i+numThreads){		
		getInterSig(cache,data+i*dim, ipSignature+i*keySize);
	}
	pthread_exit(NULL);
}

void getInterSigBatch(ipCache *cache, float *data, uint *ipSignature, uint numData, uint numProc)
{
	int maxThreads = numProc;
	int rc =0;
	int i =0;
	printf("Number of processors: %d\n",maxThreads);
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