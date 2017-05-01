#include "ipCalculator.h"
#include <float.h>
#include <string.h>






ipMemory * allocateIPMemory(int inDim,int outDim)
{
	int m = inDim;
	int n = outDim;
	
	ipMemory *mb = malloc(sizeof(ipMemory));
	mb->I = malloc(sizeof(intersection));
	if(m < n){
		/* 
		If the spacial dim is less than the number of hyperplanes then we want to 
		take the distance to the closest instersection (which will be of full rank).
		*/
		mb->minMN = m;
	} else {
		mb->minMN = n;
	}
	// Each pointer is followed by the number of floats it has all to itself.
	mb->s = malloc(m*sizeof(float)); //m
	mb->py = malloc(m*sizeof(float));
	mb->px = malloc(m*sizeof(float)); //m
	mb->u = malloc(n*n*sizeof(float)); //n^2
	mb->vt = malloc(m*m*sizeof(float)); //m^2
	mb->c = malloc(n*m*sizeof(float)); //m*n
	mb->localCopy =  malloc(n*m*sizeof(float)); //m*n
	mb->superb = malloc(mb->minMN*sizeof(float)); // minMN
	mb->I->subA = malloc(n*m*sizeof(float)); // m*minMN
	mb->I->subB = malloc(n*sizeof(float)); // m
	
	mb->distances = malloc(n*sizeof(distanceWithIndex)); // n
	
	return mb;
}

void freeIPMemory(ipMemory *mb)
{
	free(mb->s);
	free(mb->py);
	free(mb->px);
	free(mb->u);
	free(mb->vt);
	free(mb->c);
	free(mb->localCopy);
	free(mb->superb);
	free(mb->I->subB);
	free(mb->I->subA);
	free(mb->I);
	free(mb->distances);
	free(mb);
}

typedef struct ipCacheInput {
	//This should be a constant
	const ipCache * info;
	const ipMemory *mb;
	//This shouldn't be
	kint *key;
} ipCacheInput;

// Types should be: kernel projection or perp kernel projection

typedef struct ipCacheData {
	char type;
	float *solution;
	float *projection;
} ipCacheData;



/*
 This can only handle neural networks whose input dimension is greater than the number of nodes in the first layer
*/
void solve(float *A, MKL_INT outDim, MKL_INT inDim, float *b, struct ipCacheData * myBasis, const ipMemory *mb)
{	
	const MKL_INT inout = (inDim-outDim);
	const float oneF = 1.0;
	const float zeroF = 0.0;
	const char N = 'N';
	#ifdef DEBUG
		printf("------------------Solve--------------\n");
		printf("The dimensions are inDim: %d, outDim: %d \n", inDim, outDim);
		if(inDim < 20 && outDim < 20){	
			printf("A: \n");
			printMatrix(A,inDim,outDim);
			printf("b:");
			printFloatArr(b,outDim);
		}
	#endif
	MKL_INT info;
	MKL_INT minMN = ((inDim)>(outDim)?(outDim):(inDim));
	

	/* Standard SVD, not the new type */
	
	info = LAPACKE_sgesvd( LAPACK_ROW_MAJOR, 'A', 'A',
						  outDim, inDim, A, inDim,
		    			  mb->s, mb->u, outDim,
		    			  mb->vt, inDim,
		    			  mb->superb);
	/*
	info = LAPACKE_sgesdd( LAPACK_ROW_MAJOR, 'A', outDim, inDim, A, inDim, 
							mb->s, 
							mb->u, outDim, 
							mb->vt, inDim );
	*/
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

	// Multiplying Sigma+t with u
	int i = 0;
	while(i<outDim && mb->s[i] !=0){
		cblas_sscal(outDim,(1/mb->s[i]),mb->u+i,outDim);
		i++;
	}
	#ifdef DEBUG
		if(inDim < 10 && outDim < 20){
			printf("U*(D+):\n");
			printMatrix(mb->u,outDim,outDim);
		}
	#endif
	outDim = i;
	// Multiplying (u sigma+t) with vt
	/*
	cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans,
				outDim, inDim, outDim, 
				1, mb->u, outDim, 
				mb->vt, inDim,
				0,mb->c, inDim);
	
	*/
	sgemm (&N,&N, 
		&inDim, &outDim, &outDim, 
		&oneF, mb->vt, &inDim, 
		mb->u, &outDim, 
		&zeroF,mb->c,&inDim);
	
	#ifdef DEBUG
		if(inDim < 10 && outDim < 20){
			printf(" V*(U*(D+))T:\n");
			printMatrix(mb->c,inDim,outDim);

		}
		printf("Multiplying U*(D+) * Vt with b for the solution \n");
	#endif
	// Multiplying v sigma+ u with b for the solution				\/ param 7
	cblas_sgemv (CblasRowMajor, CblasTrans, outDim, inDim,1, mb->c, inDim, b, 1, 0, mb->px, 1);
	/*
	sgemm (&N,&N, 
		&oneI, &inDim, &outDim, 
		&oneF, b, &oneI, 
		mb->c, &outDim, 
		&zeroF,mb->px,&oneI);
	*/
	//sgemv (&N, &inDim, &outDim, &oneF, mb->c, &inDim, b, &oneI, &zeroF,mb->px,&oneI);
			
	#ifdef DEBUG
		printf("Result of the previous multiplication (solution):\n");
		if(inDim < 10){
			printFloatArr(mb->px,inDim);
		}
		
	#endif
	// Saving the kernel basis from vt
	
	switch(myBasis->type){
		case 0:
			memcpy(myBasis->solution, mb->px, inDim*sizeof(float));
			break;
		case 1:
			memcpy(myBasis->projection, mb->vt, outDim*inDim*sizeof(float));
			//cblas_scopy(outDim*inDim, mb->vt, 1, myBasis->projection,1);
			cblas_sgemv (CblasRowMajor, CblasNoTrans, outDim, inDim,1, myBasis->projection, inDim, mb->px, 1, 0, myBasis->solution, 1);
	
			//sgemm (&N,&N, &oneI, &outDim, &inDim, &oneF, mb->px, &oneI, myBasis->projection, &inDim, &zeroF,myBasis->solution,&oneI);
			
			#ifdef DEBUG
				if(inDim < 10 && outDim < 20){
					printf("Result of the memcpy:\n");
					printMatrix(myBasis->projection,inDim,outDim);
					printf("Proj(solution)\n");
					printFloatArr(myBasis->solution,outDim);
				}
			#endif
			break;
		case 2:
			memcpy(myBasis->projection, mb->vt+outDim*inDim, (inDim-outDim)*inDim*sizeof(float));
			memcpy(myBasis->solution, mb->px, inDim*sizeof(float));
			#ifdef DEBUG
				if(inDim < 10 && outDim < 20){
					printf("Result of the memcpy:\n");
					printMatrix(myBasis->projection,inDim,inout);
					printf("Proj(solution)\n");
					printFloatArr(myBasis->solution,inDim);
				}
			#endif
			break;
		default:
			printf("myBasis type is not one of the designated types\n");
			exit(-1);
	}
		


	// Incase of memory leaks:
	//mkl_thread_free_buffers();

	#ifdef DEBUG
		printf("-----------------/Solve--------------\n");
	#endif
}



void fillIntersection(kint *key, nnLayer *layer, intersection *I)
{
	uint inDim = layer->inDim;
	uint outDim = layer->outDim;

	uint i = 0;
	I->numHps = 0;
	#ifdef DEBUG 
		printf("Processing Key\n");
		printKey(key,outDim);
	#endif
	for(i=0;i<outDim;i++){
		if(checkIndex(key,i)){
			memcpy(I->subA + I->numHps*inDim, layer->A +i*inDim, inDim*sizeof(float));
			I->subB[I->numHps] = layer->b[i];
			I->numHps++;
		}
	}
}

void fillIntersectionOrdered(distanceWithIndex *distances, int distCount, nnLayer *layer, intersection *I)
{
	uint inDim = layer->inDim;
	
	for(int i=0;i<distCount;i++){
		memcpy(I->subA + i*inDim, layer->A+(distances[i].index)*inDim, inDim*sizeof(float));
		I->subB[i] = layer->b[distances[i].index];
	}
} 

int detectType(int m, int n)
{
	if(m <= n){
		if(m == n){ 
			return 0;
		} else if(m<2*(n-m)) {
			// In this case the fastest projection is onto the image, so that is created and saved.
			return 1;
		} else {
			// In this case the fastest projection is onto the kernel.
			return 2;
		}
	}
	return -1;
}

void * ipCacheDataCreator(void * input)
{
	struct ipCacheData *myBasis = malloc(sizeof(struct ipCacheData) );
	myBasis->type = -1;
	return myBasis;
}

void ipCacheDataModifier(void * input, void * data)
{
	#ifdef DEBUG 
		printf("-------------------ipCacheDataModifier--------------------------------\n");
	#endif
	struct ipCacheData *myBasis;
	myBasis = (struct ipCacheData *) data;
	
	if(myBasis->type == -1){
		struct ipCacheInput *myInput;
		myInput = (struct ipCacheInput *) input;
		nnLayer *layer = myInput->info->layer;
		const ipMemory *mb = myInput->mb;
		//fillIntersection(myInput->key, layer, mb->I);
		MKL_INT m = mb->I->numHps;
		MKL_INT n = layer->inDim;

		myBasis->type = detectType(m,n);
		//If there's less included hyperplanes there will be a kernel
		switch(myBasis->type){
			case 0:
				myBasis->solution = malloc(n*sizeof(float));
				myBasis->projection = NULL;
				break;
			case 1:
				// In this case the fastest projection is onto the image, so that is created and saved.
				
				myBasis->projection = malloc(m*n*sizeof(float));
				myBasis->solution = malloc(m*sizeof(float));
				break;
			case 2:
				// In this case the fastest projection is onto the kernel.
				myBasis->projection = malloc(((n-m)*n)*sizeof(float));
				myBasis->solution = malloc(n*sizeof(float));
				break;
			default:
				printf("myBasis type is not one of the designated types\n");
				exit(-1);
		}
		solve(mb->I->subA, m, n, mb->I->subB,myBasis,mb);	
	}
	
	#ifdef DEBUG 
		printf("-------------------/ipCacheDataCreator--------------------------------\n");
	#endif
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
			memcpy(cache->hpOffsetVecs+i*inDim, cache->hpNormals+i*inDim, inDim*sizeof(float));
		}
	}
	#ifdef DEBUG
		printf("---------------------------------\n");
	#endif
}

ipCache * allocateCache(nnLayer *layer, float threshhold, long long int freeMemory)
{
	ipCache *cache = malloc(sizeof(ipCache));
	uint keyLen = calcKeyLen(layer->outDim);
	uint minMN = layer->outDim;
	if(layer->inDim < layer->outDim){
		minMN = layer->inDim;
	}
	int maxProjection = layer->inDim*layer->outDim;
	if(3*layer->outDim > 2*layer->inDim){
		maxProjection = (2*layer->inDim*layer->inDim)/3;
	}

	long int maxMemory = (7*(freeMemory))/(8*sizeof(float));

	cache->bases = createTree(keyLen, minMN, maxProjection, maxMemory, ipCacheDataCreator, ipCacheDataModifier, ipCacheDataDestroy);
	int rc = pthread_mutex_init(&(cache->balanceLock), 0);
	if (rc != 0) {
        printf("balanceLock Initialization failed at");
	}
	printf("Creating cache of inDim %d and outDim %d with threshhold %f \n", layer->inDim, layer->outDim, threshhold);
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


float computeDist(float * p, kint *ipSignature, ipCache *cache, const int rank, ipMemory *mb)
{
	MKL_INT inDim = cache->layer->inDim;
	int memoryUseage = 0;
	//If there's less included hyperplanes there will be a kernel
	switch(detectType(rank,inDim)){
		case 0:
			memoryUseage = inDim;
			break;
		case 1:
			// In this case the fastest projection is onto the image, so that is created and saved.
			memoryUseage = rank*(inDim+1);
			break;
		case 2:
			// In this case the fastest projection is onto the kernel.
			memoryUseage = inDim*(inDim-rank+1);
			break;
		default:
			printf("myBasis type is not one of the designated types\n");
			exit(-1);
	}
	ipCacheInput myInput = {.info = cache, .key = ipSignature, .mb = mb};
	struct ipCacheData *myBasis = addData(cache->bases, ipSignature,rank-1, &myInput, memoryUseage);
	
	//struct ipCacheData *myBasis = ipCacheDataCreator(&myInput);
	
	const MKL_INT oneI = 1;
	const MKL_INT inout = (inDim-rank);
	const float oneF = 1.0;
	const float negoneF = -1.0;
	const char N = 'N';

	if(myBasis){
		float norm;
		switch(myBasis->type){
			case 0:
				memcpy(mb->px, myBasis->solution, inDim*sizeof(float));
				#ifdef DEBUG
					printf("Type 0, inDim: %d", inDim);
				#endif
				saxpy(&inDim, &negoneF, p, &oneI, mb->px, &oneI);
				#ifdef DEBUG
					printf("Final px: ");
					printFloatArr(mb->px, rank);
				#endif
				norm = cblas_snrm2 (inDim, mb->px, 1);
				break;
			case 1:
				memcpy(mb->px, myBasis->solution, rank*sizeof(float));
				#ifdef DEBUG
					printf("Type 1, inDim: %d, rank: %d", inDim, rank);
				#endif
				//cblas_sgemv (CblasRowMajor, CblasNoTrans, rank, inDim,1,myBasis->projection, inDim, mb->px, 1, 0, mb->py, 1);
				//printMatrix(myBasis->projection,inDim,rank);
				sgemm (&N,&N, &oneI, &rank, &inDim, &oneF, p, &oneI, myBasis->projection, &inDim, &negoneF,mb->px,&oneI);
				//sgemv(&T, &inDim, &rank, &oneF, myBasis->projection, &inDim, mb->px, &oneI, &zeroF, mb->px, &oneI);
				/*for(int i = 0; i< rank; i++){
					mb->py[i] = cblas_sdot (inDim, myBasis->projection + i*inDim, 1, mb->px, 1);
				}*/
				
				norm = cblas_snrm2 (rank, mb->px, 1);
				break;
			case 2:	
				memcpy(mb->px, myBasis->solution, (inDim)*sizeof(float));
				memset(mb->py,0, (inDim-rank)*sizeof(float));
				saxpy(&inDim, &negoneF, p, &oneI, mb->px, &oneI);
				#ifdef DEBUG
					printf("Type 2, inDim: %d, rank: %d", inDim,rank);
				#endif
				//cblas_sgemv (CblasRowMajor, CblasNoTrans, outDim, inDim,1,myBasis->projection, inDim, p, 1, 1, mb->px, 1);
				sgemm (&N,&N, &oneI, &inout, &inDim, &oneF, mb->px, &oneI, myBasis->projection, &inDim, &negoneF,mb->py,&oneI);
				printFloatArr(mb->py, inDim-rank);
				//sgemv (&N, &inout, &inDim, &oneF, myBasis->projection, &inout, mb->py, &oneI, &negoneF,mb->px,&oneI);
	
				cblas_sgemv (CblasRowMajor, CblasTrans, inDim-rank, inDim,1,myBasis->projection, inDim, mb->py, 1, -1, mb->px, 1);
				norm = cblas_snrm2 (inDim, mb->px, 1);
				break;
			default:
				printf("myBasis type is not one of the designated types\n");
				exit(-1);
		}
		
		if(norm < 0){
			norm = -norm;
		}
		return norm;
	} else {
		return -1;
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


void computeDistToHPS(float *p, ipCache *cache, distanceWithIndex *distances, ipMemory *mb)
{
	uint outDim = cache->layer->outDim;
	uint inDim = cache->layer->inDim;
	memcpy(mb->localCopy, cache->hpOffsetVecs, inDim*outDim*sizeof(float));
	
	#ifdef DEBUG
		printf("--------computeDistToHPS---------\n");
	#endif

	float curDist = 0;

	for(uint i =0;i<outDim;++i){
		distances[i].index = i;
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
		cblas_saxpy (inDim,-1,p,1,mb->localCopy + i*inDim,1);
		curDist = cblas_sdot (inDim, mb->localCopy + i*inDim, 1, cache->hpNormals + i*inDim, 1);
		if(curDist == 0){
			// Sometimes the normal vector is 0, ie, this hyperplane is not used
			// We check for this, and if so, essentially remove this distance from the situation.
			float norm = cblas_snrm2 (inDim, cache->hpNormals + i*inDim, 1);
			if(norm == 0){
				distances[i].dist = FLT_MAX;
			} else {
				distances[i].dist = 0;
			}
		} else if(curDist < 0){
			distances[i].dist = -curDist;
		} else {
			distances[i].dist = curDist;
		}
		
		#ifdef DEBUG
			printf("p + offset vector: ");
			if(inDim < 10){
				printFloatArr(mb->localCopy+i*inDim,inDim);
			}
			if(distances[i].dist == FLT_MAX){
				printf("Final distance is massive as the hp does not exist\n");
			} else {
				printf("Final distance (normal dotted with p+offset): %f \n",distances[i].dist);
			}
			printf("---------\n");
		#endif
	}
	qsort(distances, outDim, sizeof(distanceWithIndex), distOrderingCmp);
	
	#ifdef DEBUG
		for(uint i = 0; i< outDim; i++){
			printf("Distance: %f, index: %d\n", distances[i].dist, distances[i].index);
		}
		printf("---------------------------------\n");
	#endif

	
}



void getInterSig(ipCache * cache, float *p, kint * ipSignature, ipMemory *mb)
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

	computeDistToHPS(p, cache, mb->distances, mb);
	fillIntersectionOrdered(mb->distances, numRelevantHP-1, cache->layer, mb->I);
	for(i=0;i<numRelevantHP;i++){
		if(mb->distances[i].dist != FLT_MAX){
			#ifdef DEBUG
				printf("%d,",mb->distances[i].index);
			#endif
			addIndexToKey(ipSignature,mb->distances[i].index);
		} 
	}
	
	float posetDist;
	float hpDist;
	
	
	i = numRelevantHP - 1;
	removeIndexFromKey(ipSignature,mb->distances[i].index);
	mb->I->numHps = i;
	hpDist = mb->distances[i].dist;
	posetDist = computeDist(p, ipSignature, cache, i, mb);
	#ifdef DEBUG
		printf("Loop %d \n",i);
		printf("Checking xu{i}(dist:%f) vs %f*{j}(dist:%f)\n",posetDist,cache->threshold,hpDist);
		printf("Difference between poset and hp distance:%f\n",cache->threshold*posetDist - hpDist);
		printf("xu{i}:");
		printKey(ipSignature,outDim);
		printf("i: %u \n", mb->distances[i-1].index);
		printf("j: %u \n", mb->distances[i].index);
	#endif

	while(cache->threshold*posetDist - hpDist > 0 && i > 1) {
		i--;
		removeIndexFromKey(ipSignature,mb->distances[i].index);
		mb->I->numHps = i;
		hpDist = mb->distances[i].dist;
		posetDist = computeDist(p, ipSignature, cache, i, mb);
		
		#ifdef DEBUG
			printf("Loop %d \n",i);
			printf("Checking xu{i}(dist:%f) vs %f*{j}(dist:%f)\n",posetDist,cache->threshold,hpDist);
			printf("Difference between poset and hp distance:%f\n",cache->threshold*posetDist - hpDist);
			printf("xu{i}:");
			printKey(ipSignature,outDim);
			printf("i: %u \n", mb->distances[i-1].index);
			printf("j: %u \n", mb->distances[i].index);
		#endif
		
		
	} 
	if(i == 1){
		posetDist = mb->distances[0].dist;
		hpDist = mb->distances[1].dist;
		#ifdef DEBUG
			printf("Loop %d \n",i);
			printf("Checking {i}(dist:%f) vs %f*{j}(dist:%f)\n",posetDist,cache->threshold,hpDist);
			printf("Difference between poset and hp distance:%f\n",cache->threshold*posetDist - hpDist);
			printf("{i}:");
			printKey(ipSignature,outDim);
			printf("i: %u \n", mb->distances[0].index);
			printf("j: %u \n", mb->distances[1].index);
		#endif
		if(cache->threshold*posetDist - hpDist > 0){
			removeIndexFromKey(ipSignature,mb->distances[0].index);;
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
	int * joinCondition = myargs->joinCondition;

	ipCache *cache = myargs->cache;

	uint dim = cache->layer->inDim;
	uint keySize = calcKeyLen(cache->layer->outDim);
	ipMemory * mb = allocateIPMemory(cache->layer->inDim,cache->layer->outDim);
	uint i = 0;
	for(i=tid;i<numData;i=i+numThreads){		
		getInterSig(cache,data+i*dim, ipSignature+i*keySize, mb);
		joinCondition[tid] = 1;
		//printf("Thread %d at mutex with %u nodes \n",tid,cache->bases->numNodes);
		pthread_mutex_lock(&(cache->balanceLock));
			if(cache->bases->currentMemoryUseage > cache->bases->maxTreeMemory){
				//printf("Balancing Tree in thread %d. Current number of nodes: %d.\n",tid,cache->bases->numNodes);
				checkJoinCondition(joinCondition,numThreads);
				balanceAndTrimTree(cache->bases, 4*cache->bases->maxTreeMemory/5);
			}
		pthread_mutex_unlock(&(cache->balanceLock));
		joinCondition[tid] = 0;
	}
	freeIPMemory(mb);
	pthread_exit(NULL);
}

void getInterSigBatch(ipCache *cache, float *data, kint *ipSignature, uint numData, uint numProc)
{
	int maxThreads = numProc;
	int rc =0;
	int i =0;
	printf("ipCalc working on %u data points, with %u threads and %ld free memory\n",numData,maxThreads,cache->bases->maxTreeMemory);
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
		joinCondition[i] = 0;
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
	free(joinCondition);
	free(thread_args);
}