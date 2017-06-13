#include "adaptiveTools.h"

/*
static unsigned int uintlog2 (unsigned int val) {
    if (val == 0) return UINT_MAX;
    if (val == 1) return 0;
    unsigned int ret = 0;
    while (val > 1) {
        val >>= 1;
        ret++;
    }
    return ret;
}
*/


static inline uint32_t uintlog2(const uint32_t x) {
  uint32_t y;
  asm ( "\tbsr %1, %0\n"
      : "=r"(y)
      : "r" (x)
  );
  return y;
}


int getNextIPGroup(mapTreeNode ** locArr, int startingIndex, int maxLocIndex, vector * ipGroup)
{
	if(startingIndex >= maxLocIndex){
		return maxLocIndex;
	}
	kint * curIpSig = locArr[startingIndex]->ipKey;
	int keyLen = locArr[startingIndex]->createdKL;
	#ifdef DEBUG
		printf("--------------\n");

		printf("Creating an ip group starting at %d with key (%p):\n",startingIndex,curIpSig);
		printKeyArr(curIpSig, keyLen);
	#endif
	while(startingIndex < maxLocIndex && compareKey(curIpSig,locArr[startingIndex]->ipKey,keyLen) == 0){
		#ifdef DEBUG
			printf("Checking next, index at %d with key (%p):\n",startingIndex,locArr[startingIndex]->ipKey);
			printKeyArr(locArr[startingIndex]->ipKey, keyLen);
			if(compareKey(curIpSig,locArr[startingIndex]->ipKey,keyLen) != 0){
				printf("Keys are different. Stopping.\n");
			}
		#endif
		vector_add(ipGroup, locArr[startingIndex]);
		startingIndex++;
		
	}
	#ifdef DEBUG
		printf("ipGroup Pointers\n");
		vector_print_pointers(ipGroup);
		printf("-------------\n");

	#endif
	return startingIndex;
}

void getHyperplanesCrossed(vector * group, kint *hpCrossSig)
{
	kint *seedSig = ((mapTreeNode *)vector_get(group,0))->regKey;
	kint *compSig;
	int keyLen = ((mapTreeNode *)vector_get(group,0))->createdKL;
	memset(hpCrossSig,0,keyLen*sizeof(kint));
	int total = vector_total(group);
	int i = 0, j = 0;
	for(i = 1; i< total;i++){ // Already have the first one
		compSig = ((mapTreeNode *)vector_get(group,i))->regKey;
		#ifdef DEBUG
			printf("<<>>\n");
			printf("Computing the cornerDim\n");
			printf("seedSig: %p\n",seedSig);
			printKey(seedSig, 32*keyLen);
			printf("crossedSig: %p\n",seedSig);
			printKey(hpCrossSig, 32*keyLen);
			printf("compSig: %p\n",seedSig);
			printKey(compSig, 32*keyLen);
		#endif
		for(j = 0; j<keyLen;j++){
			hpCrossSig[j] |= (seedSig[j] ^ compSig[j]);
		}
	}
}

int getGroupCornerDim(vector * group, kint *hpCrossSigTemp)
{	
	mapTreeNode * seedLoc = (mapTreeNode *)vector_get(group,0);
	int keyLen = seedLoc->createdKL;
	kint *ipSig = seedLoc->ipKey;
	#ifdef DEBUG
		printf("<<<<<<<<>>>>>>>>\n");
		printf("Computing the cornerDim\n");
		printf("ipSig: %p keyLen %d\n",seedLoc->ipKey,keyLen);
		printKey(ipSig, 32*keyLen);
	#endif
	uint maxDim = numberOfOneBits(ipSig, keyLen);	

	
	getHyperplanesCrossed(group,hpCrossSigTemp);
	uint numHpCrossed = numberOfOneBits(hpCrossSigTemp, keyLen);
	uint total = vector_total(group);
	#ifdef DEBUG
		printf("hpCrossSigTemp:\n");
		printKey(hpCrossSigTemp, 32*keyLen);
		printf("Number of one bits in ipKey: %u, in hpCrossedKey:%u, and num locs: %u \n", maxDim,numHpCrossed,total);
	#endif
	#ifdef DEBUG
		printf("<<<<<<<<>>>>>>>>\n");
	#endif
	if(uintlog2(total) == numHpCrossed){
		return (int)maxDim - numHpCrossed;
	} else {
		return -1;
	}
	
}

int getGroupPop(vector * subGroup)
{
	int groupCount = vector_total(subGroup);
	int curPopCount = 0;
	mapTreeNode * curLoc;

	for(int j = 0; j<groupCount; j++){
		curLoc = (mapTreeNode *) vector_get(subGroup,j);
		curPopCount += location_total(&(curLoc->loc),1);   // We want the points classified as errors
	}
	
	return curPopCount;
}

void createGroup(vector * ipCollection, vector * subGroup, float *selectionVec, float selectionBias,int n)
{
	mapTreeNode * seedLoc = (mapTreeNode *)vector_get(ipCollection,0);
	int keyLen = calcKeyLen(n);
	vector_delete(ipCollection,0);
	int selection = evalSig(seedLoc->regKey, selectionVec, selectionBias, n);
	vector_add(subGroup,seedLoc);
	int lastCombine = 1;
	mapTreeNode * curBaseLoc;
	mapTreeNode * compareLoc;

	int j = 0,k = 0;

	#ifdef DEBUG
		printf("---------\n");

		printf("Creating a group out of IP group\n");
		printf("ipGroup Pointers (minus seedLoc)\n");
		vector_print_pointers(ipCollection);
	
		printf("Seed Location (%p). Response: %d regSig:\n",seedLoc,selection);
		printKeyArr(seedLoc->regKey, keyLen);
	#endif
	while(lastCombine && vector_total(ipCollection) > 0){
		lastCombine = 0;
		#ifdef DEBUG
			printf("-----\n");
			printf("Starting a combine run.\n");
			printf("Current group size: %d\n",vector_total(subGroup));
			vector_print_pointers(subGroup);
		#endif
		for(j = 0;j < vector_total(subGroup); j++){
			curBaseLoc = (mapTreeNode *)vector_get(subGroup,j);
			#ifdef DEBUG
				printf("=====\n");
				printf("Combining to a base location. ");
				printf("CurrentBase Location. Response: %d regSig:\n",evalSig(curBaseLoc->regKey, selectionVec, selectionBias, n));
				printKeyArr(curBaseLoc->regKey, keyLen);
				printf("Remaining ip group size %d\n", vector_total(ipCollection));
			#endif
			for(k = 0; k < vector_total(ipCollection); k++){
				printf("===\n");

				compareLoc = (mapTreeNode *)vector_get(ipCollection,k);

				#ifdef DEBUG
					printf("Current IP Collection\n");
					vector_print_pointers(ipCollection);
					printf("Current subgroup\n");
					vector_print_pointers(subGroup);
					printf("CompareLoc Pointer (%p), index %d\n",compareLoc,k );
					printf("compareLoc Location. Response: %d regSig:\n",evalSig(compareLoc->regKey, selectionVec, selectionBias, n));
					printKeyArr(compareLoc->regKey, keyLen);
					printf("Off by one: %d\n", offByOne(curBaseLoc->regKey, compareLoc->regKey,keyLen));
				#endif

				if(offByOne(curBaseLoc->regKey, compareLoc->regKey,keyLen) && 
					evalSig(compareLoc->regKey, selectionVec, selectionBias, n) == selection){
					vector_delete(ipCollection,k);
					k--;
					vector_add(subGroup,compareLoc);
					#ifdef DEBUG
						printf("Adding compareLoc to subgroup.\n");
					#endif
					
					lastCombine = 1;
				}
			}
		}
		#ifdef DEBUG
			printf("End of a combine run. Combine Tracker: %d\n",lastCombine);
			printf("-----\n");
		#endif
	}
	#ifdef DEBUG
		printf("---------\n");
	#endif
}

mapTreeNode ** unpackGroup(vector * group)
{
	int total = vector_total(group);
	mapTreeNode ** ret = malloc(total*sizeof(mapTreeNode *));
	for(int i =0; i < total; i++){
		ret[i] = (mapTreeNode *)vector_get(group,i);
	}
	return ret;
}

void freeMaxPopGroupData(maxPopGroupData * group)
{
	if(group){
		if(group->locations){
			free(group->locations);
		}
		if(group->hpCrossed){
			free(group->hpCrossed);
		}
		free(group);
	}
}

maxPopGroupData * refineMapAndGetMax(mapTreeNode ** locArr, int maxLocIndex, nnLayer * selectionLayer)
{
	float *selectionMat = selectionLayer->A;
	float *selectionBias = selectionLayer->b;
	int n = selectionLayer->inDim;
	int m = selectionLayer->outDim;

	int keyLen = locArr[0]->createdKL;
	kint * hpCrossSigTemp = malloc(keyLen * sizeof(kint));
	
	maxPopGroupData * maxData = malloc(sizeof(maxPopGroupData));
	maxData->hpCrossed = malloc(keyLen * sizeof(kint));
	maxData->count = 0;

	vector curIP;
	vector curGroup;
	vector maxGroup;
	vector_init(&curIP);
	vector_init(&curGroup);
	vector_init(&maxGroup);

	int i = 0;
	int j = 0;

	

	int cornerDim = 0;
	for(i = 0; i < m; i++){
		#ifdef DEBUG
			printf("------------------------------------------------------------\n");
			printf("On selection %d with bias %f\n",i,selectionBias[i]);
			printf("------------------------------------------------------------\n");

		#endif
		j = 0;
		j = getNextIPGroup(locArr,j,maxLocIndex,&curIP);
		while(j<maxLocIndex){
			#ifdef DEBUG
				printf("===========================\n");
				printf("On Created an ip group, ending in index %d\n",j);
				printf("===========================\n");

			#endif
			while(vector_total(&curIP) > 0){
				#ifdef DEBUG
					printf("=============\n");
					printf("Creating a group to check\n");
					printf("=============\n");
				#endif
				createGroup(&curIP,&curGroup,selectionMat + i*n, selectionBias[i],n);
				cornerDim = getGroupCornerDim(&curGroup,hpCrossSigTemp);
				#ifdef DEBUG
					printf("=\n");
					printf("Corner Dim: %d\n",cornerDim);
					printf("=\n");
				#endif
				if(cornerDim > 0){
					if(maxData->count < (cornerDim * getGroupPop(&curGroup))){
						#ifdef DEBUG
							printf("Switching maxGroup\n");

						#endif
						maxData->count = cornerDim * getGroupPop(&curGroup);
						maxData->selectionIndex = i;
						memcpy(maxData->hpCrossed,hpCrossSigTemp,keyLen * sizeof(kint));
						vector_copy(&maxGroup,&curGroup);
					}
				}
				vector_reset(&curGroup);
			}
			vector_reset(&curIP);
			j = getNextIPGroup(locArr,j,maxLocIndex,&curIP);
		}
	}
	#ifdef DEBUG
		printf("=============\n");
		printf("Final Group:\n");
	#endif
	maxData->locations = unpackGroup(&maxGroup);
	#ifdef DEBUG
		printf("\n");
		printf("=============\n");
	#endif
	vector_free(&curIP);
	vector_free(&curGroup);
	vector_free(&maxGroup);
	free(hpCrossSigTemp);

	return maxData;
}



vector * getRegSigs(mapTreeNode ** locArr, int numNodes)
{
	vector * regSigs = malloc(sizeof(vector));
	vector_init(regSigs);
	qsort(locArr, numNodes, sizeof(mapTreeNode *), regOrder);
	int i = 0;
	uint keyLen = locArr[0]->createdKL;
	kint * currentSig = locArr[i]->regKey;
	vector_add(regSigs,currentSig);
	for(i=0;i<numNodes;i++){
		if(compareKey(locArr[i]->regKey, currentSig, keyLen)){
			currentSig = locArr[i]->regKey;
			vector_add(regSigs,currentSig);
		}
	}
	return regSigs;
}

void unpackRegSigs(vector * regSigs, uint dim, float * unpackedSigs)
{
	int i = 0;
	int total = vector_total(regSigs);
	for(i=0;i<total;i++){
		convertFromKeyToFloat(vector_get(regSigs,i), unpackedSigs + i*dim, dim);
	}
}

// Produces a vector that always point towards the corner in question, and is located to intersect with the average error given
void createNewHPVec(maxPopGroupData * maxErrorGroup, float * avgError, float *solution, float *newHPVec, float *offset, float *A, float *b, uint inDim, uint outDim)
{
	uint keyLen = calcKeyLen(outDim);
	kint * ipKey = maxErrorGroup->locations[0]->ipKey;
	int numVectors = 0;
	uint i = 0;
	uint j = 0;
	
	// Collect the relevant vectors
	kint * relevantHPKey = calloc(keyLen,sizeof(kint));
	float * relevantVec = malloc(inDim*outDim*sizeof(float));
	for(i = 0;i < keyLen;i++){
		relevantHPKey[i] = ipKey[i] ^ maxErrorGroup->hpCrossed[i];
	}
	for(i = 0; i< outDim; i++){
		if(checkIndex(relevantHPKey, i)){
			memcpy(relevantVec + numVectors*inDim, A + i*inDim,inDim*sizeof(float));
			numVectors++;
		}
	}

	// Get them pointing the right way
	float * shiftedAvgError = malloc(inDim * sizeof(float));
	memcpy(shiftedAvgError,avgError,inDim*sizeof(float));
	cblas_saxpy(inDim,1,solution,1,shiftedAvgError,1);
	for(i = 0; i< numVectors;i++){
		if(cblas_sdot(inDim,shiftedAvgError,1,relevantVec + i*inDim,1) > 0){
			// Pointing toward the avg error, should be away from
			cblas_sscal(inDim, -1.0, relevantVec + i*inDim, 1);
		}
	}

	// Take the average direction
	float norm;
	memset(newHPVec, 0, inDim*sizeof(float));
	for(i=0;i<numVectors;i++){
		norm = cblas_snrm2(inDim, relevantVec + i*inDim,1);
		cblas_saxpy(inDim,norm/numVectors,solution,1,newHPVec,1);
	}

	// Get the offset and save it.

	offset[0] = - cblas_sdot(inDim,shiftedAvgError,1,newHPVec,1);
}