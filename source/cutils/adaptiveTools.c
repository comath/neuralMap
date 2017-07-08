#include "adaptiveTools.h"


// Takes the integral Log_2 of the provided unsigned int. Copied from StackOverflow.
static inline uint32_t uintlog2(const uint32_t x) {
  uint32_t y;
  asm ( "\tbsr %1, %0\n"
      : "=r"(y)
      : "r" (x)
  );
  return y;
}

/*
Creates a group of all regions near a intersection. locArr has to be provided in ipSig primary order. (Lexicographically with ipKey first.)

Inputs: startingIndex, maxLocIndex, locArr
Outputs: ipGroup
Returns: the final index
*/

int getNextIPGroup(mapTreeNode ** locArr, int startingIndex, int maxLocIndex, vector * ipGroup)
{
	if(startingIndex >= maxLocIndex){
		return maxLocIndex;
	}
	kint * curIpSig = locArr[startingIndex]->ipKey;
	int keyLen = locArr[startingIndex]->createdKL;
	#ifdef DEBUG
		printf("--------------\n");

		printf("Creating an ip group starting at %d with key (%p): ",startingIndex,curIpSig);
		printKey(curIpSig, 32*keyLen);
	#endif
	while(startingIndex < maxLocIndex && compareKey(curIpSig,locArr[startingIndex]->ipKey,keyLen) == 0){
		#ifdef DEBUG
			printf("Checking next, index at %d with key (%p): ",startingIndex,locArr[startingIndex]->ipKey);
			printKey(locArr[startingIndex]->ipKey, 32*keyLen);
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

// Forms a signature that indicates which hyperplanes are crossed.

void getHyperplanesCrossed(vector * group, kint *hpCrossSig)
{
	kint *seedSig = ((mapTreeNode *)vector_get(group,0))->regKey;
	kint *compSig;
	int keyLen = ((mapTreeNode *)vector_get(group,0))->createdKL;
	memset(hpCrossSig,0,keyLen*sizeof(kint));
	int total = vector_total(group);
	int i = 0, j = 0;
	#ifdef DEBUG
		printf("<<>>\n");
		printf("Getting List of hp crossed\n");
	#endif
	for(i = 1; i< total;i++){ // Already have the first one
		compSig = ((mapTreeNode *)vector_get(group,i))->regKey;
		#ifdef DEBUG
			printf("seedSig: %p",seedSig);
			printKey(seedSig, 32*keyLen);
			printf("crossedSig: %p",seedSig);
			printKey(hpCrossSig, 32*keyLen);
			printf("compSig: %p",seedSig);
			printKey(compSig, 32*keyLen);
		#endif
		for(j = 0; j<keyLen;j++){
			/*
			The XOR of the seed and the compSig is the difference between the two regions.
			We want to add that to the hpCrossedSig, so we OR that with with what we have so far. 
			*/ 
			hpCrossSig[j] |= (seedSig[j] ^ compSig[j]);
		}
	}
	#ifdef DEBUG
		printf("<<>>\n");
	#endif
}

// For a given group, this computes the corner dimension. Returns -1 for groups that do not form a corner.

int getGroupCornerDim(vector * group, kint *hpCrossSigTemp)
{	
	mapTreeNode * seedLoc = (mapTreeNode *)vector_get(group,0);
	int keyLen = seedLoc->createdKL;
	kint *ipSig = seedLoc->ipKey;
	#ifdef DEBUG
		printf("<<<<<<<<>>>>>>>>\n");
		printf("Computing the cornerDim\n");
		printf("ipSig (%p): ",seedLoc->ipKey);
		printKey(ipSig, 32*keyLen);
	#endif
	uint maxDim = numberOfOneBits(ipSig, keyLen);	

	
	getHyperplanesCrossed(group,hpCrossSigTemp);
	uint numHpCrossed = numberOfOneBits(hpCrossSigTemp, keyLen);
	uint total = vector_total(group);
	#ifdef DEBUG
		printf("hpCrossSigTemp: ");
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

// For a group computes the total error.

int getGroupErrorPop(vector * subGroup)
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


/*
For a given ip group, this creates a group of locations that are connected (their union forms a nearly connected component)
and share a selection value for the provided selector.

The ipCollection is modified, The elements of the subGroup are removed from the ipCollection.

inputs: ipCollection, selectionVec, selctionBias,n
outputs, subGroup
*/
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
	
		printf("Seed Location (%p). Response: %d regSig: ",seedLoc,selection);
		printKey(seedLoc->regKey, 32*keyLen);
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
				printf("CurrentBase Location. Response: %d regSig: ",evalSig(curBaseLoc->regKey, selectionVec, selectionBias, n));
				printKey(curBaseLoc->regKey, 32*keyLen);
				printf("Remaining ip group size %d\n", vector_total(ipCollection));
			#endif
			for(k = 0; k < vector_total(ipCollection); k++){
				#ifdef DEBUG
					printf("===\n");
				#endif
				compareLoc = (mapTreeNode *)vector_get(ipCollection,k);

				#ifdef DEBUG
					printf("Current IP Collection\n");
					vector_print_pointers(ipCollection);
					printf("Current subgroup\n");
					vector_print_pointers(subGroup);
					printf("CompareLoc Pointer (%p), index %d\n",compareLoc,k );
					printf("compareLoc Location. Response: %d regSig: ",evalSig(compareLoc->regKey, selectionVec, selectionBias, n));
					printKey(compareLoc->regKey, 32*keyLen);
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

// Converts a group vector into an array.
mapTreeNode ** unpackGroup(vector * group)
{
	int total = vector_total(group);
	mapTreeNode ** ret = malloc(total*sizeof(mapTreeNode *));
	for(int i =0; i < total; i++){
		ret[i] = (mapTreeNode *)vector_get(group,i);
		#ifdef DEBUG
			printf("Node %d (%p), with error location count %d",i,ret[i],ret[i]->loc.total_error);
		#endif
	}
	return ret;
}

void freeMaxErrorCorner(maxErrorCorner * group)
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



maxErrorCorner * refineMapAndGetMax(mapTreeNode ** locArr, int maxLocIndex, nnLayer * selectionLayer)
{
	float *selectionMat = selectionLayer->A;
	float *selectionBias = selectionLayer->b;
	int n = selectionLayer->inDim;
	int m = selectionLayer->outDim;

	int keyLen = locArr[0]->createdKL;
	kint * hpCrossSigTemp = malloc(keyLen * sizeof(kint));
	
	maxErrorCorner * maxData = malloc(sizeof(maxErrorCorner));
	maxData->selectionIndex = -1;
	maxData->hpCrossed = malloc(keyLen * sizeof(kint));
	maxData->weightedCount = 0;
	maxData->locCount = 0;


	vector curIP;
	vector curGroup;
	vector maxGroup;
	vector_init(&curIP);
	vector_init(&curGroup);
	vector_init(&maxGroup);

	int i = 0;
	int j = 0;

	

	int cornerDim = 0;
	int errorPop = 0;
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
				errorPop = getGroupErrorPop(&curGroup);
				#ifdef DEBUG
					printf("=\n");
					printf("Corner Dim: %d, Error population: %d, Old weighted count: %d\n",cornerDim,errorPop,maxData->weightedCount);
					printf("=\n");
				#endif
				if(cornerDim > 1){
					if(maxData->weightedCount < (cornerDim * cornerDim * errorPop)){
						#ifdef DEBUG
							printf("Switching maxGroup\n");

						#endif
						maxData->locCount = vector_total(&curGroup);
						maxData->weightedCount = cornerDim * cornerDim * errorPop;
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

// Aquires the region signatures from the total list of locations. 
vector * getRegSigs(mapTreeNode ** locArr, int numNodes)
{
	#ifdef DEBUG
		printf("extracting regSigs from location array with initial pointer %p\n", locArr);
	#endif
	vector * regSigs = malloc(sizeof(vector));
	vector_init(regSigs);
	qsort(locArr, numNodes, sizeof(mapTreeNode *), regOrder);
	int i = 0;
	uint keyLen = locArr[0]->createdKL;
	kint * currentSig = locArr[0]->regKey;
	vector_add(regSigs,currentSig);
	for(i=1;i<numNodes;i++){
		#ifdef DEBUG
			printf("On %p, the %dth one. RegKey:", locArr[i],i);
			printKey(locArr[i]->regKey,keyLen*32);
			printf("ipKey:");
			printKey(locArr[i]->ipKey,keyLen*32);
		#endif
		if(compareKey(locArr[i]->regKey, currentSig, keyLen)){
			#ifdef DEBUG
				printf("Accepted\n");
			#endif			
			currentSig = locArr[i]->regKey;
			vector_add(regSigs,currentSig);
		}
		#ifdef DEBUG
			else {
				printf("Denied\n");
			}
		#endif
	}
	return regSigs;
}

// Should be improved, slow search algo
int checkIfListed(vector *listedRegSigs, kint *regSig,uint keyLength)
{
	int i = 0;
	int total = vector_total(listedRegSigs);
	for(i=0; i<total; i++){
		if(compareKey((kint *)vector_get(listedRegSigs,i), regSig, keyLength) == 0){
			return 1;
		}
	}
	return 0;
}

void createData(maxErrorCorner *maxErrorGroup, nnLayer *selectionLayer, vector *regSigs, float *unpackedSigs, int * labels)
{
	int selectionIndex = maxErrorGroup->selectionIndex;
	nnLayer selector;
	selector.inDim = selectionLayer->inDim;
	uint dim = selectionLayer->inDim;
	selector.outDim = 1;
	selector.A = selectionLayer->A + selectionIndex*dim;
	selector.b = selectionLayer->b + selectionIndex;
	uint i = 0;
	float output;
	kint * regSig;
	uint keyLen = maxErrorGroup->locations[0]->createdKL;
	uint total = vector_total(regSigs);

	vector *importantSigs = getRegSigs(maxErrorGroup->locations, maxErrorGroup->locCount);

	for(i = 0; i < total;i++){
		regSig = (kint*)vector_get(regSigs,i);
		convertFromKeyToFloat(regSig, unpackedSigs + i*(dim+1), dim);
		unpackedSigs[i*(dim+1) + dim] = 0;
		convertFromKeyToFloat(regSig, unpackedSigs + (total+i)*(dim+1), dim);
		unpackedSigs[(i+total)*(dim+1) + dim] = 1;
		#ifdef DEBUG
			printf("Unpacking regSig %u\n", i);
			printf("Created normal data. Final entry: %f (should be 0)\n", unpackedSigs[i*(dim+1) + dim]);
			printf("Created augment data. Final entry: %f (should be 1)\n", unpackedSigs[(i+total)*(dim+1) + dim]);
		#endif
		evalLayer(&selector, unpackedSigs + i*(dim+1), &output);
		if(checkIfListed(importantSigs,regSig,keyLen)){
			// The following creates label data according to the scheme outlined in the main paper.
			if(output > 0){
				labels[i] = 1;
				labels[i+total] = 0;
			} else if(output < 0) {
				labels[i] = 0;
				labels[i+total] = 1;
			} else {
				labels[i] = 0;
				labels[i+total] = 0;
			}
			#ifdef DEBUG
				printf(".....regSig is listed. output: %f, label for normal: %d, label for augment: %d\n", output, labels[i],labels[i+total]);
			#endif
		} else {
			if(output > 0){
				labels[i] = 1;
				labels[i+total] = 1;
			} else if(output < 0) {
				labels[i] = 0;
				labels[i+total] = 0;
			} else {
				labels[i] = 0;
				labels[i+total] = 0;
			}
			#ifdef DEBUG
				printf(".....regSig is NOT listed. output: %f, label for normal: %d, label for augment: %d\n", output, labels[i],labels[i+total]);

			#endif
		}
	}
	vector_free(importantSigs);
	free(importantSigs);
}

float *getSolutionPointer(_nnMap *map)
{
	return map->tc->solution;
}


void getAverageError(maxErrorCorner * maxErrorGroup, float *data, float * avgError, int dim)
{
	location curLoc;
	memset(avgError,0,dim*sizeof(float));
	int i = 0, j = 0;
	for(i = 0;i<maxErrorGroup->locCount;i++){
		curLoc = maxErrorGroup->locations[i]->loc;
		for(j = 0; j< curLoc.total_error; j++){
			cblas_saxpy(dim,1, data + (curLoc.pointIndexes_error[j])*dim,1,avgError,1);
		}
	}
}


// Produces a vector that always point towards the corner in question, and is located to intersect with the average error given
void createNewHPVec(maxErrorCorner * maxErrorGroup, float * avgError, float *solution, nnLayer *hpLayer, float *newHPVec, float *newHPOff)
{
	uint outDim = hpLayer->outDim;
	uint inDim = hpLayer->inDim;
	float * A = hpLayer->A;				// We only need to the normals, we don't need the bias's as they are imbedded in the solution.
	uint keyLen = calcKeyLen(outDim);
	kint * ipKey = maxErrorGroup->locations[0]->ipKey;
	uint numVectors = 0;
	uint i = 0;
	
	// Collect the relevant vectors
	kint * relevantHPKey = calloc(keyLen,sizeof(kint));
	float * relevantVec = calloc(inDim*outDim,sizeof(float));
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
	float * shiftedAvgError = calloc(inDim, sizeof(float));
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
		cblas_saxpy(inDim,norm/numVectors,relevantVec + i*inDim,1,newHPVec,1);
	}

	// Get the offset and save it.
	newHPOff[0] = - cblas_sdot(inDim,shiftedAvgError,1,newHPVec,1);
}

int getSelectionIndex(maxErrorCorner * maxGroup)
{
	return maxGroup->selectionIndex;
}