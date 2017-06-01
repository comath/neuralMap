#include "refineMap.h"
#include <stdint.h>

static unsigned int log2 (unsigned int val) {
    if (val == 0) return UINT_MAX;
    if (val == 1) return 0;
    unsigned int ret = 0;
    while (val > 1) {
        val >>= 1;
        ret++;
    }
    return ret;
}

/*
static inline uint32_t log2(const uint32_t x) {
  uint32_t y;
  asm ( "\tbsr %1, %0\n"
      : "=r"(y)
      : "r" (x)
  );
  return y;
}
*/

int getNextIPGroup(mapTreeNode ** locArr, int startingIndex, int maxLocIndex, vector * ipGroup)
{
	kint * curIpSig = locArr[startingIndex]->ipSig;
	int keylen = locArr[startingIndex]->createdKL;
	while(startingIndex < maxLocIndex && compareKey(curIpSig,locArr[startingIndex]->ipSig,keyLen) == 0){
		vector_add(ipGroup, locArr + startingIndex);
		startingIndex++;
	}
	return startingIndex;
}

void getHyperplanesCrossed(vector * group, kint *hpCrossSig)
{
	kint *seedSig = ((mapTreeNode *)vector_get(group,0))->regSig;
	int keyLen = ((mapTreeNode *)vector_get(group,0))->createdKL;
	memset(0,hpCrossSig,keyLen*sizeof(kint));
	int total = vector_total(group);
	int i = 0, j = 0;
	for(i = 1; i< total;i++){ // Already have the first one
		for(j = 0; j<keyLen;j++){
			hpCrossSig[j] |= (seedSig[j] ^ ((mapTreeNode *)vector_get(group,i))->regSig[j]);
		}
	}
}

int getGroupCornerDim(vector * group, kint *hpCrossSigTemp)
{	
	int keyLen = ((mapTreeNode *)vector_get(group,0))->createdKL;
	kint *ipSig = ((mapTreeNode *)vector_get(group,0))->ipSig;
	uint maxDim = numberOfOneBits(ipSig, keyLen);
	
	getHyperplanesCrossed(group,hpCrossSigTemp);
	uint numHpCrossed = numberOfOneBits(hpCrossSigTemp, keyLen);
	uint total = vector_total(group);
	if(log2(total) == numHpCrossed){
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

	for(j = 0; j<groupCount; j++){
		curLoc = (mapTreeNode *) vector_get(curGroup,i);
		curPopCount += cornerDim*vector_total(curLoc->loc);
	}
	
	return curPopCount;
}

void createGroup(vector * ipCollection, vector * subGroup, float *selectionVec, float selectionBias,int n)
{
	mapTreeNode * seedLoc = (mapTreeNode *)vector_get(&ipCollection,0);
	vector_delete(&ipCollection,0);
	int selection = evalSig(seedLoc->regSig, selectionVec, selectionBias, n);
	vector_add(curGrouping,seedLoc);
	int lastCombine = 1;
	mapTreeNode * curBaseLoc;
	mapTreeNode * compareLoc;

	while(lastCombine){
		lastCombine = 0;
		for(j = 0;j < vector_total(curGrouping); j++){
			curBaseLoc = (mapTreeNode *)vector_get(curGrouping,j);

			for(k = 0; k < vector_total(&ipCollection); k++){
				compareLoc = (mapTreeNode *)vector_get(&ipCollection,k);
				
				if(offByOne(curBaseLoc->regSig,compareLoc->regSig,keyLen) && 
					evalSig(compareLoc->regSig, selectionVec, selectionBias, n) == selection){
					vector_delete(&ipCollection,k);
					k--;
					vector_add(curGrouping,compareLoc);
					lastCombine = 1;
				}
			}
		}
	}
}

maxPopGroupData * refineMapAndGetMax(mapTreeNode ** locArr, int maxLocIndex, float **selectionVec, float *selectionBias, int n, int m)
{
	int keyLen = locArr[0]->createdKL;
	kint * hpCrossSigTemp = malloc(keyLen * sizeof(kint));
	
	maxPopGroupData * maxData = malloc(sizeof(maxPopGroupData))
	maxData->hpCrossed = malloc(keyLen * sizeof(kint));
	maxData->count = 0;

	vector curIP;
	mapTreeNode * seedLoc
	int i = 0;
	int j = 0;
	

	vector_init(&curIP);
	vector_init(&curGrouping);


	int cornerDim = 0;
	i = getNextIPGroup(locArr,i,maxLocIndex,&curIP);
	for(i = 0; i < m; i++){
		while(j<maxLocIndex){
			while(vector_total(&curIP) > 0){
				createGroup(%curIP,&curGrouping,selectionMat + i*n, selectionBias[i],n);
				cornerDim = getGroupCornerDim(&curGrouping,hpCrossSigTemp);
				if(cornerDim > 0 && maxData->count < (cornerDim * getGroupPop(&curGrouping))){
					maxData->count = cornerDim * getGroupPop(&curGrouping);
					maxData->selectionIndex = i;
					memcpy(maxData->hpCrossed,hpCrossSigTemp,keyLen * sizeof(kint));
					vector_copy(&(maxData->locations),maxGroup);
				}
				vector_reset(&curGrouping);
			}
			vector_reset(&curIP);
			j = getNextIPGroup(locArr,i,maxLocIndex,&curIP);
		}
	}
}