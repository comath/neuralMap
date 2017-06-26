#include "selectionTrainer.h"
#include "math.h"

nnLayer expandAndRescaleLayer(nnLayer *selectionLayer, int selectionIndex, float * newSelectionWeights, float *newSelectionBias)
{

	int inDim = selectionLayer->inDim;
	int outDim = selectionLayer->outDim;
	nnLayer newSelectionLayer;
	newSelectionLayer.inDim = inDim + 1;
	newSelectionLayer.outDim = outDim;
	newSelectionLayer.A = newSelectionWeights;
	newSelectionLayer.b = newSelectionBias;

	memcpy(newSelectionLayer.b, selectionLayer->b,outDim*sizeof(float));

	for(int i = 0; i<outDim;i++){
		memcpy(newSelectionLayer.A + i*(inDim + 1), selectionLayer->A + i*inDim,inDim*sizeof(float));
		newSelectionLayer.A[inDim] = 0.001;
	}

	float norm = cblas_snrm2(inDim,selectionLayer->A+selectionIndex*inDim,1);
	cblas_sscal(inDim, 1/(5*norm), newSelectionLayer.A + selectionIndex*(inDim+1), 1);
	newSelectionLayer.b[selectionIndex] *= (1.0/(5*norm));
	#ifdef DEBUG
		printf("==\n");
		printf("Old selection layer:\n");
		printLayer(selectionLayer);
		printf("Rescaled selectionLayer:\n");
		printLayer(&newSelectionLayer);
		printf("==\n");
	#endif
	return newSelectionLayer;
}

void oneGradientDecentForSelector(nnLayer * newSelector, float *reg, float label, float stepsize)
{
	uint dim = newSelector->inDim;
	float output;
	evalLayer(newSelector, reg, &output);
	float error = label - output;
	cblas_saxpy(dim, stepsize * error, reg, 1, newSelector->A,1);
	newSelector->b[0] += error * stepsize;
}

void trainNewSelector(nnLayer *selectionLayer, mapTreeNode ** locArr, int locArrLen, maxErrorCorner *maxGroup, float *newSelectionWeights, float *newSelectionBias)
{
	#ifdef DEBUG
		printf("==\n");
		printf("Training new selector\n");
		printf("==\n");
	#endif
	int selectionIndex = maxGroup->selectionIndex;

	vector *regionVec = getRegSigs(locArr, locArrLen);

	nnLayer newSelectionLayer = expandAndRescaleLayer(selectionLayer,selectionIndex,newSelectionWeights,newSelectionBias);

	nnLayer selector;
	selector.inDim = selectionLayer->inDim;
	uint dim = newSelectionLayer.inDim;
	selector.outDim = 1;
	selector.A = newSelectionLayer.A + selectionIndex*dim;
	selector.b = newSelectionLayer.b + selectionIndex;

	#ifdef DEBUG
		printf("==\n");
		printf("Creating the global data.\n");
		printf("==\n");
	#endif

	int dataLength = 2*vector_total(regionVec);
	float *unpackedSigs = malloc(2*dataLength*(dim)*sizeof(float));
	int *labels = malloc(2*dataLength*sizeof(int));
	createData(maxGroup, selectionLayer, regionVec,unpackedSigs,labels);

	#ifdef DEBUG
		printf("==\n");
		printf("Creating the local data that will recieve extra training\n");
		printf("==\n");
	#endif

	vector * importantRegVec = getRegSigs(maxGroup->locations, maxGroup->locCount);
	int importantDataLength = 2*vector_total(importantRegVec);
	float *importantUnpackedSigs = malloc(2*importantDataLength*(dim)*sizeof(float));
	int *importantLabels = malloc(2*importantDataLength*sizeof(int));
	createData(maxGroup, selectionLayer, importantRegVec,importantUnpackedSigs,importantLabels);
	

	#ifdef DEBUG
		printf("==\n");
		printf("Training new selector\n");
		printf("==\n");
	#endif
	float stepsize = INITSTEPSIZE;
	for(int i = 0; i<NUMEPOCHS; i++){
		for(int j = 0; j<dataLength; j++){
			oneGradientDecentForSelector(&selector,unpackedSigs + j*dim, labels[j],stepsize);
		}
		for(int k = 0; k<importantDataLength; k++){
			oneGradientDecentForSelector(&selector,importantUnpackedSigs + k*dim, importantLabels[k],2*stepsize);
		}
		stepsize *=0.95;
		#ifdef DEBUG
			if(i % 100 ==0){
				printf("On Step %d\n",i);
				printLayer(&newSelectionLayer);
			}
		#endif
	}
	free(unpackedSigs);
	free(labels);
	free(importantUnpackedSigs);
	free(importantLabels);
	vector_free(importantRegVec);
	free(importantRegVec);
	return newSelectionLayer;
}