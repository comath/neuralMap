#import "nnLayerUtils.h"

nnLayer * createLayer(float * A, float *b, uint inDim, uint outDim)
{
	nnLayer * layer = malloc(sizeof(nnLayer));
	layer->A = malloc((inDim*outDim + outDim)*sizeof(float));
	layer->b = layer->A + inDim*outDim;
	cblas_scopy (inDim*outDim, A, 1, layer->A, 1);
	cblas_scopy (outDim, A, 1, layer->b, 1);
	layer->inDim = inDim;
	layer->outDim = outDim;
}

void freeLayer(nnLayer *layer)
{
	if(layer){
		if(layer->A){
			free(layer->A);
		}
		if(layer->b){
			free(layer->b);
		}
		free(layer);
	}
}