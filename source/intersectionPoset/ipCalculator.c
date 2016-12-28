#include "ipCalculator.h"

void * (*dataCreator)(void * input);
void (*dataModifier)(void * input, void * data);
void (*dataDestroy)(void * data);

ipCache * allocateCache(nnLayer *hpLayer)
{
	ipCache *cache = malloc(sizeof(ipCache));
	cache->bases = createTree()
}