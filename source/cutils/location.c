// from http://eddmann.com/posts/implementing-a-dynamic-vector-array-in-c/

#include <stdio.h>
#include <stdlib.h>

#include "location.h"

void location_init(location *v,int m)
{
    typedef struct location {
    float *traceDists;
    int *traceRaws;
    int *pointIndexes;
    v->capacity = location_INIT_CAPACITY;
    v->total = 0;
    v->m = 0;
    v->pointIndexes = malloc(sizeof(int *) * v->capacity);
    v->traceRaws = malloc(sizeof(int *) * v->capacity * v->m);
    v->traceDists = malloc(sizeof(float *) * v->capacity * v->m);
}

int location_total(location *v)
{
    return v->total;
}

static void location_resize(location *v, int capacity)
{
    #ifdef DEBUG_ON
    printf("location_resize: %d to %d\n", v->capacity, capacity);
    #endif

    int     *pointIndexes = realloc(v->pointIndexes, sizeof(void *) * capacity);
    int     *traceRaws    = realloc(v->traceRaws, sizeof(void *) * capacity * v->m);
    float   *traceDists   = realloc(v->traceDists, sizeof(void *) * capacity * v->m);
    if (items && traceDists && traceRaws) {
        v->pointIndexes = pointIndexes;
        v->traceRaws = traceRaws;
        v->traceDists = traceDists;
        v->capacity = capacity;
    }
}

void location_add(location *v, pointInfo *item)
{
    if (v->capacity == v->total){
        location_resize(v, v->capacity * 2);
    }
    int i = v->total++;
    v->pointIndexes[i] = item.pointIndex;
    memcpy(v->traceRaws[i*v->m], item.traceRaws, v->m*sizeof(int));
    memcpy(v->traceDists[i*v->m], item.traceDists, v->m*sizeof(float));
}

void location_set(location *v, int index, pointInfo *item)
{
    if (index >= 0 && index < v->total){
        v->pointIndexes[index] = item.pointIndex;
        memcpy(v->traceRaws[index*v->m], item.traceRaws, v->m*sizeof(int));
        memcpy(v->traceDists[index*v->m], item.traceDists, v->m*sizeof(float));
    }
}

pointInfo location_get(location *v, int index)
{
    if (index >= 0 && index < v->total){
        pointInfo pi;
        pi.pointIndex = v->pointIndexes[index];
        pi.traceDists = v->traceDists +index*v->m;
        pi.traceRaw = v->traceRaws +index*v->m;
        return pi;
    }
    return NULL;
}

void location_free(location *v)
{
    free(v->pointIndexes);
    free(v->traceDists);
    free(v->traceRaws);
}