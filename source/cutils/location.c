// from http://eddmann.com/posts/implementing-a-dynamic-vector-array-in-c/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "location.h"

pointInfo * allocPointInfo(int m)
{
    pointInfo * pi = malloc(sizeof(pointInfo));
    if(pi){
        pi->traceRaw = malloc(m * sizeof(int));
        pi->traceDists = malloc(m * sizeof(float));
        if(pi->traceDists && pi->traceRaw){
            return pi;
        } 
    }
    return NULL;
}

void freePointInfo(pointInfo *pi)
{
     if(pi){
        if(pi->traceDists){
            free(pi->traceDists);
        }
        if(pi->traceRaw){
            free(pi->traceRaw);
        }
        free(pi);
    }
}

void location_init(location *v,int m)
{
    v->capacity = LOCATION_INIT_CAPACITY;
    v->total = 0;
    v->m = m;
    v->pointIndexes = malloc(sizeof(int *) * v->capacity);
    v->traceRaws = malloc(sizeof(int *) * v->capacity * v->m);
    v->traceDists = malloc(sizeof(float *) * v->capacity * v->m);
}

int location_total(location *v)
{
    return v->total;
}

void location_resize(location *v, int capacity)
{
    #ifdef DEBUG_ON
    printf("location_resize: %d to %d\n", v->capacity, capacity);
    #endif

    int     *pointIndexes = realloc(v->pointIndexes, sizeof(int *) * capacity);
    int     *traceRaws    = realloc(v->traceRaws, sizeof(int *) * capacity * v->m);
    float   *traceDists   = realloc(v->traceDists, sizeof(float *) * capacity * v->m);
    if (pointIndexes && traceDists && traceRaws) {
        v->pointIndexes = pointIndexes;
        v->traceRaws = traceRaws;
        v->traceDists = traceDists;
        v->capacity = capacity;
    }
}

void location_add(location * v, pointInfo *item)
{
    if (v->capacity == v->total){
        location_resize(v, v->capacity * 2);
    }
    int i = v->total++;
    v->pointIndexes[i] = item->index;
    memcpy(v->traceRaws + i*v->m, item->traceRaw, v->m*sizeof(int));
    memcpy(v->traceDists + i*v->m, item->traceDists, v->m*sizeof(float));
}

void location_set(location *v, int index, pointInfo *item)
{
    if (index >= 0 && index < v->total){
        v->pointIndexes[index] = item->index;
        memcpy(v->traceRaws + index*v->m, item->traceRaw, v->m*sizeof(int));
        memcpy(v->traceDists + index*v->m, item->traceDists, v->m*sizeof(float));
    }
}

pointInfo location_get(location *v, int index)
{
    pointInfo pi;
    if (index >= 0 && index < v->total){
        pi.index = v->pointIndexes[index];
        pi.traceDists = v->traceDists +index*v->m;
        pi.traceRaw = v->traceRaws +index*v->m;
    } else {
        pi.index = -1;
    }
    return pi;
}

void location_free(location *v)
{
    if(v->pointIndexes)
        free(v->pointIndexes);
    if(v->traceDists)
        free(v->traceDists);
    if(v->traceRaws)
        free(v->traceRaws);
}

