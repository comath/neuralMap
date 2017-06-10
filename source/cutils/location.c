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
    v->capacity_error = LOCATION_INIT_CAPACITY;
    v->total = 0;
    v->m = m;
    v->pointIndexes = malloc(sizeof(int *) * v->capacity);
    v->traceRaws = malloc(sizeof(int *) * v->capacity * v->m);
    v->traceDists = malloc(sizeof(float *) * v->capacity * v->m);
    v->total_error = 0;
    v->pointIndexes_error = malloc(sizeof(int *) * v->capacity_error);
    v->traceRaws_error = malloc(sizeof(int *) * v->capacity_error * v->m);
    v->traceDists_error = malloc(sizeof(float *) * v->capacity_error * v->m);
}

int location_total(location *v, int errorClass)
{
    if(errorClass == 0){
        return v->total;
    } else {
        return v->total_error;
    }
}

int location_complete_total(location *v)
{
    return v->total + v->total_error;
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

void location_resize_error(location *v, int capacity_error)
{
    #ifdef DEBUG_ON
    printf("location_resize: %d to %d\n", v->capacity_error, capacity_error);
    #endif

    int     *pointIndexes_error = realloc(v->pointIndexes_error, sizeof(int *) * capacity_error);
    int     *traceRaws_error    = realloc(v->traceRaws_error, sizeof(int *) * capacity_error * v->m);
    float   *traceDists_error  = realloc(v->traceDists_error, sizeof(float *) * capacity_error * v->m);
    if (pointIndexes_error && traceDists_error && traceRaws_error) {
        v->pointIndexes_error = pointIndexes_error;
        v->traceRaws_error = traceRaws_error;
        v->traceDists_error = traceDists_error;
        v->capacity_error = capacity_error;
    }
}

void location_add(location * v, pointInfo *item)
{
    if(item->errorClass == 0){
        if (v->capacity == v->total){
            location_resize(v, v->capacity * 2);
        }
        int i = v->total++;
        v->pointIndexes[i] = item->index;
        memcpy(v->traceRaws + i*v->m, item->traceRaw, v->m*sizeof(int));
        memcpy(v->traceDists + i*v->m, item->traceDists, v->m*sizeof(float));
    } else {
        if (v->capacity_error == v->total_error){
            location_resize_error(v, v->capacity_error * 2);
        }
        int i = v->total_error++;
        v->pointIndexes_error[i] = item->index;
        memcpy(v->traceRaws_error + i*v->m, item->traceRaw, v->m*sizeof(int));
        memcpy(v->traceDists_error + i*v->m, item->traceDists, v->m*sizeof(float));
    }
}

void location_set(location *v, int index, pointInfo *item)
{
    if(item->errorClass == 0){
        if (index >= 0 && index < v->total){
            v->pointIndexes[index] = item->index;
            memcpy(v->traceRaws + index*v->m, item->traceRaw, v->m*sizeof(int));
            memcpy(v->traceDists + index*v->m, item->traceDists, v->m*sizeof(float));
        }
    } else {
        if (index >= 0 && index < v->total_error){
            v->pointIndexes[index] = item->index;
            memcpy(v->traceRaws_error + index*v->m, item->traceRaw, v->m*sizeof(int));
            memcpy(v->traceDists_error + index*v->m, item->traceDists, v->m*sizeof(float));
        }
    }
}

pointInfo location_get(location *v, int index, int errorClass)
{
    pointInfo pi;
    if(errorClass == 0){
        if (index >= 0 && index < v->total){
            pi.index = v->pointIndexes[index];
            pi.traceDists = v->traceDists +index*v->m;
            pi.traceRaw = v->traceRaws +index*v->m;
        } else {
            pi.index = -1;
        }
    } else {
        if (index >= 0 && index < v->total_error){
            pi.index = v->pointIndexes_error[index];
            pi.traceDists = v->traceDists_error +index*v->m;
            pi.traceRaw = v->traceRaws_error +index*v->m;
        } else {
            pi.index = -1;
        }
    }
    return pi;
}

void location_get_indexes(location *v, int *indexHolder, int errorClass)
{
    if(errorClass == 0){
        memcpy(indexHolder, v->pointIndexes, v->total*sizeof(int));
    } else {
        memcpy(indexHolder, v->pointIndexes_error, v->total_error*sizeof(int));
    }
}

void location_free(location *v)
{
    if(v->pointIndexes)
        free(v->pointIndexes);
    if(v->traceDists)
        free(v->traceDists);
    if(v->traceRaws)
        free(v->traceRaws);
    if(v->pointIndexes_error)
        free(v->pointIndexes_error);
    if(v->traceDists_error)
        free(v->traceDists_error);
    if(v->traceRaws_error)
        free(v->traceRaws_error);
}

