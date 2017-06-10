// From http://eddmann.com/posts/implementing-a-dynamic-vector-array-in-c/

#ifndef LOCATION_H
#define LOCATION_H

#define LOCATION_INIT_CAPACITY 4

typedef struct pointInfo {
	int * traceRaw;
	float * traceDists;
	long int index;
	int errorClass;
} pointInfo;

typedef struct location {
    float *traceDists;
    int *traceRaws;
    int *pointIndexes;
    int m;
    int capacity;
    int total;
    float *traceDists_error;
    int *traceRaws_error;
    int *pointIndexes_error;
    int capacity_error;
    int total_error;
} location;

pointInfo * allocPointInfo(int m);
void freePointInfo(pointInfo *pi);

void location_init(location * l,int m);
int location_total(location * l, int errorClass);
int location_complete_total(location * l);
void location_add(location * l, pointInfo * pi);
void location_set(location * l, int index, pointInfo * pi);
pointInfo location_get(location * l, int index, int errorClass);
void location_free(location * l);
void location_get_indexes(location * l, int* indexHolder,int errorClass);


#endif