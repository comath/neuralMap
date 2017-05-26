// From http://eddmann.com/posts/implementing-a-dynamic-vector-array-in-c/

#ifndef LOCATION_H
#define LOCATION_H

#define VECTOR_INIT_CAPACITY 4

typedef struct pointInfo {
	int * traceRaw;
	float * traceDists;
	long int pointIndex;
}

typedef struct location {
    float *traceDists;
    int *traceRaw;
    int *pointIndex;
    int m;
    int capacity;
    int total;
} location;

void location_init(location *);
int location_total(location *);
static void location_resize(location *, int);
void location_add(location *, pointInfo *);
void location_set(location *, int, pointInfo *);
pointInfo *location_get(location *, int);
void location_delete(location *, int);
void location_free(location *);


#endif