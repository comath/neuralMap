// From http://eddmann.com/posts/implementing-a-dynamic-vector-array-in-c/
#ifndef VECTOR_H
#define VECTOR_H

#define VECTOR_INIT_CAPACITY 4

#define VECTOR_INIT(vec) vector vec; vector_init(&vec)
#define VECTOR_ADD(vec, item) vector_add(&vec, (void *) item)
#define VECTOR_SET(vec, id, item) vector_set(&vec, id, (void *) item)
#define VECTOR_GET(vec, type, id) (type) vector_get(&vec, id)
#define VECTOR_DELETE(vec, id) vector_delete(&vec, id)
#define VECTOR_TOTAL(vec) vector_total(&vec)
#define VECTOR_FREE(vec) vector_free(&vec)


/*
a std:vector implementation. 

However it does not resize down with sufficent deletions as we use it as a buffer for in adaptiveTools
*/
typedef struct vector {
    void **items;
    int capacity;
    int total;
} vector;

void vector_init(vector *);
int vector_total(vector *);
void vector_add(vector *, void *);
void vector_set(vector *, int, void *);
void *vector_get(vector *, int);
void vector_delete(vector *, int);
void vector_free(vector *);

// My additions
// Creates a copy of the source and puts it in the target
void vector_copy(vector *target, vector *source);
/*
Resets the vector so that we can reuse the buffer. 
Called in adaptive tools as we repeatedly fill a vector with locations.
*/
void vector_reset(vector *v);  // Doesn't deallocate, just puts the total to 0
// Prints the points in a vector for debugging purposes.
void vector_print_pointers(vector *vec);


#endif