#include "ipCalculator.h"

struct ipCacheInput {
	//This should be a constant
	nnLayer * layer0;
	//This shouldn't be
	uint *key;
} ipCacheInput;

struct ipCacheData {
	float *solution;
	float *kernelBasis;
	uint dimKernel;
} ipCacheData;

struct ipCacheData * solve(float *A, MKL_INT m, MKL_INT n, float *b)
{
	CBLAS_LAYOUT    layout = CblasRowMajor;
	CBLAS_TRANSPOSE noTrans = CblasNoTrans;
	CBLAS_TRANSPOSE trans = CblasTrans;
	MKL_INT lda = n;
	MKL_INT ldu = m;
	MKL_INT ldvt = n;
	MKL_INT info;
	float superb[((n)>(m)?(m):(n))-1]; // Min of n and m
	/* Local arrays */
	float *s = calloc(n,sizeof(float));
	float *u = calloc(ldu*m , sizeof(float));
	float *vt = calloc(ldvt*n, sizeof(float));


	/* Executable statements */
	printf( "LAPACKE_sgesvd (row-major, high-level) Example Program Results\n" );
	info = LAPACKE_sgesvd( LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda,
		    s, u, ldu, vt, ldvt, superb );
	// Incase of memory leaks:
	//mkl_thread_free_buffers();
	if( info > 0 ) {
		printf( "The algorithm computing SVD failed to converge.\n" );
		exit( 1 );
	}


	// Multiplying Sigma+t with u
	int i = 0;
	for(i=0;i<m;i++){
		cblas_sscal(m,(1/s[i]),u+i*m,1);
	}
	float * kernelBasis;
	if(n == m){
		output->kernelBasis = NULL;
	} else {
		output->kernelBasis = calloc((n-m)*m,sizeof(float));
	}
	 
	float * solution = calloc(n, sizeof(float));
	float *c = calloc(n*m , sizeof(float));
	// Multiplying u sigma+t with vt
	cblas_sgemm (layout, noTrans, noTrans, m, n, m, 1, u, ldu, vt, ldvt, 0,c, n);
	// Multiplying v sigma+ u with b for the solution
	cblas_sgemv (layout, trans, m, n,1, c, n, b, 1, 0, solution, 1);
	// Saving the kernel basis from vt
	if(m<n){
		cblas_scopy ((n-m)*n, vt+m*n, 1, kernelBasis, 1);
	}
	free(s);
	free(u);
	free(vt);
	free(c);
	// Incase of memory leaks:
	//mkl_thread_free_buffers();

	struct ipCacheData *output = malloc(sizeof(struct ipCacheData));
	output->solution = solution;
	output->kernelBasis = kernelBasis;
	output->dimKernel = m;
	return output;
}




void * dataCreator(void * input)
{
	struct ipCacheInput *myInput;
	myInput = (struct ipCacheInput *) input;

	uint inDim = myInput->layer0->inDim;
	uint outDim = myInput->layer0->outDim;

	float *subA = calloc(inDim*outDim, sizeof(float));
	float *subB = calloc(outDim,sizeof(float));
	uint i = 0;
	uint numHps = 0;
	for(i=0;i<inDim;i++){
		if(checkIndex(myInput->key,i)){
			cblas_scopy (inDim, myInput->layer0->A +i*inDim, 1, subA + numHps*inDim, 1);
			subB[numHps] = myInput->layer0->b[i];
			numHps++;
		}
	}
	//If there's less included hyperplanes there will be a kernel
	if(numHps <= inDim){
		MKL_INT m = numHps;
		MKL_INT n = inDim;
		return solve(subA, m, n, subB, kernelBasis, solution);
	}


}
void dataDestroy(void * data)
{

}

ipCache * allocateCache(nnLayer *hpLayer)
{
	ipCache *cache = malloc(sizeof(ipCache));
	cache->bases = createTree(8, )
}

float computeDist(float * p, uint dimP, uint * indexes, uint numIndexes)
{
	#ifdef DEBUG
		cout << "Finding the distance between the intersection ";
		printset(indexes);
		cout << "and the vector: " << endl << p;
	#endif
	IntersectionBasis curIB = this->getIntersectionBasis(indexes);
	
	if(curIB.success){
		vec px = p+curIB.aSolution;
		vec pparallel = zeros<vec>(p.n_rows);
		for (unsigned i = 0; i < (curIB.basis).n_cols; ++i){
			vec Bi = (curIB.aSolution).col(i);
			pparallel += dot(Bi,px)*Bi;
		}
		return norm(px-pparallel);
	} else {
		return -1;
	}	
}