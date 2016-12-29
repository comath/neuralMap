#include "ipCalculator.h"

void solve(float *A, MKL_INT m, MKL_INT n, float *b, float *kernelBasis, uint numKer, float *solution)
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
  /* Compute SVD */
  info = LAPACKE_sgesvd( LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda,
            s, u, ldu, vt, ldvt, superb );
  // Incase of memory leaks:
  //mkl_thread_free_buffers();
  /* Check for convergence */
  if( info > 0 ) {
    printf( "The algorithm computing SVD failed to converge.\n" );
    exit( 1 );
  }

  /* Print singular values */
  print_matrix( "Singular values", 1, m, s, 1 );
  /* Print left singular vectors */
  print_matrix( "Left singular vectors (stored columnwise)", m, m, u, ldu );
  /* Print right singular vectors */
  print_matrix( "Right singular vectors (stored rowwise)", n, n, vt, ldvt );

  int i = 0;
  for(i=0;i<m;i++){
    cblas_sscal(m,(1/s[i]),u+i*m,1);
  }

  print_matrix( "Left singular vectors (stored columnwise) after multplying by sigma+", m, m, u, ldu );
  float *c = calloc(n*m , sizeof(float));

  cblas_sgemm (layout, noTrans, noTrans, m, n, m, 1, u, ldu, vt, ldvt, 0,c, n);
  print_matrix( "Product of u s vt:", m, n, c, n );

  cblas_sgemv (layout, trans, m, n,1, c, n, b, 1, 0, solution, 1);
  print_matrix( "Product of v s+t ut b:", 1, n, solution, 1 );

  cblas_scopy ((n-m)*n, vt+m*n, 1, kernelBasis, 1);
  print_matrix( "kernelBasis:", n-m, n, kernelBasis, n-m );

  free(s);
  free(u);
  free(vt);
  free(c);
  // Incase of memory leaks:
  //mkl_thread_free_buffers();
}

void * (*dataCreator)(void * input);
void (*dataModifier)(void * input, void * data);
void (*dataDestroy)(void * data);

ipCache * allocateCache(nnLayer *hpLayer)
{
	ipCache *cache = malloc(sizeof(ipCache));
	cache->bases = createTree()
}