typedef struct ipMemory {
	float *s;  // Diagonal entries
	float *u; 
	float *vt; 
	float *c; 
	float *distances; 
	float *distancesForSorting; 
	float *localCopy;
	float *superb; 
	float *px; 
	float *subA;
	float * subB;
	MKL_INT info;
	MKL_INT minMN;
	uint numHps;
	uint maxNumHps;
	uint *hpDistIndexList;
	
} ipMemory;

ipMemory * allocateIPMemory(inDim,outDim)
{
	int m = inDim;
	int n = outDim;
	int numRelevantHP = 0;
	int minMN = 0
	if(m < n){
		/* 
		If the spacial dim is less than the number of hyperplanes then we want to 
		take the distance to the closest instersection (which will be of full rank).
		*/
		numRelevantHP = m + 1;
		minMN = m;
	} else {
		numRelevantHP = n;
		minMN = n;
	}
	ipMemory *mb = malloc(sizeof(ipMemory));
	// Each pointer is followed by the number of floats it has all to itself.
	float * mainBuffer = malloc((2*m+2*n+n*n+m*m+2*n*m+minMN+(m+1)*numRelevantHP)*sizeof(float));
	mb->s = mainBuffer+0; //m
	mb->px = mainBuffer+ m; //m
	mb->distances = mainBuffer+ m; // n
	mb->distancesForSorting = n; //n
	mb->u = mainBuffer+ n; //n^2
	mb->vt = mainBuffer+ n*n; //m^2
	mb->c = mainBuffer+ m*m; //m*n
	mb->localCopy =  mainBuffer+ m*n; //m*n
	mb->superb = mainBuffer+ m*n; // minMN
	mb->subA = mainBuffer+ minMN; // m*numRelevantHP
	mb->subB = mainBuffer+ m*numRelevantHP; // numRelevantHP
	mb->hpDistIndexList = malloc(numRelevantHP*sizeof(uint));
	return mb;
}

void freeIPMemory(ipMemory *mb)
{
	free(mb->s);
	free(mb->hpDistIndexList);
}