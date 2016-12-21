CC = gcc -Wall -Wextra

OPT = -O2
DEBUG = -DDEBUG
GDB = -g

MKLROOT = /opt/intel/compilers_and_libraries/linux/mkl
MKLFLAGS =  -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -ldl
PTHREADFLAGS = -lpthread -lm 

MKLINC = $(MKLROOT)/include/

CCFLAGS = $(DEBUG) $(GDB) $(FINAL) $(OPT) $(EXTRA_OPT)
BIN=./bin/
SOURCE=./source/
UTILS = utils/

all: parallelTree_test key_test

linearAlgebra.o: $(SOURCE)$(UTILS)linearAlgebra.c
	$(CC) $(CCFLAGS) -I$(MKLINC) -c $< -o $(BIN)$@ 

linearAlgebra_test.o: $(SOURCE)$(UTILS)linearAlgebra_test.c
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@

parallelTree.o: $(SOURCE)$(UTILS)parallelTree.c 
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@

parallelTree_test.o: $(SOURCE)$(UTILS)parallelTree_test.c $(SOURCE)$(UTILS)parallelTree.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

parallelTree_test: parallelTree_test.o parallelTree.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)parallelTree.o -o $@ $(LIB_FLAGS) $(PTHREADFLAGS)

key_test.o: $(SOURCE)$(UTILS)key_test.c $(SOURCE)$(UTILS)parallelTree.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

key_test: key_test.o parallelTree.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)parallelTree.o -o $@ $(LIB_FLAGS) $(PTHREADFLAGS)

linearAlgebra_test: linearAlgebra_test.o linearAlgebra.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)linearAlgebra.o -o $@ $(LIB_FLAGS) $(MKLFLAGS) $(PTHREADFLAGS)

.PHONY: clean

clean:
	rm -f key_test parallelTree_test bin/*.o

