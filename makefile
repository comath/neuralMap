CC = gcc -Wall -Wextra

OPT = -O2
DEBUG = -DDEBUG
GDB = -g


CCFLAGS = $(DEBUG) $(GDB) $(FINAL) $(OPT) $(EXTRA_OPT)
BIN=./build/
LIB=./lib/
UTILS = ./source/utils/
TEST = ./source/test/

LIB_FLAGS = -ldl -lpthread -lm


all: parallelTree_test key_test

#The tree structure used for most of this
parallelTree.o: $(UTILS)parallelTree.c 
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@


#The linear algebra library with MKL
MKLROOT = /opt/intel/compilers_and_libraries/linux/mkl
MKLINC = $(MKLROOT)/include/
linAlgMKL.o: $(UTILS)linAlgMKL.c
	$(CC) $(CCFLAGS) -fPIC -I$(MKLINC) -c $< -o $(BIN)$@

linAlgWrap.o: linAlgMKL.o
	ld -r -o $(BIN)linAlgWrap.o $(BIN)linAlgMKL.o $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a 

liblinAlg.a: linAlgWrap.o 
	ar rcs $(LIB)liblinAlg.a $(BIN)linAlgWrap.o
	

#Testing
parallelTree_test.o: $(TEST)parallelTree_test.c $(UTILS)parallelTree.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

parallelTree_test: parallelTree_test.o parallelTree.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)parallelTree.o -o $@ $(LIB_FLAGS) 

key_test.o: $(TEST)key_test.c $(UTILS)parallelTree.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

key_test: key_test.o parallelTree.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)parallelTree.o -o $@ $(LIB_FLAGS) 

linAlg_test.o: $(TEST)linAlg_test.c
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@

linAlg_test: linAlg_test.o liblinAlg.a
	$(CC) $(CCFLAGS) $(BIN)$< -o $@  -L$(LIB) -llinAlg $(LIB_FLAGS)

.PHONY: clean

clean:
	rm -f key_test parallelTree_test linAlg_test bin/*.o

