CC = gcc -Wall -Wextra

OPT = -O2
DEBUG = -DDEBUG
GDB = -g

MKLROOT = /opt/intel/compilers_and_libraries/linux/mkl
MKLINC = $(MKLROOT)/include/
MKLLIB = -Wl --start-group $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a --end-group

CCFLAGS = $(DEBUG) $(GDB) $(FINAL) $(OPT) $(EXTRA_OPT)
BIN=./build/
LIB=./lib/
UTILS = ./source/utils/
TEST = ./source/test/

LIB_FLAGS = -ldl -lpthread -lm


all: parallelTree_test key_test

#The utilities used for most of this
parallelTree.o: $(UTILS)parallelTree.c 
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@
key.o: $(UTILS)key.c 
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@
nnLayerUtils.o: $(UTILS)nnLayerUtils.c 
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@


#Testing
parallelTree_test.o: $(TEST)parallelTree_test.c $(UTILS)parallelTree.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

parallelTree_test: parallelTree_test.o parallelTree.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)parallelTree.o -o $@ $(LIB_FLAGS) 

key_test.o: $(TEST)key_test.c $(UTILS)key.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

key_test: key_test.o key.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)key.o -o $@ $(LIB_FLAGS) 


.PHONY: clean

clean:
	rm -f key_test parallelTree_test bin/*.o

