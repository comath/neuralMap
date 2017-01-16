CC = gcc -Wall -Wextra -DMKL

OPT = -O2
DEBUG = -DDEBUG
GDB = -g

MKLROOT = /opt/intel/compilers_and_libraries/linux/mkl
MKLINC = -I$(MKLROOT)/include/
MKLLIB = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group

PYINCDIR = $(shell python -c \
			"from distutils import sysconfig; print(sysconfig.get_python_inc())")

PYLIBS = $(shell python -c \
			"from distutils import sysconfig; print(sysconfig.get_config_var('LIBS'))")

CCFLAGS = $(DEBUG) $(GDB) $(FINAL) $(OPT) $(EXTRA_OPT) 
BIN=./build/
WRAP = ./source/pythonInterface/
UTILS = ./source/utils/
TEST = ./source/test/

LIB_FLAGS = -ldl -lpthread -lm


all: parallelTree_test key_test

#The utilities used for most of this
parallelTree.o: $(UTILS)parallelTree.c $(UTILS)key.c
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@
key.o: $(UTILS)key.c 
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@
nnLayerUtils.o: $(UTILS)nnLayerUtils.c 
	$(CC) $(CCFLAGS) $(MKLINC) -c $< -o $(BIN)$@

# IP calculator
ipCalculator.o: $(UTILS)ipCalculator.c  $(UTILS)nnLayerUtils.c $(UTILS)parallelTree.c
	$(CC) $(CCFLAGS) $(MKLINC) -c $< -o $(BIN)$@

#Cython Wrappers
nnLayerUtilsWrap.c:
	cython -I $(UTILS) -v --line-directives $(WRAP)nnLayerUtilsWrap.pyx

#Testing
ipCalculator_test.o: $(TEST)ipCalculator_test.c $(UTILS)ipCalculator.c $(UTILS)parallelTree.c
	$(CC) $(CXXFLAGS) $(MKLINC) -c $< -o $(BIN)$@

ipCalculator_test: ipCalculator_test.o ipCalculator.o parallelTree.o key.o nnLayerUtils.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)ipCalculator.o $(BIN)nnLayerUtils.o $(BIN)parallelTree.o $(BIN)key.o -o $@ $(MKLLIB) $(LIB_FLAGS) 

parallelTree_test.o: $(TEST)parallelTree_test.c $(UTILS)parallelTree.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

parallelTree_test: parallelTree_test.o parallelTree.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)parallelTree.o $(BIN)key.o -o $@ $(LIB_FLAGS) 

key_test.o: $(TEST)key_test.c $(UTILS)key.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

key_test: key_test.o key.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)key.o -o $@ $(LIB_FLAGS) 


.PHONY: clean

clean:
	rm -f key_test parallelTree_test bin/*.o

