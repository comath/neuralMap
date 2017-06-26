CC = gcc -Wall -Wextra -DMKL

OPT = -O1
DEBUG = -DDEBUG
GDB = -g
PROFILE = -lprofiler

MKLROOT = /opt/intel/compilers_and_libraries/linux/mkl
MKLINC = -I$(MKLROOT)/include/
MKLSTATICLIB = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group
MKLONEDYNAMICLIB =  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_rt 
PYINCDIR = $(shell python -c \
			"from distutils import sysconfig; print(sysconfig.get_python_inc())")

PYLIBS = $(shell python -c \
			"from distutils import sysconfig; print(sysconfig.get_config_var('LIBS'))")

CCFLAGS = $(DEBUG) $(GDB) $(FINAL) $(OPT) $(EXTRA_OPT) 
BIN=./build/
WRAP = ./source/pythonInterface/
UTILS = ./source/cutils/
TEST = ./source/test/

LIB_FLAGS = -ldl -lpthread -lm


all: parallelTree_test key_test

#The utilities used for most of this
parallelTree.o: $(UTILS)parallelTree.c $(UTILS)key.c
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@
mapperTree.o: $(UTILS)mapperTree.c $(UTILS)key.c
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@
key.o: $(UTILS)key.c 
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@
location.o: $(UTILS)location.c 
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@
vector.o: $(UTILS)vector.c 
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@
nnLayerUtils.o: $(UTILS)nnLayerUtils.c 
	$(CC) $(CCFLAGS) $(MKLINC) -c $< -o $(BIN)$@

# Main Utilities
ipCalculator.o: $(UTILS)ipCalculator.c  $(UTILS)nnLayerUtils.c $(UTILS)parallelTree.c
	$(CC) $(CCFLAGS) $(MKLINC) -c $< -o $(BIN)$@
ipTrace.o: $(UTILS)ipTrace.c  $(UTILS)nnLayerUtils.c $(UTILS)key.c
	$(CC) $(CCFLAGS) $(MKLINC) -c $< -o $(BIN)$@
mapper.o: $(UTILS)mapper.c $(UTILS)ipTrace.c  $(UTILS)nnLayerUtils.c $(UTILS)mapperTree.c $(UTILS)location.c
	$(CC) $(CCFLAGS) $(MKLINC) -c $< -o $(BIN)$@
adaptiveTools.o: $(UTILS)adaptiveTools.c $(UTILS)nnLayerUtils.c $(UTILS)mapperTree.c $(UTILS)location.c $(UTILS)key.c $(UTILS)vector.c 
	$(CC) $(CCFLAGS) $(MKLINC) -c $< -o $(BIN)$@
selectionTrainer.o: $(UTILS)selectionTrainer.c $(UTILS)nnLayerUtils.c $(UTILS)adaptiveTools.c $(UTILS)location.c $(UTILS)key.c $(UTILS)vector.c 
	$(CC) $(CCFLAGS) $(MKLINC) -c $< -o $(BIN)$@

#Python Interface
mapperWrap:
	python $(WRAP)buildMapperWrap.py build_ext -i
#Python Interface
ipCalculatorWrap:
	python $(WRAP)buildipCalculatorWrap.py build_ext -i
ipTraceWrap:
	python $(WRAP)buildIPTraceWrap.py build_ext -i


#Testing
ipCalculator_test.o: $(TEST)ipCalculator_test.c $(UTILS)ipCalculator.c $(UTILS)parallelTree.c
	$(CC) $(CXXFLAGS) $(MKLINC) -c $< -o $(BIN)$@

ipTrace_test.o: $(TEST)ipTrace_test.c $(UTILS)ipTrace.c $(UTILS)key.c
	$(CC) $(CXXFLAGS) $(MKLINC) -c $< -o $(BIN)$@

mapper_test.o: $(TEST)mapper_test.c $(UTILS)mapper.c $(UTILS)ipTrace.c $(UTILS)mapperTree.c $(UTILS)location.c $(UTILS)adaptiveTools.c
	$(CC) $(CXXFLAGS) $(MKLINC) -c $< -o $(BIN)$@

ipCalculator_test: ipCalculator_test.o ipCalculator.o parallelTree.o key.o nnLayerUtils.o
	$(CC) $(CCFLAGS) -DDEBUG $(BIN)$< $(BIN)ipCalculator.o $(BIN)nnLayerUtils.o $(BIN)parallelTree.o $(BIN)key.o -o $@ $(MKLONEDYNAMICLIB) $(LIB_FLAGS)

ipTrace_test: ipTrace_test.o ipTrace.o key.o nnLayerUtils.o
	$(CC) $(CCFLAGS) -DDEBUG $(BIN)$< $(BIN)ipTrace.o $(BIN)nnLayerUtils.o $(BIN)key.o -o $@ $(MKLONEDYNAMICLIB) $(LIB_FLAGS) 

mapper_test: mapper_test.o mapper.o ipTrace.o mapperTree.o key.o nnLayerUtils.o location.o adaptiveTools.o vector.o selectionTrainer.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)mapper.o $(BIN)ipTrace.o $(BIN)nnLayerUtils.o $(BIN)mapperTree.o $(BIN)location.o $(BIN)selectionTrainer.o $(BIN)vector.o $(BIN)adaptiveTools.o $(BIN)key.o -o $@ $(MKLONEDYNAMICLIB) $(LIB_FLAGS) 

parallelTree_test.o: $(TEST)parallelTree_test.c $(UTILS)parallelTree.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

parallelTree_test: parallelTree_test.o parallelTree.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)parallelTree.o $(BIN)key.o -o $@ $(LIB_FLAGS) 

key_test.o: $(TEST)key_test.c $(UTILS)key.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

key_test: key_test.o key.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)key.o -o $@ $(LIB_FLAGS) 

#2d Visualization

all: imagetest ann


.PHONY: clean

clean:
	rm -f key_test parallelTree_test bin/*.o

