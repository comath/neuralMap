CC = gcc -Wall -Wextra -DMKL

OPT = -O1
DEBUG = -DDEBUG
GDB = -pg
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
TEST = ./examples/test/

LIB_FLAGS = -ldl -lpthread -lm


all: parallelTree_test key_test

#The utilities used for most of this
parallelTree.o: $(UTILS)parallelTree.c $(UTILS)key.c
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@
key.o: $(UTILS)key.c 
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@
nnLayerUtils.o: $(UTILS)nnLayerUtils.c 
	$(CC) $(CCFLAGS) $(MKLINC) -c $< -o $(BIN)$@

# Main Utilities
ipCalculator.o: $(UTILS)ipCalculator.c  $(UTILS)nnLayerUtils.c $(UTILS)parallelTree.c
	$(CC) $(CCFLAGS) $(MKLINC) -c $< -o $(BIN)$@
mapper.o: $(UTILS)mapper.c $(UTILS)ipCalculator.c  $(UTILS)nnLayerUtils.c $(UTILS)parallelTree.c
	$(CC) $(CCFLAGS) $(MKLINC) -c $< -o $(BIN)$@

#Python Interface
mapperWrap:
	python $(WRAP)buildMapperWrap.py build_ext -i
#Python Interface
ipCalculatorWrap:
	python $(WRAP)buildipCalculatorWrap.py build_ext -i

#Testing
ipCalculator_test.o: $(TEST)ipCalculator_test.c $(UTILS)ipCalculator.c $(UTILS)parallelTree.c
	$(CC) $(CXXFLAGS) $(MKLINC) -c $< -o $(BIN)$@

mapper_test.o: $(TEST)mapper_test.c $(UTILS)mapper.c $(UTILS)ipCalculator.c $(UTILS)parallelTree.c
	$(CC) $(CXXFLAGS) $(MKLINC) -c $< -o $(BIN)$@

ipCalculator_test: ipCalculator_test.o ipCalculator.o parallelTree.o key.o nnLayerUtils.o
	$(CC) $(CCFLAGS)  $(BIN)$< $(BIN)ipCalculator.o $(BIN)nnLayerUtils.o $(BIN)parallelTree.o $(BIN)key.o -o $@ $(MKLONEDYNAMICLIB) $(LIB_FLAGS) 

mapper_test: mapper_test.o mapper.o ipCalculator.o parallelTree.o key.o nnLayerUtils.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)mapper.o $(BIN)ipCalculator.o $(BIN)nnLayerUtils.o $(BIN)parallelTree.o $(BIN)key.o -o $@ $(MKLONEDYNAMICLIB) $(LIB_FLAGS) 

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

selectiontrainer.o: $(SOURCE)$(NEURAL)selectiontrainer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@

selectiontrainer: selectiontrainer.o
	$(CXX) $(CXXFLAGS) $(BIN)$<  -o $@ $(LIB_FLAGS) -lpthread -lm


pgmreader.o: $(SOURCE)$(IMAGE)pgmreader.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@ 

annpgm.o: $(SOURCE)$(IMAGE)annpgm.cpp $(SOURCE)$(IMAGE)pgmreader.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@ 

ann.o: $(SOURCE)$(NEURAL)ann.cpp $(SOURCE)$(IMAGE)pgmreader.cpp $(SOURCE)$(IMAGE)annpgm.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@  

nnanalyzer.o: $(SOURCE)$(NEURAL)nnanalyzer.cpp $(SOURCE)$(NEURAL)ann.cpp $(SOURCE)$(NEURAL)selectiontrainer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@

nnmap.o: $(SOURCE)$(NEURAL)nnmap.cpp $(SOURCE)$(NEURAL)ann.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@

imagetest.o: $(SOURCE)imagetest.cpp $(SOURCE)$(IMAGE)pgmreader.cpp $(SOURCE)$(NEURAL)ann.cpp $(SOURCE)$(NEURAL)nnanalyzer.cpp $(SOURCE)$(NEURAL)nnmap.cpp $(SOURCE)$(IMAGE)annpgm.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(BIN)$@

imagetest: imagetest.o pgmreader.o ann.o nnanalyzer.o nnmap.o annpgm.o selectiontrainer.o
	$(CXX) $(CXXFLAGS)  $(BIN)$< $(BIN)pgmreader.o $(BIN)ann.o $(BIN)nnanalyzer.o $(BIN)nnmap.o $(BIN)annpgm.o $(BIN)selectiontrainer.o -o $@ $(LIB_FLAGS) -lpthread -lm


.PHONY: clean

clean:
	rm -f key_test parallelTree_test bin/*.o

