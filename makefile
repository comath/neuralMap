CC = gcc -Wall

DEBUG = -DDEBUG
GDB = -g

OPT = -O2

CCFLAGS = $(DEBUG) $(GDB) $(FINAL) $(OPT) $(EXTRA_OPT)
BIN=./bin/
SOURCE=./source/

all: imagetest ann

paralleltree.o: $(SOURCE)paralleltree.c 
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@

paralleltree_test.o: $(SOURCE)paralleltree_test.c $(SOURCE)paralleltree.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

paralleltree_test: paralleltree_test.o paralleltree.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)paralleltree.o -o $@ $(LIB_FLAGS) -lpthread -lm

key_test.o: $(SOURCE)key_test.c $(SOURCE)paralleltree.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

key_test: key_test.o paralleltree.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)paralleltree.o -o $@ $(LIB_FLAGS) -lpthread -lm


.PHONY: clean

clean:
	rm -f imagetest bin/*.o

