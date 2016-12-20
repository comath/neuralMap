CC = gcc -Wall -Wextra

DEBUG = -DDEBUG
GDB = -g

OPT = -O2

CCFLAGS = $(DEBUG) $(GDB) $(FINAL) $(OPT) $(EXTRA_OPT)
BIN=./bin/
SOURCE=./source/
TREE = binaryTree/

all: paralleltree_test key_test

paralleltree.o: $(SOURCE)$(TREE)paralleltree.c 
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@

paralleltree_test.o: $(SOURCE)$(TREE)paralleltree_test.c $(SOURCE)$(TREE)paralleltree.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

paralleltree_test: paralleltree_test.o paralleltree.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)paralleltree.o -o $@ $(LIB_FLAGS) -lpthread -lm

key_test.o: $(SOURCE)$(TREE)key_test.c $(SOURCE)$(TREE)paralleltree.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

key_test: key_test.o paralleltree.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)paralleltree.o -o $@ $(LIB_FLAGS) -lpthread -lm


.PHONY: clean

clean:
	rm -f key_test paralleltree_test bin/*.o

