# You may need to edit this file to reflect the type and capabilities of your system.
# The defaults are for a Linux system and may need to be changed for other systems (eg. Mac OS X).


CC = gcc -Wall

#CXX=CC
## When using the Sun Studio compiler




DEBUG = -DDEBUG
GDB = -g


OPT = -O2


#EXTRA_OPT = -fwhole-program
## Uncomment the above line if you're compiling all source files into one program in a single hit


#DEBUG = -DARMA_EXTRA_DEBUG
## Uncomment the above line to enable low-level debugging.
## Lots of debugging information will be printed when a compiled program is run.
## Please enable this option when reporting bugs.


#FINAL = -DARMA_NO_DEBUG
## Uncomment the above line to disable Armadillo's checks.
## Not recommended unless your code has been first thoroughly tested!


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





.PHONY: clean

clean:
	rm -f imagetest bin/*.o

