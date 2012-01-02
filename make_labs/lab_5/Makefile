CC	= gcc
DEBUG	= -g
OPT	= -O3
MATH    = -lm
CFLAGS	= $(DEBUG) $(OPT) $(MATH)
OBJECTS = data.o main.o io.o

project1: $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(OBJECTS) 

data.o: data.c data.h
main.o: data.h io.h main.c
io.o  : io.h io.c

clean:
	/bin/rm -f *.o core 

realclean: clean
	/bin/rm -f project1

