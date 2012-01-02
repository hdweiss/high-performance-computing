# Sample Makefile
CC = gcc
OPT = -g -O3
WARN = -Wall
CFLAGS = $(OPT) $(WARN) # the C compiler flags OBJECTS = data.o main.o io.o
OBJECTS = data.o main.o io.o

project1 : $(OBJECTS)
	$(CC) $(CFLAGS) -o project1 $(OBJECTS)

clean:
	@rm -f *.o core

realclean : clean
	@rm -f project1

# file dependecies
data.o : data.c data.h
main.o : data.h io.h main.c
io.o : io.h io.c
