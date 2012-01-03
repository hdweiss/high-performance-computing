# Sample Makefile
CC = g++
OPT = -g -O3 
WARN = -Wall
CFLAGS = $(OPT) $(WARN)
OBJECTS = main.o matrix.o

mm : $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(OBJECTS)

clean:
	@rm -f *.o core

realclean : clean
	@rm -f mm

depend:
	makedepend -Y. *.c
# DO NOT DELETE

main.o: matrix.h
matrix.o: matrix.h
