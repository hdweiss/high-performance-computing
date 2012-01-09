# Sample Makefile
CC = suncc
OPT = -g -fast -xlic_lib=sunperf -xrestrict -xO5
WARN = 
CFLAGS = $(OPT) $(WARN) # the C compiler flags OBJECTS = data.o main.o io.o
OBJECTS = main.o

autopar : $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(OBJECTS)

clean:
	@rm -f *.o core

realclean : clean
	@rm -f autopar

depend:
	makedepend -Y. *.c

run4:
	OMP_NUM_THREADS=4 ./autopar
