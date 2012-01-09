# Sample Makefile
CC = suncc
OPENMP = -xopenmp
OPT = -g -fast -xlic_lib=sunperf -xrestrict -xO5 -lm -xautopar -xloopinfo
CFLAGS = $(OPT)
OBJECTS = main.o mxv.o

all : main main_mp

main : $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(OBJECTS)

main_mp: $(OBJECTS)
	$(CC) $(CFLAGS) $(OPENMP) -o $@ $(OBJECTS)

clean:
	@rm -f *.o core

realclean : clean
	@rm -f main main_mp main_unopt

depend:
	makedepend -Y. *.c

runtest:
	./runTest.sh

# DO NOT DELETE

main.o: mxv.h
mxv.o: mxv.h
