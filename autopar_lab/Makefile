# Sample Makefile
CC = suncc
OPENMP = -xopenmp
SERIAL = -xautopar -xloopinfo
OPT = -g -fast -xlic_lib=sunperf -xrestrict -xO5
CFLAGS = $(OPT)
OBJECTS = main.o integrand.o mxv.o

all : main main_mp main_unopt

main : $(OBJECTS)
	$(CC) $(CFLAGS) $(SERIAL) -o $@ $(OBJECTS)

main_unopt: $(OBJECTS)
	$(CC) -o $@ $(OBJECTS)

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

main.o: integrand.h mxv.h
mxv.o: mxv.h
