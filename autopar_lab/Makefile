# Sample Makefile
CC = suncc
OPT = -g -fast -xlic_lib=sunperf -xrestrict -xO5 -xautopar -xloopinfo
WARN = 
CFLAGS = $(OPT) $(WARN)
OBJECTS = main.o integrand.o

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
# DO NOT DELETE

main.o: integrand.h
