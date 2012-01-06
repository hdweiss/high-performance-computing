# Sample Makefile
CC = suncc
OPT = -g -fast -xlic_lib=sunperf -xrestrict
WARN = 
CFLAGS = $(OPT) $(WARN)

OBJECTS = main.o matrix.o xtime.o

mm: $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(OBJECTS)


clean:
	@rm -f *.o core

realclean : clean
	@rm -f mm

depend:
	makedepend -Y. *.c

exp: cleanexp flops cache

cleanexp:
	rm -Rf experiments/*

flops:
	./runAnalyzerFlops.sh 2> /dev/null | grep -v WARNING
	./fetchFlops.sh | sort --general-numeric-sort > flops.dat
	gnuplot flops.p

cache:
	./runAnalyzerCacheMiss.sh 2> /dev/null | grep -v WARNING
	./fetchCacheMisses.sh | sort --general-numeric-sort > cache.dat
	gnuplot cache.p

# DO NOT DELETE

main.o: matrix.h xtime.h
matrix.o: matrix.h xtime.h
