# Sample Makefile
CC = suncc
CHIP = -xchip=opteron
OPT = -g -fast -xlic_lib=sunperf -xrestrict -xO5 $(CHIP)
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

exp: cleanexp flops cache blocks

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

blocks:
	./runAnalyzerBlocks.sh 2> /dev/null | grep -v WARNING
	./fetchBlocks.sh | sort --general-numeric-sort > blocks.dat
	gnuplot blocks.p	

# DO NOT DELETE

main.o: matrix.h xtime.h
matrix.o: matrix.h xtime.h
