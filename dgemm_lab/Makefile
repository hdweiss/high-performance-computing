# Sample Makefile
CC = suncc
OPT = -g -fast -xlic_lib=sunperf
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
# DO NOT DELETE

main.o: matrix.h xtime.h
matrix.o: matrix.h xtime.h
