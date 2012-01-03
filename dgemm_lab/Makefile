# Sample Makefile
CC = gcc
OPT = -g -O3
WARN = -Wall
CFLAGS = $(OPT) $(WARN)
OBJECTS = main.o

project1 : $(OBJECTS)
	$(CC) $(CFLAGS) -o project1 $(OBJECTS)

clean:
	@rm -f *.o core

realclean : clean
	@rm -f project1

depend:
	makedepend -Y. *.c
# DO NOT DELETE
