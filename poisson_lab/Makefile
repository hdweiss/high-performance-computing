TARGET	= poisson
OBJS	= main.o poisson.o writepng.o 

OPT	= -g -fast -lm -xopenmp
ISA	= 
PARA	= 

PNGWRITERPATH = pngwriter
ARCH	      = $(shell uname -p)
PNGWRTLPATH   = $(PNGWRITERPATH)/lib/$(ARCH)
PNGWRTIPATH   = $(PNGWRITERPATH)/include
PNGWRITERLIB  = $(PNGWRTLPATH)/libpngwriter.a

CCC	= CC
CXX	= CC
CXXFLAGS= -I $(PNGWRTIPATH)

CFLAGS	= $(OPT) $(ISA) $(PARA) $(XOPT)

F90C  	= f90
LIBS	= -Yl,/usr/bin -L $(PNGWRTLPATH) -lpngwriter -lpng 


all: $(PNGWRITERLIB) $(TARGET)

$(TARGET): $(OBJS) 
	$(CCC) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

$(PNGWRITERLIB):
	@cd pngwriter/src && $(MAKE)

clean:
	@/bin/rm -f *.o core

realclean: clean
	@cd pngwriter/src && $(MAKE) clean
	@rm -f $(PNGWRITERLIB)
	@rm -f $(TARGET)

depend:
	makedepend -Y. *.c

# dependencies
# DO NOT DELETE

main.o: writepng.h poisson.h
poisson.o: poisson.h
