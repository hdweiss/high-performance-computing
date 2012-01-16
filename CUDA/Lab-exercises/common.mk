# use "nvcc" to compile source files.
CC = nvcc

# the linker is also "nvcc". It might be something else with other compilers.
LD = nvcc

# include directories
INCLUDEDIRS = \
	-I$(CUDA_INSTALL_PATH) \
	-I$(CUDA_SDK_DIR) \
    -I$(CUDA_SDK_DIR)/common/inc 

# library directories
LIBDIRS = -L$(CUDA_SDK_DIR)/lib

# libraries to be linked with
LIBS = -lcutil

# compiler flags go here.
CFLAGS =  -g -O3 $(INCLUDEDIRS) -Dcimg_use_xshm -Dcimg_use_xrandr -Xptxas=-v $(ARCH)

# linker flags go here. Currently there aren't any, but if we'll switch to
# code optimization, we might add "-s" here to strip debug info and symbols.
LDFLAGS = $(LIBDIRS) $(LIBS)

# program executable file name.
#PROG = MyFirst

# list of generated object files.
OBJS = $(PROG).o

# list of dependencies and header files
DEPS = 

# top-level rule, to compile everything.
all: $(PROG) $(DEPS)

# rule to link the program
$(PROG): $(OBJS)
	$(LD) $(LDFLAGS) $(OBJS) -o $(PROG)

# now comes a meta-rule for compiling any "C" source file.

%.o: %.cu
	$(CC) $(CFLAGS) -c $<

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

%.0: %.c
	$(CC) $(CFLAGS) -c $<

# use this command to erase files.
RM = /bin/rm -f

# rule for cleaning re-compilable files.
clean:
	$(RM) $(PROG) $(OBJS) $(CUDA_PROFILE_LOG) cuda_profile_0.log

# rule for running compiled files.
run:
	./$(PROG) 
