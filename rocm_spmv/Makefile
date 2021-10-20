ROCSPARSE_INCLUDE=/opt/rocm/rocsparse/include
ROCSPARSE_LIB_PATH=/opt/rocm/rocsparse/lib
ROCSPARSE_LIB=rocsparse
HIP_INCLUDE=/opt/rocm/hip/include
LDFLAGS=-lm -L$(ROCSPARSE_LIB_PATH) -l$(ROCSPARSE_LIB)
LD=hipcc
CFLAGS=-w -I$(ROCSPARSE_INCLUDE) -I$(HIP_INCLUDE) -I.
CPP=hipcc
SRC=spmv_csr_largeDataset.cpp mmio_readbanner.cpp
OBJ=host.o
EXE=host.exe

#$(OBJ): $(SRC)
#	$(CPP) -c $(SRC) -o $@ $(CFLAGS)

$(EXE) : $(SRC)
	$(LD) $(CFLAGS) $(LDFLAGS) $(SRC) -o $@ 

run:
	./$(EXE) data/bcsstk36.mtx

clean:
	rm -f $(EXE) $(OBJ)

