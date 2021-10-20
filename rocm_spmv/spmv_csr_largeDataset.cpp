/*
 *
 *
 * Code : Siddharth Mody
 *  Compare rocSparse on GPU vs SPMV CSR on CPU
 *
 */



#include <hip/hip_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <rocsparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <vector>
#include <map>
#include <cassert>
#include <chrono>
# include "mmio.hpp"


using DTYPE = float;
using uint32 = uint32_t;
//=======================================
void readSpec ( const char *input_filename,
	uint32 &row_cnt,
	uint32 &nnz_cnt,
	std::vector<DTYPE> &nnz,
	std::vector<uint32> &rowptr,
	std::vector<uint32> &colidx
    ) 
{
    FILE *input_file;
    MM_typecode matcode;
    int m,n,nz;
    int error;

    printf ( "\n" );
    printf ( "TEST01:\n" );
    printf ( "  MM_READ_BANNER reads the header line of\n" );
    printf ( "    a Matrix Market file.\n" );
    printf ( "\n" );
    printf ( "  Reading \"%s\".\n", input_filename );

    if ((input_file = fopen(input_filename, "r" )) == NULL) {
        std::cout << "Could not open matrix file" << std::endl;
        exit(1);
    }

    if (mm_read_banner(input_file, &matcode) != 0) {
      std::cout << "Could not process Matrix Market Banner" << std::endl;
      exit(1);
    }

    //  This is how one can screen matrix types if their application 
    //  only supports a subset of the Matrix Market data types.      
    if (!mm_is_real(matcode) || 
        !mm_is_coordinate(matcode) ||
        !mm_is_sparse(matcode) ) {   
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }   

    printf ( "\n" );
    printf ( "  MM_typecode[0] = %c\n", matcode[0] );
    printf ( "  MM_typecode[1] = %c\n", matcode[1] );
    printf ( "  MM_typecode[2] = %c\n", matcode[2] );
    printf ( "  MM_typecode[3] = %c\n", matcode[3] );

    // -------------------------------------------------------------------
    printf ( "  MM_READ_MTX_CRD_SIZE reads the lines of\n" );
    printf ( "    an MM coordinate file;\n" );
    if ((error = mm_read_mtx_crd_size( input_file, &m, &n, &nz )) != 0) exit (-1);
    assert(m == n);
    row_cnt = m;
    nnz_cnt = nz;
    printf ( "\n" );
    printf ( "  Coordinate sizes:\n" );
    printf ( "    M  = %d\n", row_cnt );
    printf ( "    N  = %d\n", row_cnt );     // It is always a square matrix
    printf ( "    NZ = %d\n", nnz_cnt );
    // -------------------------------------------------------------------
  
    std::map <std::pair<uint32, uint32>, DTYPE> matrix; // rowid,colid : nnz
    std::map <uint32, uint32> rowsize; // row id : rowlen

    for (auto k=0; k<nnz_cnt;k++) {
      uint32 ri, ci;
      DTYPE v;
      fscanf(input_file, "%d %d %f\n", &ri, &ci, &v);   // Reading from the input file
      assert(ri>=0 && ci>=0);
      matrix[{ri-1,ci-1}] = v;
      rowsize[ri-1]++;
      if( feof(input_file) ) {
        break;
      } 
    }
    fclose(input_file);

    std::cout << "Matrix Size :" << matrix.size() << std::endl;

    for (auto itr = matrix.begin(); itr != matrix.end(); ++itr) {
        nnz.push_back(itr->second);
        colidx.push_back(itr->first.second);
        //printf("rid = %d cid = %d nnz = %f\n", itr->first.first, itr->first.second, itr->second);
    }

    std::cout << "rowsize : " << rowsize.size() << std::endl;
    rowptr.push_back(0);
    uint32 prev_val = 0;
    for (auto itr = rowsize.begin(); itr != rowsize.end(); ++itr) {
        rowptr.push_back(itr->second + prev_val);
        prev_val += itr->second;
    }
    std::cout << "rowptr size : " << rowptr.size() << std::endl;

  } // end of test func
//=======================================
void generate_vector (
        std::vector<DTYPE> &vin,
        uint32 dim
        )
{
    srand(54321);
    for (auto i=0; i<dim; ++i) {
        DTYPE temp = (rand()/(float)RAND_MAX); 
        vin.push_back(temp);
        //printf("vin[%d] = %f\n", i, temp);
    }
}

//=======================================
void spmv_cpu (
	uint32 row_cnt,
	std::vector<DTYPE> &nnz,
	std::vector<uint32> &rowptr,
	std::vector<uint32> &colidx,
    std::vector<DTYPE> &vin,
    std::vector<DTYPE> &vout
    ) 
{

    for (auto i = 0; i < row_cnt; ++i) {
        DTYPE temp = 0;        
        for (auto j = rowptr[i]; j < rowptr[i+1]; ++j) {
            temp += nnz[j] * vin[colidx[j]];
        }
        vout[i] = temp;
    }
}
//=======================================
int main(int argc, char* argv[])
{
     if (argc < 2) {
        std::cout << "<>.exe <matrix dataset>" << std::endl;
        std::cout << "For eg : ./host.exe data/bwm200.mtx" << std::endl;
        return 1;
    }
    
    float     alpha           = 1.0f;
    float     beta            = 0.0f;
    std::string inFile = argv[1];
	uint32 row_cnt;
	uint32 nnz_cnt;
    std::vector<DTYPE> nnz;
    std::vector<uint32> rowptr;
    std::vector<uint32> colidx;

    readSpec(inFile.c_str(), row_cnt, nnz_cnt, nnz, rowptr, colidx);
    std::vector<DTYPE> vin;
    std::vector<DTYPE> vout(row_cnt, 0);
    generate_vector(vin, row_cnt);
    
    std::vector<DTYPE> vout_cpu(row_cnt);
    {
    auto start = std::chrono::steady_clock::now();
    spmv_cpu(row_cnt, nnz, rowptr, colidx, vin, vout_cpu);
    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;
    printf("CPU SPMV duration : %0.3f ms\n", elapsed.count());
    }

    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    hipDeviceProp_t devProp;
    int             device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    std::cout << "Device: " << devProp.name << std::endl;

    // Offload data to device
    uint32* dAptr = NULL;
    uint32* dAcol = NULL;
    DTYPE* dAval = NULL;
    DTYPE* dvin    = NULL;
    DTYPE* dvout    = NULL;

    hipMalloc((void**)&dAptr, sizeof(uint32) * (row_cnt + 1));
    hipMalloc((void**)&dAcol, sizeof(uint32) * nnz_cnt);
    hipMalloc((void**)&dAval, sizeof(DTYPE) * nnz_cnt);

    hipMalloc((void**)&dvin, sizeof(DTYPE) * row_cnt);
    hipMalloc((void**)&dvout, sizeof(DTYPE) * row_cnt);
    
    // Types 
    rocsparse_indextype itype = rocsparse_indextype_i32;
    rocsparse_indextype jtype = rocsparse_indextype_i32;
    rocsparse_datatype  ttype = rocsparse_datatype_f32_r;

    rocsparse_spmat_descr A;
    rocsparse_dnvec_descr x;
    rocsparse_dnvec_descr y;

    rocsparse_create_csr_descr(&A, row_cnt, row_cnt, nnz_cnt, dAptr, dAcol, dAval, itype, jtype, rocsparse_index_base_zero, ttype);
    rocsparse_create_dnvec_descr(&x, row_cnt, dvin, ttype);
    rocsparse_create_dnvec_descr(&y, row_cnt, dvout, ttype);
    void* temp_buffer;

  for(auto itr=0; itr<10; ++itr) {
    auto start = std::chrono::steady_clock::now();
    hipMemcpy(dAptr, rowptr.data(), sizeof(uint32) * (row_cnt + 1), hipMemcpyHostToDevice);
    hipMemcpy(dAcol, colidx.data(), sizeof(uint32) * nnz_cnt, hipMemcpyHostToDevice);
    hipMemcpy(dAval, nnz.data(), sizeof(DTYPE) * nnz_cnt, hipMemcpyHostToDevice);
    hipMemcpy(dvin, vin.data(), sizeof(DTYPE) * row_cnt, hipMemcpyHostToDevice);
    hipMemcpy(dvout, vout.data(), sizeof(DTYPE) * row_cnt, hipMemcpyHostToDevice);
    
    // Query for buffer size
    size_t buffer_size;
    rocsparse_spmv(handle, rocsparse_operation_none, &alpha, A, x, &beta, y, ttype, rocsparse_spmv_alg_default, &buffer_size, nullptr);

    hipMalloc(&temp_buffer, buffer_size);

    rocsparse_spmv(handle, rocsparse_operation_none, &alpha, A, x, &beta, y, ttype, rocsparse_spmv_alg_default, &buffer_size, temp_buffer);

    hipMemcpy(vout.data(), dvout, sizeof(DTYPE)*row_cnt, hipMemcpyDeviceToHost);
  
    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;
    printf("GPU SPMV duration : %0.3f ms\n", elapsed.count());
  }
    printf ( " Comparing top function with testbench \n");
    bool err = false;
    uint32 err_cnt = 0;
    for (int i = 0; i < row_cnt; i++) {
        auto diff = abs(vout_cpu[i] - vout[i]);
        auto valid = 0.02 * abs(vout[i]);
        if (diff > valid) {
            printf("diff = %f valid = %f\n", diff, valid);
            printf("cpu[%d] = %f gpu[%d] = %f\n", i, vout_cpu[i], i, vout[i]);
	        printf("FAILED : Result Mismatch\n\n");
            err = true;             
            err_cnt++;
        }
    }
    printf("Computed %d incorrect results\n", err_cnt);
    hipFree(temp_buffer);

    rocsparse_destroy_spmat_descr(A);
    rocsparse_destroy_dnvec_descr(x);
    rocsparse_destroy_dnvec_descr(y);

    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dvin);
    hipFree(dvout);

    rocsparse_destroy_handle(handle);

    return (err)? 1 : 0;
}
