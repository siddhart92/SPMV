#include <hip/hip_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <rocsparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc, char* argv[])
{
    // Host problem definition
    const int m      = 4;  // Row
    const int n      = 4;  // Col
    const int nnz    = 9;
    int       hAptr[] = { 0, 3, 4, 7, 9 };
    int       hAcol[] = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    float     hAval[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    float     hX[]            = { 1.0f, 2.0f, 3.0f, 4.0f };
    float     hY[]            = { 0.0f, 0.0f, 0.0f, 0.0f };
    float     hY_result[]     = { 19.0f, 8.0f, 51.0f, 52.0f };
    float     alpha           = 1.0f;
    float     beta            = 0.0f;
    ////////////////////////////////////////////

    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    hipDeviceProp_t devProp;
    int             device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    std::cout << "Device: " << devProp.name << std::endl;

    // Offload data to device
    int* dAptr = NULL;
    int* dAcol = NULL;
    float* dAval = NULL;
    float* dx    = NULL;
    float* dy    = NULL;

    hipMalloc((void**)&dAptr, sizeof(int) * (m + 1));
    hipMalloc((void**)&dAcol, sizeof(int) * nnz);
    hipMalloc((void**)&dAval, sizeof(float) * nnz);

    hipMalloc((void**)&dx, sizeof(float) * n);
    hipMalloc((void**)&dy, sizeof(float) * m);

    hipMemcpy(dAptr, hAptr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
    hipMemcpy(dAcol, hAcol, sizeof(int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dAval, hAval, sizeof(float) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dx, hX, sizeof(float) * n, hipMemcpyHostToDevice);
    hipMemcpy(dy, hY, sizeof(float) * n, hipMemcpyHostToDevice);

    // Types 
    rocsparse_indextype itype = rocsparse_indextype_i32;
    rocsparse_indextype jtype = rocsparse_indextype_i32;
    rocsparse_datatype  ttype = rocsparse_datatype_f32_r;
    

    rocsparse_spmat_descr A;
    rocsparse_dnvec_descr x;
    rocsparse_dnvec_descr y;

    rocsparse_create_csr_descr(&A, m, n, nnz, dAptr, dAcol, dAval, itype, jtype, rocsparse_index_base_zero, ttype);
    rocsparse_create_dnvec_descr(&x, n, dx, ttype);
    rocsparse_create_dnvec_descr(&y, m, dy, ttype);

    // Query for buffer size
    size_t buffer_size;
    rocsparse_spmv(handle, rocsparse_operation_none, &alpha, A, x, &beta, y, ttype, rocsparse_spmv_alg_default, &buffer_size, nullptr);

    void* temp_buffer;
    hipMalloc(&temp_buffer, buffer_size);

    rocsparse_spmv(handle, rocsparse_operation_none, &alpha, A, x, &beta, y, ttype, rocsparse_spmv_alg_default, &buffer_size, temp_buffer);

    hipMemcpy(hY,dy,sizeof(float)*m,hipMemcpyDeviceToHost);

    int correct = 1;
    for (int i = 0; i < m; i++) {
        if (hY[i] != hY_result[i]) { // direct floating point comparison is not reliable
            correct = 0;             
            break;
        }
    }

    if (correct)
        printf("spmv_csr_example test PASSED\n");
    else
        printf("spmv_csr_example test FAILED: wrong result\n");

    hipFree(temp_buffer);

    rocsparse_destroy_spmat_descr(A);
    rocsparse_destroy_dnvec_descr(x);
    rocsparse_destroy_dnvec_descr(y);

    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dx);
    hipFree(dy);

    rocsparse_destroy_handle(handle);

    return 0;
}
