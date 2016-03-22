#include <bl_config.h>
#include "bl_dgemm_kernel.h"

//micro-panel a is stored in column major, lda=DGEMM_MR.
//#define a(i,j) a[ (j)*DGEMM_MR + (i) ]
//micro-panel b is stored in row major, ldb=DGEMM_NR.
//#define b(i,j) b[ (i)*DGEMM_NR + (j) ]
//result      c is stored in column major.
//#define c(i,j) c[ (j)*ldc + (i) ]

//micro-panel a is stored in column major, lda=DGEMM_MR.
#define a(i,j) (a + (j)*DGEMM_MR + (i) )
//micro-panel b is stored in row major, ldb=DGEMM_NR.
#define b(i,j) (b + (i)*DGEMM_NR + (j) )
//result      c is stored in column major.
#define c(i,j) (c + (j)*ldc + (i) )

void bl_dgemm_ukr( int    k,
                   double *a,
                   double *b,
                   double *c,
                   dim_t ldc,
                   aux_t* data )
{
    dim_t l, j, i;

    for ( l = 0; l < DGEMM_NR / k; ++l )
    {
        for ( i = 0; i < DGEMM_NR; ++i )
        {
            bl_daxpy_asm_4x1( b + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR,
                              a + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR,
                              c + DGEMM_NR * i);

            bl_daxpy_asm_4x1( b + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR + 1,
                              a + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR,
                              c + DGEMM_NR * i);

            bl_daxpy_asm_4x1( b + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR + 2,
                              a + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR,
                              c + DGEMM_NR * i);

            bl_daxpy_asm_4x1( b + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR + 3,
                              a + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR,
                              c + DGEMM_NR * i);

            bl_daxpy_asm_4x1( b + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR + 4,
                              a + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR,
                              c + DGEMM_NR * i);

            bl_daxpy_asm_4x1( b + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR + 5,
                              a + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR,
                              c + DGEMM_NR * i);

            bl_daxpy_asm_4x1( b + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR + 6,
                              a + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR,
                              c + DGEMM_NR * i);

            bl_daxpy_asm_4x1( b + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR + 7,
                              a + l * DGEMM_NR * DGEMM_NR + i * DGEMM_NR,
                              c + DGEMM_NR * i);

                //c( i, j ) += a( i, l ) * b( l, j );
            
        }
    }

}

void bl_daxpy_asm_4x1(
        double *alpha,
        double *x,
        double *y
        )
{
    __asm__ volatile
    (
  "                                            \n\t"
  "movq                %1, %%rax               \n\t" // load address of x.              ( v )
  "movq                %2, %%rbx               \n\t" // load address of y.              ( v )
  "movq                %0, %%rcx               \n\t" // load address of alpha.          ( s )
  "                                            \n\t"
    "vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to 0                   ( v )
    "vxorpd    %%ymm1,  %%ymm1,  %%ymm1          \n\t" // set ymm1 to 0                   ( v )
    "vxorpd    %%ymm2,  %%ymm2,  %%ymm2          \n\t" // set ymm2 to 0                   ( v )
  "                                            \n\t"
    "vmovapd   0 * 32(%%rax), %%ymm0             \n\t" // load x
    "vmovapd   0 * 32(%%rbx), %%ymm1             \n\t" // load y
  "                                            \n\t"
  "vbroadcastsd       0 *  8(%%rcx), %%ymm2    \n\t" // load alpha, broacast to ymm2
  "vfmadd231pd       %%ymm2, %%ymm0, %%ymm1    \n\t" // y := alpha * x + y (fma)
  "vmovaps           %%ymm1, 0 * 32(%%rbx)     \n\t" // store back y
  "                                            \n\t"
  ".DDONE:                                     \n\t"
  "                                            \n\t"
  : // output operands (none)
  : // input operands
    "m" (alpha),        // 0
    "m" (x),            // 1
    "m" (y)             // 2
  : // register clobber list
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
    "xmm0", "xmm1", "xmm2", "xmm3",
    "xmm4", "xmm5", "xmm6", "xmm7",
    "xmm8", "xmm9", "xmm10", "xmm11",
    "xmm12", "xmm13", "xmm14", "xmm15",
    "memory"
  );
}

