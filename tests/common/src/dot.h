namespace src {

    const char *dot =
R"(
void dot(TYPE * A __noalias __readonly __aligned(16),
         TYPE * B __noalias __readonly __aligned(16),
         TYPE * C,
         int M, int N, int K,
         int lda __multipleof(8),
         int ldb __multipleof(8),
         int ldc) {
  // prologue
  int ridx = get_program_id(0);
  int ridy = get_program_id(1);
  int rm[TM] = ridx * TM + 0 ... TM;
  int rn[TN] = ridy * TN + 0 ... TN;
  int rk[TK] = 0 ... TK;
  float c[TM, TN] = 0;
  // pointers to operands
  TYPE* pa[SHAPE_A] = A + rk[BROADCAST_AK] * STRIDE_AK + rm[BROADCAST_AM] * STRIDE_AM;
  TYPE* pb[SHAPE_B] = B + rk[BROADCAST_BK] * STRIDE_BK + rn[BROADCAST_BN] * STRIDE_BN;
  // prefetches operands
  bool checka[SHAPE_A] = rk[BROADCAST_AK] < K;
  bool checkb[SHAPE_B] = rk[BROADCAST_BK] < K;
  TYPE a[SHAPE_A] = checka ? *pa : 0;
  TYPE b[SHAPE_B] = checkb ? *pb : 0;
  // reduction loop
  for(int k = K; k > 0; k-= TK){
    c += USEA @ USEB;
    pa += TK * STRIDE_AK;
    pb += TK * STRIDE_BK;
    bool checka[SHAPE_A] = k > TK;
    bool checkb[SHAPE_B] = k > TK;
    a = checka ? *pa : 0;
    b = checkb ? *pb : 0;
  }
  // epilogue
  int rxm[TM] = get_program_id(0) * TM + 0 ... TM;
  int rxn[TN] = get_program_id(1) * TN + 0 ... TN;
  TYPE* pc[TM, TN] = C + rxm[:, newaxis] + rxn[newaxis, :] * ldc;
  bool checkc[TM, TN] = rxm[:, newaxis] && rxn[newaxis, :];
  *?(checkc)pc = c;
}
)";

}
