#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

#ifndef CPU_ONLY
template<>
void caffe_cpu_gemm<float16>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float16 alpha, const float16* A, const float16* B, const float16 beta,
    float16* C) {
  if (M <= 0 || N <= 0 || K <= 0) {
    return;
  }
  std::vector<float> a(M*K), b(K*N), c(M*N);
  caffe_cpu_convert(a.size(), A, &a.front());
  caffe_cpu_convert(b.size(), B, &b.front());
  caffe_cpu_convert(c.size(), C, &c.front());
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, static_cast<float>(alpha), &a.front(),
      lda, &b.front(), ldb, static_cast<float>(beta), &c.front(), N);
  caffe_cpu_convert(c.size(), &c.front(), C);
}
#endif

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

#ifndef CPU_ONLY
template <>
void caffe_cpu_gemv<float16>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float16 alpha, const float16* A, const float16* x,
    const float16 beta, float16* y) {
  if (M <= 0 || N <= 0) {
    return;
  }
  const int lx = (TransA == CblasNoTrans) ? N : M;
  const int ly = (TransA == CblasNoTrans) ? M : N;
  std::vector<float> a(M*N), xv(lx), yv(ly);
  caffe_cpu_convert(a.size(), A, &a.front());
  caffe_cpu_convert(xv.size(), x, &xv.front());
  caffe_cpu_convert(yv.size(), y, &yv.front());
  cblas_sgemv(CblasRowMajor, TransA, M, N, static_cast<float>(alpha), &a.front(), N,
      &xv.front(), 1, static_cast<float>(beta), &yv.front(), 1);
  caffe_cpu_convert(yv.size(), &yv.front(), y);
}
#endif

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

#ifndef CPU_ONLY
template<>
void caffe_axpy<float16>(const int N, const float16 alpha, const float16* X,
    float16* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha * X[i] + Y[i];
  }
}
#endif

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

#ifndef CPU_ONLY
template void caffe_set<float16>(const int N, const float16 alpha, float16* Y);
#endif

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

#ifndef CPU_ONLY
template <>
void caffe_add_scalar(const int N, const float16 alpha, float16* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = Y[i] + alpha;
  }
}
#endif

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      caffe_gpu_memcpy(sizeof(Dtype) * N, X, Y);
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

#ifndef CPU_ONLY
template void caffe_copy<float16>(const int N, const float16* X, float16* Y);
#endif

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

#ifndef CPU_ONLY
template <>
void caffe_scal<float16>(const int N, const float16 alpha, float16 *X) {
  // cblas_hscal(N, alpha, X, 1); ?
  for (int i = 0; i < N; ++i) {
    X[i] = alpha * X[i];
  }
}
#endif

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

#ifndef CPU_ONLY
template <>
void caffe_cpu_axpby<float16>(const int N, const float16 alpha,
    const float16* X, const float16 beta, float16* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha * X[i] + beta * Y[i];
  }
}
#endif

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

#ifndef CPU_ONLY
template <>
void caffe_add<float16>(const int n, const float16* a, const float16* b,
    float16* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] + b[i];
  }
}
#endif

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

#ifndef CPU_ONLY
template <>
void caffe_sub<float16>(const int n, const float16* a, const float16* b,
    float16* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] - b[i];
  }
}
#endif

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

#ifndef CPU_ONLY
template <>
void caffe_mul<float16>(const int n, const float16* a, const float16* b, float16* y) {
  for (int i = 0; i < n; ++i) {
    //  vhMul(n, a, b, y);
    y[i] = a[i] * b[i];
  }
}
#endif

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

#ifndef CPU_ONLY
template <>
void caffe_div<float16>(const int n, const float16* a, const float16* b, float16* y) {
  //  vhDiv(n, a, b, y);
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] / b[i];
  }
}
#endif

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

#ifndef CPU_ONLY
template <>
void caffe_powx<float16>(const int n, const float16* a, const float16 b,
    float16* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = pow(static_cast<float>(a[i]), static_cast<float>(b));
  }
}
#endif

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

#ifndef CPU_ONLY
template <>
void caffe_sqr<float16>(const int n, const float16* a, float16* y) {
  vhSqr(n, a, y);
}
#endif

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

#ifndef CPU_ONLY
template <>
void caffe_exp<float16>(const int n, const float16* a, float16* y) {
  vhExp(n, a, y);
}
template <>
void caffe_log<float16>(const int n, const float16* a, float16* y) {
  vhLn(n, a, y);
}
#endif  // ! CPU_ONLY

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

#ifndef CPU_ONLY
template <>
void caffe_abs<float16>(const int n, const float16* a, float16* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = fabs(a[i]);
  }
}
#endif

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(int n, float a, float b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<float> random_distribution(a, caffe_nextafter<float>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<float> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(int n, float a, float b, float* r);

template
void caffe_rng_uniform<double>(int n, float a, float b, double* r);

#ifndef CPU_ONLY
template
void caffe_rng_uniform<float16>(int n, float a, float b, float16* r);
#endif

template <typename Dtype>
void caffe_rng_gaussian(int n, float a, float sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<float> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<float> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(int n, float mu, float sigma, float* r);

template
void caffe_rng_gaussian<double>(int n, float mu, float sigma, double* r);

#ifndef CPU_ONLY
template
void caffe_rng_gaussian<float16>(const int n, float mu, float sigma, float16* r);
#endif

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

#ifndef CPU_ONLY
template<>
void caffe_rng_bernoulli<float16>(const int n, const float16 p, int* r) {
  caffe_rng_bernoulli(n, static_cast<const float>(p), r);
}
#endif

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

#ifndef CPU_ONLY
template<>
void caffe_rng_bernoulli<float16>(const int n, const float16 p,
    unsigned int* r) {
  caffe_rng_bernoulli(n, static_cast<const float>(p), r);
}
#endif

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

#ifndef CPU_ONLY
template <>
float16 caffe_cpu_strided_dot<float16>(const int n, const float16* x,
    const int incx, const float16 *y, const int incy) {
  float sum = 0.0f;
  int idx_x, idx_y;
  for (int i = 0; i < n; ++i) {
    idx_x = i*incx;
    idx_y = i*incy;
    sum += x[idx_x] * y[idx_y];
  }
  return float16(sum);
}
#endif

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);
template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

#ifndef CPU_ONLY
template
float16 caffe_cpu_dot<float16>(const int n, const float16* x, const float16* y);
#endif

template <>
int caffe_cpu_hamming_distance<float>(const int n, const float* x,
                                      const float* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcount(static_cast<uint32_t>(x[i]) ^
                               static_cast<uint32_t>(y[i]));
  }
  return dist;
}

template <>
int caffe_cpu_hamming_distance<double>(const int n, const double* x,
                                       const double* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcountl(static_cast<uint64_t>(x[i]) ^
                                static_cast<uint64_t>(y[i]));
  }
  return dist;
}

#ifndef CPU_ONLY
template <>
int caffe_cpu_hamming_distance<float16>(const int n, const float16* x,
                                        const float16* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcount(x[i].getx() ^ y[i].getx());
  }
  return dist;
}
#endif

template <typename Dtype>
Dtype caffe_cpu_amax(const int n, const Dtype* x) {
  Dtype y = 0.;
  for (int i = 0; i < n; ++i) {
    if (x[i] > 0)
      y = std::max(y, x[i]);
    else
      y = std::max(y, -x[i]);
  }
  return y;
}

template
float caffe_cpu_amax<float>(const int n, const float* x);
template
double caffe_cpu_amax<double>(const int n, const double* x);
#ifndef CPU_ONLY
template
float16 caffe_cpu_amax<float16>(const int n, const float16* x);
#endif

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}
template <>
float caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

#ifndef CPU_ONLY
template <>
float caffe_cpu_asum<float16>(const int n, const float16 *x) {
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum += fabs(x[i]);
  }
  return sum;
}
#endif

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

#ifndef CPU_ONLY
template <>
void caffe_cpu_scale<float16>(const int n, const float16 alpha,
    const float16 *x, float16 *y) {
  for (int i = 0; i < n; i++) {
    y[i] = alpha * x[i];
  }
}
#endif

// y[i]= max(a*x[i], b*y[i])
template <typename Dtype>
void caffe_cpu_eltwise_max(const int N, const Dtype alpha, const Dtype* x,
  const Dtype beta, Dtype* y) {
  for (int i = 0; i < N; ++i) {
    y[i] = std::max(alpha * x[i], beta * y[i]);
  }
}
template void caffe_cpu_eltwise_max<float>(const int N,
    const float alpha, const float* x, const float beta, float* y);
template void caffe_cpu_eltwise_max<double>(const int N,
    const double alpha, const double* x, const double beta, double* y);
#ifndef CPU_ONLY
template void caffe_cpu_eltwise_max<float16>(const int N,
    const float16 alpha, const float16* x, const float16 beta, float16* y);
#endif

// y[i]= min(a*x[i], b*y[i])
template <typename Dtype>
void caffe_cpu_eltwise_min(const int N, const Dtype alpha, const Dtype* x,
  const Dtype beta, Dtype* y) {
  for (int i = 0; i < N; ++i) {
    y[i] = std::min(alpha * x[i], beta * y[i]);
  }
}
template void caffe_cpu_eltwise_min<float>(const int N,
    const float alpha, const float* x, const float beta, float* y);
template void caffe_cpu_eltwise_min<double>(const int N,
    const double alpha, const double* x, const double beta, double* y);
#ifndef CPU_ONLY
template void caffe_cpu_eltwise_min<float16>(const int N,
    const float16 alpha, const float16* x, const float16 beta, float16* y);
#endif


}  // namespace caffe
