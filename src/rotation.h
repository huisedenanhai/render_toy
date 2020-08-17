#pragma once

#include "math_consts.h"
#include <cmath>

// row major matrix
template <int M, int N> struct Matrix {
  float data[M][N]{};

  inline float *operator[](size_t i) {
    return data[i];
  }

  inline const float *operator[](size_t i) const {
    return data[i];
  }
};

template <int M, int N, int K>
inline Matrix<M, K> operator*(const Matrix<M, N> &lhs,
                              const Matrix<N, K> &rhs) {
  Matrix<M, K> res{};
  // NOTE memory access pattern not optimized
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      res[i][j] = 0;
      for (int k = 0; k < N; k++) {
        res[i][j] += lhs[i][k] * rhs[k][j];
      }
    }
  }
  return res;
}

using Matrix3x3 = Matrix<3, 3>;

// rotate around base axis1
inline Matrix3x3
rotate_around_base(float rad, int axis1, int axis2, int axis3) {
  Matrix3x3 res{};
  auto cosTheta = cosf(rad);
  auto sinTheta = sinf(rad);
  res[axis1][axis1] = 1.0f;
  res[axis1][axis2] = 0.0f;
  res[axis1][axis3] = 0.0f;
  res[axis2][axis1] = 0.0f;
  res[axis2][axis2] = cosTheta;
  res[axis2][axis3] = sinTheta;
  res[axis3][axis1] = 0.0f;
  res[axis3][axis2] = -sinTheta;
  res[axis3][axis3] = cosTheta;
  return res;
}

inline Matrix3x3 euler_xyz_to_matrix(float degX, float degY, float degZ) {
  return rotate_around_base(Deg2Rad * degZ, 2, 0, 1) *
         rotate_around_base(Deg2Rad * degY, 1, 2, 0) *
         rotate_around_base(Deg2Rad * degX, 0, 1, 2);
}