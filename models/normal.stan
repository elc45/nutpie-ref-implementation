data {
  int<lower=1> D;  // dimension
}

parameters {
  vector[D] y;
}

model {
  // Standard multivariate normal with identity covariance matrix
  y ~ multi_normal(rep_vector(-17, D), diag_matrix(rep_vector(1, D)));
}
