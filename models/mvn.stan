data {
  int<lower=1> D;  // dimension
}

parameters {
  vector[D] x;
}

model {
  matrix[D, D] Sigma;  // static covariance matrix
  
  // Fill covariance matrix with AR(1)-like structure with rho=0.7
  for (i in 1:D) {
    for (j in 1:D) {
      Sigma[i,j] = 0.75^abs(i-j);  // correlation decays exponentially with distance
    }
  }

  // Prior - multivariate normal with zero mean and Sigma covariance
  x ~ multi_normal(rep_vector(0, D), Sigma);
}
