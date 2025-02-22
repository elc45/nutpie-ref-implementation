data {
  int<lower=0> D;
}
parameters {
  vector<lower=0>[D] y;
}
model {
  y ~ normal(0, 1);
}
  