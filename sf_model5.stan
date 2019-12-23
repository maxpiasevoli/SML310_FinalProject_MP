  
data {
  int<lower=0> sample_index;
  int<lower=0> y[sample_index];
  int<lower=0> n[sample_index];
  int<lower=0> sample_race[sample_index];
  int<lower=0> eth_pop[sample_index];
}
parameters {
  simplex[3] alpha;
  vector[sample_index / 3] beta;
  vector[sample_index] epsilon;
  vector[1] mu;
  vector[2] sigma;
}
transformed parameters {
  vector[sample_index] lambda;
  vector[sample_index] theta;
  
  for (i in 1:sample_index) {
    theta[i] = exp(log(eth_pop[i]) + alpha[sample_race[i]] + beta[(i+2)/3]);
    
    lambda[i] = (15.0/12.0) * theta[i] * exp(mu[1] + alpha[sample_race[i]] + beta[(i+2)/3] + epsilon[i]);
  }
}
model {
  y ~ poisson(lambda);
  n ~ poisson(theta);
  beta ~ normal(0, sigma[1]);
  epsilon ~ normal(0, sigma[2]);
}