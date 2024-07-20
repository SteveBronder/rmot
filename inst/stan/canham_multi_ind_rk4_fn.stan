//Growth function
functions{
  //Growth function for use with Runge-Kutta method
  //pars = (g_max, s_max, k)
  vector DE(vector y, real max_growth, real diameter, real ind_k){
    vector[size(y)] dydt = max_growth *
    exp(-0.5 * square(log(y / diameter) / ind_k));
    return dydt;
  }
  vector concat_arr_vec(array[] vector x) {
    int x_size = size(x);
    vector[x_size] x_tmp;
    for (i in 1:x_size) {
      x_tmp[i] = x[i][1];
    }
    return x_tmp;
  }

  real rk4_step(real y, real g_max, real S_max, real k, real interval){
    real k1;
    real k2;
    real k3;
    real k4;
    real y_hat;

    k1 = DE([y]', g_max, S_max, k)[1];
    k2 = DE([y+interval*k1/2.0]', g_max, S_max, k)[1];
    k3 = DE([y+interval*k2/2.0]', g_max, S_max, k)[1];
    k4 = DE([y+interval*k3]', g_max, S_max, k)[1];

    y_hat = y + (1.0/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4) * interval;

    return y_hat;
  }

  real rk4(real y, real g_max, real S_max, real k, real interval, real step_size){
    int steps;
    real duration;
    real y_hat;
    real step_size_temp;

    duration = 0;
    y_hat = y;

    while(duration < interval){
      //Determine the relevant step size
      step_size_temp = min([step_size, interval-duration]);

      //Get next size estimate
      y_hat = rk4_step(y_hat, g_max, S_max, k, step_size_temp);

      //Increment observed duration
      duration = duration + step_size_temp;
    }

    return y_hat;
  }
}

// Data structure
data {
  real step_size;
  int n_obs;
  int n_ind;
  vector[n_obs] y_obs;
  array[n_obs] int obs_index;
  array[n_obs] real time;
  array[n_obs] int ind_id;
  array[n_ind] real y_0_obs;
  array[n_ind, 2] int group_start_stop_idx;
}

// The parameters accepted by the model.
parameters {
  //Individual level
  vector<lower=0>[n_ind] ind_y_0;
  vector<lower=0>[n_ind] ind_max_growth;
  vector<lower=0>[n_ind] ind_diameter_at_max_growth;
  vector<lower=0>[n_ind] ind_k;

  //Species level
  real species_max_growth_mean;
  real<lower=0> species_max_growth_sd;
  real species_diameter_at_max_growth_mean;
  real<lower=0> species_diameter_at_max_growth_sd;
  real species_k_mean;
  real<lower=0> species_k_sd;

  //Global level
  real<lower=0> global_error_sigma;
}

model {
  vector[n_obs - 1 + 1] mu;
  // For each tree
  for (i in 1:n_ind) {
    // Start and end indices per tree
    int start_idx = group_start_stop_idx[i][1];
    int end_idx = group_start_stop_idx[i][2];
    int num_calcs = end_idx - start_idx;
    mu[start_idx] = ind_y_0[i];
    for (j in 1:(num_calcs)) {
      mu[start_idx + j] = rk4(mu[start_idx + j - 1],
                              ind_max_growth[i],
                              ind_diameter_at_max_growth[i],
                              ind_k[i],
                              time[start_idx + j] - time[start_idx + j - 1],
                              step_size);
    }
  }
  y_obs ~ normal(mu, global_error_sigma);
  //Likelihood

  //Priors
  //Individual level
  ind_y_0 ~ normal(y_0_obs, global_error_sigma);
  ind_max_growth ~lognormal(species_max_growth_mean,
                            species_max_growth_sd);
  ind_diameter_at_max_growth ~lognormal(species_diameter_at_max_growth_mean,
                                        species_diameter_at_max_growth_sd);
  ind_k ~lognormal(species_k_mean,
                   species_k_sd);

  //Species level
  species_max_growth_mean ~normal(0, 1);
  species_max_growth_sd ~cauchy(0, 1);
  species_diameter_at_max_growth_mean ~normal(0, 1);
  species_diameter_at_max_growth_sd ~cauchy(0, 1);
  species_k_mean ~normal(0, 1);
  species_k_sd ~cauchy(0, 1);

  //Global level
  global_error_sigma ~cauchy(0, 2);
}


