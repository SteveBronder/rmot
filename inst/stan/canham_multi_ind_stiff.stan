//Growth function
functions{
  //Growth function for use with Runge-Kutta method
  //pars = (g_max, s_max, k)
  vector DE(real t, vector y, real max_growth, real diameter, real ind_k){
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
}

// Data structure
data {
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
  vector[3] pars;

  vector[n_obs - 1 + 1] mu;
  // For each tree
  for (i in 1:n_ind) {
    // Start and end indices per tree
    int start_idx = group_start_stop_idx[i][1];
    int end_idx = group_start_stop_idx[i][2];
    int num_calcs = end_idx - start_idx;
    mu[start_idx] = ind_y_0[i];
    for (j in 1:(num_calcs)) {
      mu[start_idx + j] = ode_bdf(DE, [mu[start_idx + j - 1]]',
        time[start_idx + j - 1], {time[start_idx + j]},
        ind_max_growth[i], ind_diameter_at_max_growth[i], ind_k[i])[1][1];
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
  species_max_growth_mean ~student_t(4, 0, 1);
  species_max_growth_sd ~ student_t(3, 0, 1);
  species_diameter_at_max_growth_mean ~student_t(4, 0, 1);
  species_diameter_at_max_growth_sd ~ student_t(3, 0, 1);
  species_k_mean ~student_t(4, 0, 1);
  species_k_sd ~ student_t(3, 0, 1);

  //Global level
  global_error_sigma ~ student_t(3, 0, 1);
}


