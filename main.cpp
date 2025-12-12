//
// Created by piotro on 12.12.2025.
//

// x_in: [n_in]
// phi[j][i] : univariate function on edge i -> j
// x_out: [n_out]

void kan_layer_forward(
    const std::vector<double>& x_in,
    std::vector<double>& x_out,
    const std::vector<std::vector<std::function<double(double)>>>& phi
) {
  int n_out = phi.size();
  int n_in  = x_in.size();
  x_out.assign(n_out, 0.0);

  for (int j = 0; j < n_out; ++j) {
    for (int i = 0; i < n_in; ++i) {
      x_out[j] += phi[j][i](x_in[i]);
    }
  }
}

// φ(x) = a0 + a1*x + sum_j α_j * ReLU(x - b_j)

double piecewise_linear_phi(double x,
                            double a0,
                            double a1,
                            const std::vector<double>& alpha,
                            const std::vector<double>& b)
{
  double y = a0 + a1 * x;
  for (size_t j = 0; j < alpha.size(); ++j)
    y += alpha[j] * std::max(0.0, x - b[j]);
  return y;
}

double B3(double t) {
  t = std::abs(t);
  if (t < 1.0)
    return (4.0 - 6.0*t*t + 3.0*t*t*t) / 6.0;
  if (t < 2.0)
    return std::pow(2.0 - t, 3) / 6.0;
  return 0.0;
}

double spline_phi(double x,
                  const std::vector<double>& coeffs,
                  double h) {
  double y = 0.0;
  for (int i = 0; i < coeffs.size(); ++i) {
    double t = (x - i*h) / h;
    y += coeffs[i] * B3(t);
  }
  return y;
}

double chebyshev_phi(double x,
                     const std::vector<double>& c) {
  double z = std::tanh(x);
  double Tkm1 = 1.0;
  double Tk   = z;
  double y = c[0] * Tkm1 + c[1] * Tk;

  for (size_t k = 2; k < c.size(); ++k) {
    double Tkp1 = 2*z*Tk - Tkm1;
    y += c[k] * Tkp1;
    Tkm1 = Tk;
    Tk   = Tkp1;
  }
  return y;
}

double rbf_phi(double x,
               const std::vector<double>& w,
               const std::vector<double>& c,
               double eps) {
  double y = 0.0;
  for (size_t i = 0; i < w.size(); ++i) {
    double d = (x - c[i]) / eps;
    y += w[i] * std::exp(-d*d);
  }
  return y;
}

double sinc(double x) {
  return (std::abs(x) < 1e-8) ? 1.0 : std::sin(x) / x;
}

double sinc_phi(double x,
                const std::vector<double>& c,
                double h) {
  int N = (c.size() - 1) / 2;
  double y = 0.0;
  for (int k = -N; k <= N; ++k) {
    y += c[k + N] * sinc(M_PI * (x - k*h) / h);
  }
  return y;
}

double poisson_residual(double u_xx, double u_yy, double f) {
  return u_xx + u_yy - f;
}
