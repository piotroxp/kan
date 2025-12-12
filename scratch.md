# Kolmogorov–Arnold Networks (KANs) — Engineer's Markdown Guide

Below is a **practitioner-oriented distillation** of the article, organized as **key takeaways**, each paired with  
**(i) the core math idea** and **(ii) a minimal C++ sketch** showing how that idea would actually be implemented in a real KAN codebase.

The examples are **explicitly implementable**, framework-agnostic, and suitable for a systems-level C++ mindset.

---

## 1. What a KAN *is*, mathematically

### Core idea
A **KAN layer replaces post-mix activations with pre-mix univariate functions**:

\[
x^{(\ell+1)}_j = \sum_{i=1}^{n_\ell} \phi^{(\ell)}_{j,i}\big(x^{(\ell)}_i\big)
\]

Compare to MLP:

\[
x^{(\ell+1)} = \sigma(Wx + b)
\]

- **KAN** = activate → sum  
- **MLP** = sum → activate

This reversal is the fundamental structural change.

---

### Minimal C++ (KAN layer forward)

```cpp
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
```

---

## 2. KANs are **expressively equivalent** to MLPs

### Math statement

Any ReLU(^k) MLP can be rewritten as a spline-KAN:

\[
\sigma_k(w x + b) \approx \sum_{n} c_n B^{(k)}_n(x)
\]

Conversely, any spline-KAN can be unfolded into a wider MLP.

**Parameter scaling**

* KAN:    `O(G W² L)`
* Equivalent MLP: `O(G² W⁴ L)`

Same function class, fewer parameters.

---

### C++: piecewise-linear KAN → ReLU expansion

```cpp
// φ(x) = a0 + a1*x + sum_j α_j * ReLU(x - b_j)

double piecewise_linear_phi(double x,
                            double a0,
                            double a1,
                            const std::vector<double>& alpha,
                            const std::vector<double>& b) {
    double y = a0 + a1 * x;
    for (size_t j = 0; j < alpha.size(); ++j)
        y += alpha[j] * std::max(0.0, x - b[j]);
    return y;
}
```

---

## 3. Basis functions are the *real inductive bias*

### General expansion

Each edge function is expanded in a **basis**:

\[
\phi(x) = \sum_{n=0}^{N-1} c_n \psi_n(x)
\]

Accuracy, stability, and convergence depend primarily on the choice of \(\psi_n\).

---

## 4. B-spline KAN (default, safest choice)

### Math

\[
\phi(x) = \sum_{n} c_n B^{(k)}_n(x)
\]

Properties:

* Compact support
* \(C^{k-1}\) smoothness
* Excellent Sobolev/Besov convergence

---

### C++: cubic B-spline evaluation

```cpp
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
```

---

## 5. Chebyshev KAN (smooth / PDE-dominant problems)

### Math

\[
\phi(x) = \sum_{k=0}^{K} c_k T_k(\tanh x)
\]

Recurrence:
\[
T_{k+1}(x) = 2xT_k(x) - T_{k-1}(x)
\]

Benefits:

* Flat NTK spectrum
* Faster PDE convergence
* Excellent global smoothness

---

### C++: Chebyshev activation

```cpp
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
```

---

## 6. Gaussian / RBF KAN (fast spline alternative)

### Math

\[
\phi(x) = \sum_i w_i \exp\left(-\frac{(x - c_i)^2}{\varepsilon^2}\right)
\]

Trade-offs:

* Large \(\varepsilon\): smooth but ill-conditioned
* Small \(\varepsilon\): sharp, stable, local

---

### C++: RBF activation

```cpp
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
```

---

## 7. Fourier / Sinc KAN (high-frequency & discontinuities)

### SincKAN math

\[
\phi(x) = \sum_{k=-N}^{N} c_k \mathrm{sinc}\left(\frac{\pi}{h}(x - kh)\right)
\]

Ideal for:

* Shock waves
* Discontinuities
* Band-limited signals

---

### C++: Sinc activation

```cpp
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
```

---

## 8. Physics-Informed KANs (PIKANs)

### Loss formulation

\[
\mathcal{L} = \underbrace{|u_\theta - u|^2}_{\text{data}} + \underbrace{|\mathcal{N}[u_\theta]|^2}_{\text{PDE}} + \underbrace{|u_\theta|_{\partial\Omega} - g|^2}_{\text{BC}}
\]

KAN advantages:

* Reduced spectral bias
* Better-conditioned NTK
* Higher derivative accuracy

---

### C++: PDE residual (Poisson example)

```cpp
double poisson_residual(double u_xx, double u_yy, double f) {
    return u_xx + u_yy - f;
}
```

---

## 9. Convergence & scaling laws

For \(f \in W^s(\Omega)\):

\[
|f - f_{\text{KAN}}| \le C P^{-2s/d}
\]

MLPs achieve only:
\[
P^{-s/d}
\]

KANs double the convergence rate due to structured approximation.

---

## 10. Practical rule-of-thumb (engineer’s view)

| Problem Type       | Recommended KAN      |
| ------------------ | -------------------- |
| Generic regression | B-spline KAN         |
| Smooth PDE         | Chebyshev PIKAN      |
| Shocks / fronts    | SincKAN / rKAN       |
| Periodic           | FourierKAN           |
| Speed critical     | Gaussian / ReLU-KAN  |
| Large domains      | FBKAN (domain split) |

---

## Final takeaway

> **KANs are not “better MLPs” — they are structured function approximators.**

You trade cheap dense linear algebra for:

* Learned 1D function spaces
* Higher accuracy per parameter
* Better spectral control
* Interpretability
* Principled PDE convergence

The **basis choice matters more than depth or width**.

---

### Possible next steps

* Production-grade **C++ KAN class**
* Full **PIKAN paper → runnable C++**
* **KAN vs MLP FLOP benchmarking**
* **KAN checklist for systems engineers**
