#pragma once

#include <complex>
#include <vector>

namespace xs::audio {

// The discrete Fourier transform of real input of one length n, in double precision:
// X[k] = sum_j x[j] exp(-2 pi i j k / n) for k = 0 .. n / 2 (the other bins are their conjugates).
// Any n: an even n runs as a complex transform of n / 2 points, factored into radices 4, 2, 3, 5 and any larger prime
// (a prime factor p costs O(p) per point). Holds scratch space, so one object serves one thread.
class RealFft {
public:
    explicit RealFft(int n);

    int size() const { return n_; }

    // in: n values; out: n / 2 + 1 bins.
    void forward(const double* in, std::complex<double>* out);

private:
    using Complex = std::complex<double>;

    void transform(Complex* out, const Complex* in, int stride, size_t stage) const;
    void butterfly(Complex* out, int stride, int sub, int p) const;

    int n_ = 0;
    int m_ = 0;                   // complex transform length: n / 2 for an even n, else n
    std::vector<int> radices_;    // factors of m_, in the order the transform splits by them
    std::vector<Complex> twiddles_;  // exp(-2 pi i k / m_), k < m_
    std::vector<Complex> unpack_;    // exp(-2 pi i k / n), k <= n / 2 (even n)
    std::vector<Complex> packed_, spectrum_;
};

}  // namespace xs::audio
