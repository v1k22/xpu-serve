// Mixed-radix decimation in time: a transform of L points at input stride s is p transforms of L / p points at stride
// s * p (the inputs congruent to q mod p), combined by butterflies of p points with twiddles exp(-2 pi i q u / L).

#include "audio/fft.h"

#include <cmath>
#include <cstdint>
#include <numbers>

#include "util/error.h"

namespace xs::audio {

RealFft::RealFft(int n) : n_(n) {
    XS_CHECK(n >= 1, "audio: an FFT of ", n, " points");
    m_ = n % 2 == 0 ? n / 2 : n;
    for (int rest = m_; rest > 1;) {
        int p = rest % 4 == 0 ? 4 : rest % 2 == 0 ? 2 : 3;
        while (rest % p != 0) p += 2;
        radices_.push_back(p);
        rest /= p;
    }
    const auto turn = [](int k, int len) {
        const double angle = -2.0 * std::numbers::pi * k / len;
        return Complex(std::cos(angle), std::sin(angle));
    };
    twiddles_.resize(static_cast<size_t>(m_));
    for (int k = 0; k < m_; ++k) twiddles_[static_cast<size_t>(k)] = turn(k, m_);
    if (n % 2 == 0) {
        unpack_.resize(static_cast<size_t>(m_) + 1);
        for (int k = 0; k <= m_; ++k) unpack_[static_cast<size_t>(k)] = turn(k, n);
    }
    packed_.resize(static_cast<size_t>(m_));
    spectrum_.resize(static_cast<size_t>(m_));
}

void RealFft::forward(const double* in, Complex* out) {
    if (n_ % 2 != 0) {
        for (int j = 0; j < n_; ++j) packed_[static_cast<size_t>(j)] = in[j];
        transform(spectrum_.data(), packed_.data(), 1, 0);
        for (int k = 0; k <= n_ / 2; ++k) out[k] = spectrum_[static_cast<size_t>(k)];
        return;
    }
    // z[j] = x[2j] + i x[2j + 1]; Z's even and odd parts are the transforms of x's even and odd samples.
    for (int j = 0; j < m_; ++j) packed_[static_cast<size_t>(j)] = Complex(in[2 * j], in[2 * j + 1]);
    transform(spectrum_.data(), packed_.data(), 1, 0);
    for (int k = 0; k <= m_; ++k) {
        const Complex z = spectrum_[static_cast<size_t>(k % m_)];
        const Complex zc = std::conj(spectrum_[static_cast<size_t>((m_ - k) % m_)]);
        const Complex even = 0.5 * (z + zc);
        const Complex odd = Complex(0.0, -0.5) * (z - zc);
        out[k] = even + unpack_[static_cast<size_t>(k)] * odd;
    }
}

void RealFft::transform(Complex* out, const Complex* in, int stride, size_t stage) const {
    if (stage == radices_.size()) {
        out[0] = in[0];
        return;
    }
    const int p = radices_[stage];
    const int sub = m_ / stride / p;
    for (int q = 0; q < p; ++q) transform(out + q * sub, in + q * stride, stride * p, stage + 1);
    butterfly(out, stride, sub, p);
}

void RealFft::butterfly(Complex* out, int stride, int sub, int p) const {
    const Complex* w = twiddles_.data();
    Complex y[8];
    std::vector<Complex> wide;  // a prime radix above 8
    Complex* v = p <= 8 ? y : (wide.resize(static_cast<size_t>(p)), wide.data());
    const auto minus_i = [](Complex z) { return Complex(z.imag(), -z.real()); };
    for (int u = 0; u < sub; ++u) {
        v[0] = out[u];
        for (int q = 1; q < p; ++q) v[q] = out[u + q * sub] * w[q * u * stride];
        if (p == 2) {
            out[u] = v[0] + v[1];
            out[u + sub] = v[0] - v[1];
        } else if (p == 4) {
            const Complex t0 = v[0] + v[2], t1 = v[0] - v[2], t2 = v[1] + v[3], t3 = minus_i(v[1] - v[3]);
            out[u] = t0 + t2;
            out[u + sub] = t1 + t3;
            out[u + 2 * sub] = t0 - t2;
            out[u + 3 * sub] = t1 - t3;
        } else if (p == 3) {
            constexpr double kSin = 0.86602540378443864676;  // sin(2 pi / 3)
            const Complex s = v[1] + v[2], d = minus_i(kSin * (v[1] - v[2]));
            const Complex mid = v[0] - 0.5 * s;
            out[u] = v[0] + s;
            out[u + sub] = mid + d;
            out[u + 2 * sub] = mid - d;
        } else if (p == 5) {
            constexpr double c1 = 0.30901699437494742410, c2 = -0.80901699437494742410;  // cos(2 pi k / 5), k = 1, 2
            constexpr double s1 = 0.95105651629515357212, s2 = 0.58778525229247312917;   // sin(2 pi k / 5)
            const Complex a1 = v[1] + v[4], b1 = v[1] - v[4], a2 = v[2] + v[3], b2 = v[2] - v[3];
            const Complex r1 = v[0] + c1 * a1 + c2 * a2, r2 = v[0] + c2 * a1 + c1 * a2;
            const Complex i1 = minus_i(s1 * b1 + s2 * b2), i2 = minus_i(s2 * b1 - s1 * b2);
            out[u] = v[0] + a1 + a2;
            out[u + sub] = r1 + i1;
            out[u + 2 * sub] = r2 + i2;
            out[u + 3 * sub] = r2 - i2;
            out[u + 4 * sub] = r1 - i1;
        } else {
            const int step = m_ / p;  // exp(-2 pi i / p) is twiddles_[step]
            for (int k = 0; k < p; ++k) {
                Complex sum = v[0];
                for (int q = 1; q < p; ++q) sum += v[q] * w[int64_t{q} * k % p * step];
                out[u + k * sub] = sum;
            }
        }
    }
}

}  // namespace xs::audio
