"""GPU implementations of the regularized incomplete beta and its inverse.

The regime selection, Lanczos-scaled power term, and inverse starting values
are adapted from Boost.Math. Boost.Math is distributed under the Boost
Software License, Version 1.0:
https://www.boost.org/LICENSE_1_0.txt

The kernels deliberately calculate in double precision. PAL uses these
functions for distribution and copula calculations where tail accuracy is
more important than preserving a float32 input dtype.
"""

from __future__ import annotations

import importlib
import typing as t

cp = t.cast(t.Any, importlib.import_module("cupy"))

_BETA_PREAMBLE = r"""
#include <cupy/math_constants.h>

#define PAL_DBL_MIN 2.22507385850720138309e-308
#define PAL_DBL_MAX 1.79769313486231570815e+308
#define PAL_DBL_EPSILON 2.22044604925031308085e-16
#define PAL_PI 3.141592653589793238462643383279502884

// Boost.Math lanczos13m53 coefficients, Boost Software License 1.0.
__constant__ double pal_lanczos_num[13] = {
    56906521.91347156388090791033559122686859,
    103794043.1163445451906271053616070238554,
    86363131.28813859145546927288977868422342,
    43338889.32467613834773723740590533316085,
    14605578.08768506808414169982791359218571,
    3481712.15498064590882071018964774556468,
    601859.6171681098786670226533699352302507,
    75999.29304014542649875303443598909137092,
    6955.999602515376140356310115515198987526,
    449.9445569063168119446858607650988409623,
    19.51992788247617482847860965652136208,
    0.5098416655656676188125178644804694509993,
    0.006061842346248906525783753964555936883222
};

__constant__ double pal_lanczos_den[13] = {
    0.0, 39916800.0, 120543840.0, 150917976.0, 105258076.0,
    45995730.0, 13339535.0, 2637558.0, 357423.0, 32670.0,
    1925.0, 66.0, 1.0
};

__device__ __forceinline__ double pal_lanczos_sum(const double z) {
    double numerator;
    double denominator;
    if (z <= 1.0) {
        numerator = pal_lanczos_num[12];
        denominator = pal_lanczos_den[12];
        for (int i = 11; i >= 0; --i) {
            numerator = numerator * z + pal_lanczos_num[i];
            denominator = denominator * z + pal_lanczos_den[i];
        }
    } else {
        numerator = pal_lanczos_num[0];
        denominator = pal_lanczos_den[0];
        for (int i = 1; i < 13; ++i) {
            numerator = numerator / z + pal_lanczos_num[i];
            denominator = denominator / z + pal_lanczos_den[i];
        }
    }
    return numerator / denominator;
}

__device__ __forceinline__ double pal_log_beta(double a, double b) {
    if (a < b) {
        const double temporary = a;
        a = b;
        b = temporary;
    }
    const double c = a + b;
    const double g = 6.024680040776729583740234375;
    const double agh = a + g - 0.5;
    const double bgh = b + g - 0.5;
    const double cgh = c + g - 0.5;
    const double ambh = a - 0.5 - b;

    double result = log(pal_lanczos_sum(a));
    result += log(pal_lanczos_sum(b));
    result -= log(pal_lanczos_sum(c));
    if ((fabs(b * ambh) < cgh * 100.0) && (a > 100.0)) {
        result += ambh * log1p(-b / cgh);
    } else {
        result += (a - 0.5 - b) * log(agh / cgh);
    }
    result += b * (log(agh / cgh) + log(bgh / cgh));
    result += 0.5 * (1.0 - log(bgh));
    return result;
}

// Boost's scaled Lanczos formulation avoids cancellation when a and b are
// large and x is close to the beta distribution's mode.
__device__ __forceinline__ double pal_ibeta_power_terms(
    const double a, const double b, const double x, const double y
) {
    const double c = a + b;
    const double gh = 5.524680040776729583740234375;
    const double agh = a + gh;
    const double bgh = b + gh;
    const double cgh = c + gh;

    double log_result = log(pal_lanczos_sum(c));
    log_result -= log(pal_lanczos_sum(a));
    log_result -= log(pal_lanczos_sum(b));
    log_result += 0.5 * (log(bgh) - 1.0);
    log_result += 0.5 * (log(agh) - log(cgh));

    const double l1 = ((x * b - y * a) - y * gh) / agh;
    const double l2 = ((y * a - x * b) - x * gh) / bgh;
    log_result += fabs(l1) < 0.5
        ? a * log1p(l1)
        : a * log((x * cgh) / agh);
    log_result += fabs(l2) < 0.5
        ? b * log1p(l2)
        : b * log((y * cgh) / bgh);

    if (log_result < log(PAL_DBL_MIN)) {
        return 0.0;
    }
    return exp(log_result);
}

__device__ __forceinline__ double pal_ibeta_fraction(
    const double a, const double b, const double x
) {
    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    const double tiny = PAL_DBL_MIN / PAL_DBL_EPSILON;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (fabs(d) < tiny) {
        d = tiny;
    }
    d = 1.0 / d;
    double result = d;

    for (int m = 1; m <= 256; ++m) {
        const double m2 = 2.0 * m;
        double coefficient = m * (b - m) * x;
        coefficient /= (qam + m2) * (a + m2);
        d = 1.0 + coefficient * d;
        c = 1.0 + coefficient / c;
        if (fabs(d) < tiny) {
            d = tiny;
        }
        if (fabs(c) < tiny) {
            c = tiny;
        }
        d = 1.0 / d;
        result *= d * c;

        coefficient = -(a + m) * (qab + m) * x;
        coefficient /= (a + m2) * (qap + m2);
        d = 1.0 + coefficient * d;
        c = 1.0 + coefficient / c;
        if (fabs(d) < tiny) {
            d = tiny;
        }
        if (fabs(c) < tiny) {
            c = tiny;
        }
        d = 1.0 / d;
        const double delta = d * c;
        result *= delta;
        if (fabs(delta - 1.0) <= 8.0 * PAL_DBL_EPSILON) {
            break;
        }
    }
    return result;
}

__device__ __forceinline__ double pal_ibeta(
    const double a, const double b, const double x
) {
    if (isnan(a) || isnan(b) || isnan(x)) {
        return CUDART_NAN;
    }
    if (!(a > 0.0) || !(b > 0.0) || x < 0.0 || x > 1.0) {
        return CUDART_NAN;
    }
    if (x == 0.0) {
        return 0.0;
    }
    if (x == 1.0) {
        return 1.0;
    }
    if ((a == b) && (x == 0.5)) {
        return 0.5;
    }
    if ((a == 0.5) && (b == 0.5)) {
        return 2.0 * asin(sqrt(x)) / PAL_PI;
    }
    if (b == 1.0) {
        return exp(a * log(x));
    }
    if (a == 1.0) {
        return -expm1(b * log1p(-x));
    }

    const double switch_point = (a + 1.0) / (a + b + 2.0);
    double result;
    if (x <= switch_point) {
        result = pal_ibeta_power_terms(a, b, x, 1.0 - x);
        result *= pal_ibeta_fraction(a, b, x) / a;
    } else {
        result = pal_ibeta_power_terms(b, a, 1.0 - x, x);
        result *= pal_ibeta_fraction(b, a, 1.0 - x) / b;
        result = 1.0 - result;
    }
    return fmin(1.0, fmax(0.0, result));
}

// Section 2 of Temme (1992), as used by Boost for nearly symmetric large
// parameters. This gives the inverse iteration a close, inexpensive start.
__device__ __forceinline__ double pal_temme_inverse_start(
    const double a, const double b, const double p
) {
    const double root_two = 1.4142135623730950488;
    double eta0 = erfcinv(2.0 * p) / -sqrt(a / 2.0);
    const double difference = b - a;
    const double difference2 = difference * difference;
    const double difference3 = difference2 * difference;
    double terms[4];
    double workspace[7];
    terms[0] = eta0;

    workspace[0] = -difference * root_two / 2.0;
    workspace[1] = (1.0 - 2.0 * difference) / 8.0;
    workspace[2] = -difference * root_two / 48.0;
    workspace[3] = -1.0 / 192.0;
    workspace[4] = -difference * root_two / 3840.0;
    terms[1] = workspace[4];
    for (int i = 3; i >= 0; --i) {
        terms[1] = terms[1] * eta0 + workspace[i];
    }

    workspace[0] = difference * root_two * (3.0 * difference - 2.0) / 12.0;
    workspace[1] = (20.0 * difference2 - 12.0 * difference + 1.0) / 128.0;
    workspace[2] = difference * root_two * (20.0 * difference - 1.0) / 960.0;
    workspace[3] = (16.0 * difference2 + 30.0 * difference - 15.0) / 4608.0;
    workspace[4] = difference * root_two * (21.0 * difference + 32.0) / 53760.0;
    workspace[5] = (-32.0 * difference2 + 63.0) / 368640.0;
    workspace[6] = -difference * root_two * (120.0 * difference + 17.0) / 25804480.0;
    terms[2] = workspace[6];
    for (int i = 5; i >= 0; --i) {
        terms[2] = terms[2] * eta0 + workspace[i];
    }

    workspace[0] = difference * root_two * (-75.0 * difference2 + 80.0 * difference - 16.0) / 480.0;
    workspace[1] = (-1080.0 * difference3 + 868.0 * difference2 - 90.0 * difference - 45.0) / 9216.0;
    workspace[2] = difference * root_two * (-1190.0 * difference2 + 84.0 * difference + 373.0) / 53760.0;
    workspace[3] = (-2240.0 * difference3 - 2508.0 * difference2 + 2100.0 * difference - 165.0) / 368640.0;
    terms[3] = workspace[3];
    for (int i = 2; i >= 0; --i) {
        terms[3] = terms[3] * eta0 + workspace[i];
    }

    const double inverse_a = 1.0 / a;
    const double eta = ((terms[3] * inverse_a + terms[2]) * inverse_a + terms[1])
        * inverse_a + terms[0];
    const double eta2 = eta * eta;
    if (eta2 == 0.0) {
        return 0.5;
    }
    const double exponential = -exp(-eta2 / 2.0);
    return fmin(1.0, fmax(0.0, (1.0 + eta * sqrt((1.0 + exponential) / eta2)) / 2.0));
}

__device__ __forceinline__ double pal_ibeta_inverse(
    double a, double b, double p
) {
    if (isnan(a) || isnan(b) || isnan(p)) {
        return CUDART_NAN;
    }
    if (!(a > 0.0) || !(b > 0.0) || p < 0.0 || p > 1.0) {
        return CUDART_NAN;
    }
    if (p == 0.0) {
        return 0.0;
    }
    if (p == 1.0) {
        return 1.0;
    }

    bool reflect = false;
    if (p > 0.5) {
        const double temporary = a;
        a = b;
        b = temporary;
        p = 1.0 - p;
        reflect = true;
    }

    double x;
    if ((a == 0.5) && (b == 0.5)) {
        const double sine = sin(p * PAL_PI / 2.0);
        x = sine * sine;
    } else if (b == 1.0) {
        x = exp(log(p) / a);
    } else if (a == 1.0) {
        x = -expm1(log1p(-p) / b);
    } else {
        const double log_beta = pal_log_beta(a, b);
        const double minimum = fmin(a, b);
        const double maximum = fmax(a, b);

        if ((minimum > 5.0) && (sqrt(minimum) > maximum - minimum)) {
            x = pal_temme_inverse_start(a, b, p);
        } else if ((a > 1.0) && (b > 1.0)) {
            // Didonato-Morris/AS109 approximation, also used as a Boost
            // fallback away from the Temme regions.
            const double normal = -normcdfinv(p);
            const double correction = (normal * normal - 3.0) / 6.0;
            const double scale = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
            const double root = fmax(0.0, scale + correction);
            double exponent = normal * sqrt(root) / scale;
            exponent -= (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0))
                * (correction + 5.0 / 6.0 - 2.0 / (3.0 * scale));
            exponent *= 2.0;
            if (exponent > 700.0) {
                x = PAL_DBL_MIN;
            } else if (exponent < -700.0) {
                x = 1.0 - PAL_DBL_EPSILON;
            } else {
                x = a / (a + b * exp(exponent));
            }
        } else {
            // Boost's small-shape start follows from
            // I_x(a,b) ~ x^a / (a B(a,b)).
            const double log_x = (log(p) + log(a) + log_beta) / a;
            if (log_x >= 0.0) {
                x = a / (a + b);
            } else if (log_x < log(PAL_DBL_MIN)) {
                x = PAL_DBL_MIN;
            } else {
                x = exp(log_x);
                if ((a < 1.0) && (b < 1.0)) {
                    x /= 1.0 + x;
                }
            }
        }

        x = fmin(1.0 - PAL_DBL_EPSILON, fmax(PAL_DBL_MIN, x));
        double lower = 0.0;
        double upper = 1.0;

        // Difficult small-shape tails can require many safeguarded bisection
        // steps when the density underflows and Halley's step is unavailable.
        for (int iteration = 0; iteration < 96; ++iteration) {
            const double probability = pal_ibeta(a, b, x);
            const double residual = probability - p;
            if (residual < 0.0) {
                lower = x;
            } else {
                upper = x;
            }
            if (fabs(residual) <= 8.0 * PAL_DBL_EPSILON * fmax(p, PAL_DBL_MIN)) {
                break;
            }

            const double log_density = (a - 1.0) * log(x)
                + (b - 1.0) * log1p(-x) - log_beta;
            double candidate = CUDART_NAN;
            if ((log_density > log(PAL_DBL_MIN)) &&
                (log_density < log(PAL_DBL_MAX))) {
                const double step = residual / exp(log_density);
                const double curvature = (a - 1.0) / x - (b - 1.0) / (1.0 - x);
                const double denominator = 1.0 - 0.5 * step * curvature;
                if ((denominator > 0.0) && isfinite(denominator)) {
                    candidate = x - step / denominator;
                }
            }

            if (!(candidate > lower && candidate < upper) || !isfinite(candidate)) {
                if (lower == 0.0) {
                    candidate = x / 2.0;
                    if (candidate == 0.0) {
                        x = 0.0;
                        break;
                    }
                } else if (upper == 1.0) {
                    candidate = x + (1.0 - x) / 2.0;
                } else {
                    candidate = lower + (upper - lower) / 2.0;
                }
            }
            if (candidate == x) {
                break;
            }
            x = candidate;
        }
    }
    return reflect ? 1.0 - x : x;
}
"""


_BETA_CDF_KERNEL = cp.ElementwiseKernel(
    "float64 a, float64 b, float64 x",
    "float64 result",
    "result = pal_ibeta(a, b, x);",
    "pal_gpu_beta_cdf",
    preamble=_BETA_PREAMBLE,
)

_BETA_INVERSE_KERNEL = cp.ElementwiseKernel(
    "float64 a, float64 b, float64 p",
    "float64 result",
    "result = pal_ibeta_inverse(a, b, p);",
    "pal_gpu_beta_inverse",
    preamble=_BETA_PREAMBLE,
)


def betainc(a: t.Any, b: t.Any, x: t.Any) -> t.Any:
    """Evaluate the regularized incomplete beta function on the GPU."""
    if not any(isinstance(value, cp.ndarray) for value in (a, b, x)):
        x = cp.asarray(x, dtype=cp.float64)
    return _BETA_CDF_KERNEL(a, b, x)


def betaincinv(a: t.Any, b: t.Any, p: t.Any) -> t.Any:
    """Evaluate the inverse regularized incomplete beta function on the GPU."""
    if not any(isinstance(value, cp.ndarray) for value in (a, b, p)):
        p = cp.asarray(p, dtype=cp.float64)
    return _BETA_INVERSE_KERNEL(a, b, p)


__all__ = ["betainc", "betaincinv"]
