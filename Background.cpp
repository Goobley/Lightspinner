#include "Background.hpp"
#include "Constants.hpp"
#include "JasPP.hpp"
#include <algorithm>
#include "Formal.hpp"

typedef Jasnah::Array1Own<i32> I32Arr;

template <typename T, typename U>
static inline int hunt(int len, T first, U val)
{
    auto last = first + len;
    auto it = std::upper_bound(first, last, val) - 1;
    return it - first;
}

static inline int hunt(F64View x, f64 val)
{
    return hunt(x.dim0, x.data, val);
}

inline void linear(F64View xTable, F64View yTable, F64View x, F64View y)
{
    const int Ntable = xTable.shape(0);
    const int N = x.shape(0);
    bool ascend = xTable(1) > xTable(0);
    const f64 xMin = If ascend Then xTable(0) Else xTable(Ntable-1) End;
    const f64 xMax = If ascend Then xTable(Ntable-1) Else xTable(0) End;

    for (int n = 0; n < N; ++n)
    {
        if (x(n) <= xMin)
            y(n) = If ascend Then yTable(0) Else yTable(Ntable-1) End;
        else if (x(n) >= xMax)
            y(n) = If ascend Then yTable(Ntable-1) Else yTable(0) End;
        else
        {
            int j = hunt(xTable, x(n));

            f64 fx = (xTable(j+1) - x(n)) / (xTable(j+1) - xTable(j));
            y(n) = fx * yTable(j) + (1 - fx) * yTable(j+1);
        }
    }
}

inline f64 linear(F64View xTable, F64View yTable, f64 x)
{
    const int Ntable = xTable.shape(0);
    bool ascend = xTable(1) > xTable(0);
    const f64 xMin = If ascend Then xTable(0) Else xTable(Ntable-1) End;
    const f64 xMax = If ascend Then xTable(Ntable-1) Else xTable(0) End;

    if (x <= xMin)
        return If ascend Then yTable(0) Else yTable(Ntable-1) End;

    if (x >= xMax)
        return If ascend Then yTable(Ntable-1) Else yTable(0) End;

    int j = hunt(xTable, x);

    f64 fx = (xTable(j+1) - x) / (xTable(j+1) - xTable(j));
    return fx * yTable(j) + (1 - fx) * yTable(j+1);
}

inline double Gaunt_bf(double lambda, double n_eff, int charge) {
  /* --- M. J. Seaton (1960), Rep. Prog. Phys. 23, 313 -- ----------- */

    namespace C = Constants;
    double x, x3, nsqx;

    x = (C::HC / (lambda * C::NM_TO_M)) / (C::ERydberg * square(charge));
    x3 = pow(x, 0.33333333);
    nsqx = 1.0 / (square(n_eff) * x);

    return 1.0 + 0.1728 * x3 * (1.0 - 2.0 * nsqx) -
            0.0496 * square(x3) * (1.0 - (1.0 - nsqx) * 0.66666667 * nsqx);
}

inline double Gaunt_ff(double lambda, int charge, double T) {
  /* --- M. J. Seaton (1960), Rep. Prog. Phys. 23, 313

   Note: There is a problem with this expansion at higher temperatures
         (T > 3.0E4 and longer wavelengths (lambda > 2000 nm). Set to
         1.0 when the value goes below 1.0 --          -------------- */

    namespace C = Constants;
    double x, x3, y, gIII;

    x = (C::HC / (lambda * C::NM_TO_M)) / (C::ERydberg * square(charge));
    x3 = pow(x, 0.33333333);
    y = (2.0 * lambda * C::NM_TO_M * C::KBoltzmann * T) / C::HC;

    gIII = 1.0 + 0.1728 * x3 * (1.0 + y) -
            0.0496 * square(x3) * (1.0 + (1.0 + y) * 0.33333333 * y);
    return (gIII > 1.0) ? gIII : 1.0;
}

struct SplineInterpolator
{
    bool ascend;
    F64Arr xTable;
    f64 xMin, xMax;
    F64Arr M;
    F64Arr u;
    F64Arr yTable;

    SplineInterpolator(F64View x, F64View y) : ascend(x(1) > x(0)), xTable(x),  M(x.shape(0)), u(x.shape(0)), yTable(y)
    {
        const int N = x.shape(0);
        xMin = If ascend Then x(0) Else x(N-1) End;
        xMax = If ascend Then x(N-1) Else x(0) End;

        f64 hj = x(1) - x(0);
        f64 D = (y(1) - y(0)) / hj;
        M(0) = u(0) = 0.0;
        for (int j = 1; j < N-1; ++j)
        {
            f64 hj1 = x(j+1) - x(j);
            f64 mu = hj / (hj + hj1);
            f64 D1 = (y(j+1) - y(j)) / hj1;
            f64 p = mu * M(j-1) + 2;
            M(j) = (mu - 1.0) / p;
            u(j) = ((D1 - D) * 6.0 / (hj + hj1) - mu * u(j-1)) / p;

            hj = hj1;
            D = D1;
        }

        M(N-1) = 0.0;
        for (int j = N - 1; j >= 0; --j)
        {
            M(j) = M(j) * M(j+1) + u(j);
        }
    }

    void eval(F64View x, F64View y)
    {
        const int N = xTable.shape(0);
        for (int n = 0; n < x.shape(0); ++n)
        {
            if (x(n) <= xMin)
                y(n) = If ascend Then yTable(0) Else yTable(N-1) End;
            else if (x(n) >= xMax)
                y(n) = If ascend Then yTable(N-1) Else yTable(0) End;
            else
            {
                int j = hunt(xTable, x(n));
                f64 hj = xTable(j+1) - xTable(j);
                f64 fx = (x(n) - xTable(j)) / hj;
                f64 fx1 = 1.0 - fx;

                y(n) = fx1 * yTable(j) + fx * yTable(j+1) 
                       + (fx1 * (square(fx1) - 1.0) * M(j) 
                          + fx * (square(fx) - 1.0) * M(j+1))
                       * square(hj) / 6.0;
            }
        }
    }

    f64 eval(f64 x)
    {
        const int N = xTable.shape(0);
        if (x <= xMin)
            return If ascend Then yTable(0) Else yTable(N-1) End;
        if (x >=  xMax)
            return If ascend Then yTable(N-1) Else yTable(0) End;

        int j = hunt(xTable, x);
        f64 hj = xTable(j+1) - xTable(j);
        f64 fx = (x - xTable(j)) / hj;
        f64 fx1 = 1.0 - fx;

        return fx1 * yTable(j) + fx * yTable(j+1) 
               + (fx1 * (square(fx1) - 1.0) * M(j) 
                  + fx * (square(fx) - 1.0) * M(j+1))
               * square(hj) / 6.0;
    }
};


inline f64 bilinear(int Ncol, int Nrow, const f64 *f, f64 x, f64 y) {
  int i, j, i1, j1;
  double fx, fy;

  /* --- Bilinear interpolation of the function f on the fractional
         indices x and y --                            -------------- */

  i = (int)x;
  fx = x - i;
  if (i == Ncol - 1)
    i1 = i;
  else
    i1 = i + 1;
  j = (int)y;
  fy = y - j;
  if (j == Nrow - 1)
    j1 = j;
  else
    j1 = j + 1;

  return (1.0 - fx) * (1.0 - fy) * f[j * Ncol + i] +
         fx * (1.0 - fy) * f[j * Ncol + i1] +
         (1.0 - fx) * fy * f[j1 * Ncol + i] + fx * fy * f[j1 * Ncol + i1];
}

void thomson_scattering(const Atmosphere& atmos, F64View chi)
{
    namespace C =  Constants;
    const double sigma = 8.0 * C::Pi / 3.0 *
                pow(C::QElectron/ (sqrt(4.0 * C::Pi * C::Epsilon0) *
                                (sqrt(C::MElectron) * C::CLight)),
                    4);
    for (int k = 0; k < atmos.Nspace; ++k)
        chi(k) = atmos.ne(k) * sigma;
}

// bool hydrogen_bf(const Atmosphere& atmos, F64View chi, F64View eta)
// {
//     // NOTE(cmo): Unlike Han's version, this doesn't check if Hydrogen is active. Onus on user.
//     // Actually, can this not be done in the same way as metal_bf?
//     // namespace C = Constants;
//     // for (int k = 0; k < atmos.Nspace; ++k)
//     // {
//     //     chi(k) = 0.0;
//     //     eta(k) = 0.0;
//     // }

//     // constexpr f64 twohc = 2.0 * C::HC / cube(C::NM_TO_M);
// }

bool hydrogen_ff(const Atmosphere& atmos, f64 lambda, F64View2D hPops, F64View chi)
{
    /* --- Hydrogen free-free opacity

        See: Mihalas (1978) p. 101
            --                                             -------------- */
    namespace C = Constants;
    const f64 C0 = square(C::QElectron) / (4.0 * C::Pi * C::Epsilon0) / sqrt(C::MElectron);
    const f64 sigma = 4.0 / 3.0 * sqrt(2.0 * C::Pi / (3.0 * C::KBoltzmann)) * cube(C0) / C::HC;
    const f64 nu3 = cube(lambda * C::NM_TO_M / C::CLight);
    const f64 hc_kla = (C::HC) / (C::KBoltzmann * C::NM_TO_M * lambda);

    auto np = hPops(hPops.shape(0)-1);
    for (int k = 0; k < atmos.Nspace; ++k)
    {
        f64 stim = 1.0 * exp(-hc_kla / atmos.temperature(k));
        f64 gff = Gaunt_ff(lambda, 1, atmos.temperature(k));
        chi(k) = sigma / sqrt(atmos.temperature(k)) * nu3 * atmos.ne(k) * np(k) * stim * gff;
    }
    return true;
}

struct H2Opacity
{
    static constexpr int NFF_H2 = 19;
    static constexpr int NTHETA_H2 = 8;
    static constexpr double lambdaFFMinus[NFF_H2] = {0.0,    350.5,  414.2,   506.3,  569.6,
                                                650.9,  759.4,  911.3,   1139.1, 1518.8,
                                                1822.6, 2278.3, 3037.7,  3645.2, 4556.5,
                                                6075.3, 9113.0, 11391.3, 15188.3};

    static constexpr double thetaFFMinus[NTHETA_H2] = {0.5, 0.8, 1.0, 1.2, 1.6, 2.0, 2.8, 3.6};

    static constexpr double kappaFFMinus[NFF_H2 * NTHETA_H2] = {
        /* --- lambda =     0.0 [nm] --                    -------------- */
        0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
        0.00e+00,
        /* --- lambda =   350.5 [nm] --                    -------------- */
        4.17e-02, 6.10e-02, 7.34e-02, 8.59e-02, 1.11e-01, 1.37e-01, 1.87e-01,
        2.40e-01,
        /* --- lambda =   414.2 [nm] --                    -------------- */
        5.84e-02, 8.43e-02, 1.01e-01, 1.17e-01, 1.49e-01, 1.82e-01, 2.49e-01,
        3.16e-01,
        /* --- lambda =   506.3 [nm] --                    -------------- */
        8.70e-02, 1.24e-01, 1.46e-01, 1.67e-01, 2.10e-01, 2.53e-01, 3.39e-01,
        4.27e-01,
        /* --- lambda =   569.6 [nm] --                    -------------- */
        1.10e-01, 1.54e-01, 1.80e-01, 2.06e-01, 2.55e-01, 3.05e-01, 4.06e-01,
        5.07e-01,
        /* --- lambda =   650.9 [nm] --                    -------------- */
        1.43e-01, 1.98e-01, 2.30e-01, 2.59e-01, 3.17e-01, 3.75e-01, 4.92e-01,
        6.09e-01,
        /* --- lambda =   759.4 [nm] --                    -------------- */
        1.92e-01, 2.64e-01, 3.03e-01, 3.39e-01, 4.08e-01, 4.76e-01, 6.13e-01,
        7.51e-01,
        /* --- lambda =   911.3 [nm] --                    -------------- */
        2.73e-01, 3.71e-01, 4.22e-01, 4.67e-01, 5.52e-01, 6.33e-01, 7.97e-01,
        9.63e-01,
        /* --- lambda =  1139.1 [nm] --                    -------------- */
        4.20e-01, 5.64e-01, 6.35e-01, 6.97e-01, 8.06e-01, 9.09e-01, 1.11e+00,
        1.32e+00,
        /* --- lambda =  1518.8 [nm] --                    -------------- */
        7.36e-01, 9.75e-01, 1.09e+00, 1.18e+00, 1.34e+00, 1.48e+00, 1.74e+00,
        2.01e+00,
        /* --- lambda =  1822.6 [nm] --                    -------------- */
        1.05e+00, 1.39e+00, 1.54e+00, 1.66e+00, 1.87e+00, 2.04e+00, 2.36e+00,
        2.68e+00,
        /* --- lambda =  2278.3 [nm] --                    -------------- */
        1.63e+00, 2.14e+00, 2.36e+00, 2.55e+00, 2.84e+00, 3.07e+00, 3.49e+00,
        3.90e+00,
        /* --- lambda =  3037.7 [nm] --                    -------------- */
        2.89e+00, 3.76e+00, 4.14e+00, 4.44e+00, 4.91e+00, 5.28e+00, 5.90e+00,
        6.44e+00,
        /* --- lambda =  3645.2 [nm] --                    -------------- */
        4.15e+00, 5.38e+00, 5.92e+00, 6.35e+00, 6.99e+00, 7.50e+00, 8.32e+00,
        9.02e+00,
        /* --- lambda =  4556.5 [nm] --                    -------------- */
        6.47e+00, 8.37e+00, 9.20e+00, 9.84e+00, 1.08e+01, 1.16e+01, 1.28e+01,
        1.38e+01,
        /* --- lambda =  6075.3 [nm] --                    -------------- */
        1.15e+01, 1.48e+01, 1.63e+01, 1.74e+01, 1.91e+01, 2.04e+01, 2.24e+01,
        2.40e+01,
        /* --- lambda =  9113.0 [nm] --                    -------------- */
        2.58e+01, 3.33e+01, 3.65e+01, 3.90e+01, 4.27e+01, 4.54e+01, 4.98e+01,
        5.33e+01,
        /* --- lambda = 11391.3 [nm] --                    -------------- */
        4.03e+01, 5.20e+01, 5.70e+01, 6.08e+01, 6.65e+01, 7.08e+01, 7.76e+01,
        8.30e+01,
        /* --- lambda = 15188.3 [nm] --                    -------------- */
        7.16e+01, 9.23e+01, 1.01e+02, 1.08e+02, 1.18e+02, 1.26e+02, 1.38e+02,
        1.47e+02};


    static constexpr int NFF_H2P = 15;
    static constexpr int NTEMP_H2P = 10;
    static constexpr double lambdaFFPlus[NFF_H2P] = {0.0,    384.6,  555.6,  833.3,  1111.1,
                                1428.6, 1666.7, 2000.0, 2500.0, 2857.1,
                                3333.3, 4000.0, 5000.0, 6666.7, 10000.0};

    static constexpr double tempFFPlus[NTEMP_H2P] = {2.5E+03, 3.0E+03, 3.5E+03, 4.0E+03,
                                        5.0E+03, 6.0E+03, 7.0E+03, 8.0E+03,
                                        1.0E+04, 1.2E+04};

    static constexpr double kappaFFPlus[NFF_H2P * NTEMP_H2P] = {
        /* --- lambda =      0.0 [nm] --                   -------------- */
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        /* --- lambda =    384.6 [nm] --                   -------------- */
        0.46, 0.46, 0.42, 0.39, 0.36, 0.33, 0.32, 0.30, 0.27, 0.25,
        /* --- lambda =    555.6 [nm] --                   -------------- */
        0.70, 0.62, 0.59, 0.56, 0.51, 0.43, 0.41, 0.39, 0.35, 0.34,
        /* --- lambda =    833.3 [nm] --                   -------------- */
        0.92, 0.86, 0.80, 0.76, 0.70, 0.64, 0.59, 0.55, 0.48, 0.43,
        /* --- lambda =   1111.1 [nm] --                   -------------- */
        1.11, 1.04, 0.96, 0.91, 0.82, 0.74, 0.68, 0.62, 0.53, 0.46,
        /* --- lambda =   1428.6 [nm] --                   -------------- */
        1.26, 1.19, 1.09, 1.02, 0.90, 0.80, 0.72, 0.66, 0.55, 0.48,
        /* --- lambda =   1666.7 [nm] --                   -------------- */
        1.37, 1.25, 1.15, 1.07, 0.93, 0.83, 0.74, 0.67, 0.56, 0.49,
        /* --- lambda =   2000.0 [nm] --                   -------------- */
        1.44, 1.32, 1.21, 1.12, 0.97, 0.84, 0.75, 0.67, 0.56, 0.48,
        /* --- lambda =   2500.0 [nm] --                   -------------- */
        1.54, 1.39, 1.26, 1.15, 0.98, 0.85, 0.75, 0.67, 0.55, 0.46,
        /* --- lambda =   2857.1 [nm] --                   -------------- */
        1.58, 1.42, 1.27, 1.16, 0.98, 0.84, 0.74, 0.66, 0.54, 0.45,
        /* --- lambda =   3333.3 [nm] --                   -------------- */
        1.62, 1.43, 1.28, 1.15, 0.97, 0.83, 0.72, 0.64, 0.52, 0.44,
        /* --- lambda =   4000.0 [nm] --                   -------------- */
        1.63, 1.43, 1.27, 1.14, 0.95, 0.80, 0.70, 0.62, 0.50, 0.42,
        /* --- lambda =   5000.0 [nm] --                   -------------- */
        1.62, 1.40, 1.23, 1.10, 0.90, 0.77, 0.66, 0.59, 0.48, 0.39,
        /* --- lambda =   6666.7 [nm] --                   -------------- */
        1.55, 1.33, 1.16, 1.03, 0.84, 0.71, 0.60, 0.53, 0.43, 0.36,
        /* --- lambda =  10000.0 [nm] --                   -------------- */
        1.39, 1.18, 1.02, 0.90, 0.73, 0.60, 0.52, 0.46, 0.37, 0.31};

    static constexpr int N_RAYLEIGH_H2 = 21;
    static constexpr f64 RAYLEIGH_H2_LIMIT = 121.57;
    static constexpr double a[3] = {8.779E+01, 1.323E+06, 2.245E+10};
    static constexpr double lambdaRH2[N_RAYLEIGH_H2] = {
        121.57, 130.00, 140.00, 150.00, 160.00, 170.00, 185.46,
        186.27, 193.58, 199.05, 230.29, 237.91, 253.56, 275.36,
        296.81, 334.24, 404.77, 407.90, 435.96, 546.23, 632.80};

    static constexpr double sigma[N_RAYLEIGH_H2] = {
        2.35E-06, 1.22E-06, 6.80E-07, 4.24E-07, 2.84E-07, 2.00E-07, 1.25E-07,
        1.22E-07, 1.00E-07, 8.70E-08, 4.29E-08, 3.68E-08, 2.75E-08, 1.89E-08,
        1.36E-08, 8.11E-09, 3.60E-09, 3.48E-09, 2.64E-09, 1.04E-09, 5.69E-10};

    bool computeMinus;
    bool computePlus;
    const Atmosphere& atmos;
    F64Arr thetaIndex;
    F64Arr tempIndex;
    F64View nH2;
    F64View2D nH;


    H2Opacity(const Atmosphere& a, F64View H2, F64View2D H)
        : computeMinus(a.Nspace == H2.shape(0)), 
          computePlus(a.Nspace == H.shape(1)), 
          atmos(a), 
          thetaIndex(a.Nspace), 
          tempIndex(a.Nspace), 
          nH2(H2), 
          nH(H)
    {
        namespace C = Constants;
        for (int k = 0; k < atmos.Nspace; ++k)
        {
            f64 theta = C::Theta0 / atmos.temperature(k);
            if (theta <= thetaFFMinus[0])
                thetaIndex(k) = 0.0;
            else if (theta >= thetaFFMinus[NTHETA_H2-1])
                thetaIndex(k) = NTHETA_H2 - 1;
            else
            {
                int index = hunt(NTHETA_H2, thetaFFMinus, theta);
                thetaIndex(k) = (f64)index + (theta - thetaFFMinus[index]) / (thetaFFMinus[index+1] - thetaFFMinus[index]);
            }

            f64 temp = atmos.temperature(k);
            if (temp <= tempFFPlus[0])
                tempIndex(k) = 0.0;
            else if (temp >= tempFFPlus[NTEMP_H2P-1])
                tempIndex(k) = NTEMP_H2P - 1;
            else
            {
                int index = hunt(NTEMP_H2P, tempFFPlus, temp);
                tempIndex(k) = (f64)index + (temp - tempFFPlus[index]) / (tempFFPlus[index+1] - tempFFPlus[index]);
            }
        }
    }

    bool h2minus_ff(f64 lambda, F64View chi)
    {
        /* --- H2-minus Free-Free absorption coefficients (in units of
            10E-29 m^5/J). Stimulated emission is included.

            From: Bell, K. L., (1980) J. Phys. B13, 1859.
            Also: R. Mathisen (1984), Master's thesis, Inst. Theor.
                    Astroph., University of Oslo, p. 18 */

        namespace C = Constants;
        if (!computeMinus || lambda >= lambdaFFMinus[NFF_H2 - 1])
            return false;

        int index = hunt(NFF_H2, lambdaFFMinus, lambda);
        f64 lambdaIndex = (f64)index + (lambda - lambdaFFMinus[index]) / (lambdaFFMinus[index+1] - lambdaFFMinus[index]);

        for (int k = 0; k < atmos.Nspace; ++k)
        {
            if (nH2(k) > 0.0)
            {
                f64 pe = atmos.ne(k) * C::KBoltzmann * atmos.temperature(k);
                f64 kappa = bilinear(NTHETA_H2, NFF_H2, kappaFFMinus, thetaIndex(k), lambdaIndex);
                chi(k) = (nH2(k) * 1.0e-29) * pe * kappa;
            }
            else
                chi(k) = 0.0;
        }
        return true;
    }

    bool h2plus_ff(f64 lambda, F64View chi)
    {
        /* --- H2+ Free-Free scattering coefficients in units of
            1.0E-49 m^-1 / (H atom/m^3) / (proton/M^3). Stimulated emission
            is included. This represents the following interaction:

            H + H^+ + \nu ---> H + H^+

            From: D. R. Bates (1952), MNRAS 112, 40-44
            Also: R. Mathisen (1984), Master's thesis, Inst. Theor.
                    Astroph., University of Oslo, p. 45 */

        if (!computePlus || lambda >= lambdaFFPlus[NFF_H2P-1])
            return false;

        int index = hunt(NFF_H2P, lambdaFFPlus, lambda);
        f64 lambdaIndex = (f64)index + (lambda - lambdaFFPlus[index]) / (lambdaFFPlus[index+1] - lambdaFFPlus[index]);

        auto np = nH(nH.shape(0)-1);
        for (int k = 0; k < atmos.Nspace; ++k)
        {
            f64 kappa = bilinear(NTEMP_H2P, NFF_H2P, kappaFFPlus, tempIndex(k), lambdaIndex);
            chi(k) = (nH(0,k) * 1.0e-29) * (np(k) * 1.0e-20) * kappa;
        }
        return true;
    }

    bool rayleigh_H2(f64 lambda, F64View scatt)
    {
      /* --- Rayleigh scattering by H2 molecules. Cross-section is given
         in in units of Mb, 1.0E-22 m^2.

        See: G. A. Victor and A. Dalgarno (1969), J. Chem. Phys. 50, 2535
            (for lambda <= 632.80 nm), and
            S. P. Tarafdar and M. S. Vardya (1973), MNRAS 163, 261
        Also: R. Mathisen (1984), Master's thesis, Inst. Theor.
            Astroph., University of Oslo, p. 49
         --                                            -------------- */
        if (!computePlus || lambda < RAYLEIGH_H2_LIMIT)
            return false;

        namespace C = Constants;

        f64 sigmaRH2;
        if (lambda <= lambdaRH2[N_RAYLEIGH_H2-1])
            sigmaRH2 = linear(F64View(const_cast<f64*>(lambdaRH2), N_RAYLEIGH_H2), 
                              F64View(const_cast<f64*>(sigma), N_RAYLEIGH_H2),
                              lambda);
        else
        {
            f64 lambda2 = 1.0 / square(lambda);
            sigmaRH2 = (a[0] + (a[1] + a[2] * lambda2) * lambda2) * square(lambda2);
        }
        sigmaRH2 *= C::MEGABARN_TO_M2;

        for (int k = 0; k < atmos.Nspace; ++k)
            scatt(k) = sigmaRH2 * nH2(k);
        return true;
    }
};

struct HMinusOpacity
{
    static constexpr int NBF = 34;
    static constexpr double lambdaBF[NBF] = {
        0.0,    50.0,   100.0,  150.0,  200.0,  250.0,  300.0,  350.0,  400.0,
        450.0,  500.0,  550.0,  600.0,  650.0,  700.0,  750.0,  800.0,  850.0,
        900.0,  950.0,  1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0,
        1350.0, 1400.0, 1450.0, 1500.0, 1550.0, 1600.0, 1641.9};

    static constexpr double alphaBF[NBF] = {
        0.0,  0.15, 0.33, 0.57, 0.85, 1.17, 1.52, 1.89, 2.23, 2.55, 2.84, 3.11,
        3.35, 3.56, 3.71, 3.83, 3.92, 3.95, 3.93, 3.85, 3.73, 3.58, 3.38, 3.14,
        2.85, 2.54, 2.20, 1.83, 1.46, 1.06, 0.71, 0.40, 0.17, 0.0};



    static constexpr int NFF = 17;
    static constexpr int NTHETA = 16;
    static constexpr double lambdaFF[NFF] = {0.0,    303.8,  455.6,  506.3,  569.5,  650.9,
                                             759.4,  911.3,  1013.0, 1139.0, 1302.0, 1519.0,
                                             1823.0, 2278.0, 3038.0, 4556.0, 9113.0};
        /* --- theta = 5040.0/T --                           -------------- */
    static constexpr double thetaFF[NTHETA] = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
                                               1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0};
    static constexpr double kappaFF[NFF * NTHETA] = {
        /* --- lambda =    0.0 [nm] --                     -------------- */
        0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
        0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
        0.00e+00, 0.00e+00,
        /* --- lambda =  303.8 [nm] --                     -------------- */
        3.44e-02, 4.18e-02, 4.91e-02, 5.65e-02, 6.39e-02, 7.13e-02, 7.87e-02,
        8.62e-02, 9.36e-02, 1.01e-01, 1.08e-01, 1.16e-01, 1.23e-01, 1.30e-01,
        1.38e-01, 1.45e-01,
        /* --- lambda =  455.6 [nm] --                     -------------- */
        7.80e-02, 9.41e-02, 1.10e-01, 1.25e-01, 1.40e-01, 1.56e-01, 1.71e-01,
        1.86e-01, 2.01e-01, 2.16e-01, 2.31e-01, 2.45e-01, 2.60e-01, 2.75e-01,
        2.89e-01, 3.03e-01,
        /* --- lambda =  506.3 [nm] --                     -------------- */
        9.59e-02, 1.16e-01, 1.35e-01, 1.53e-01, 1.72e-01, 1.90e-01, 2.08e-01,
        2.25e-01, 2.43e-01, 2.61e-01, 2.78e-01, 2.96e-01, 3.13e-01, 3.30e-01,
        3.47e-01, 3.64e-01,
        /* --- lambda =  569.5 [nm] --                     -------------- */
        1.21e-01, 1.45e-01, 1.69e-01, 1.92e-01, 2.14e-01, 2.36e-01, 2.58e-01,
        2.80e-01, 3.01e-01, 3.22e-01, 3.43e-01, 3.64e-01, 3.85e-01, 4.06e-01,
        4.26e-01, 4.46e-01,
        /* --- lambda =  650.9 [nm] --                     -------------- */
        1.56e-01, 1.88e-01, 2.18e-01, 2.47e-01, 2.76e-01, 3.03e-01, 3.31e-01,
        3.57e-01, 3.84e-01, 4.10e-01, 4.36e-01, 4.62e-01, 4.87e-01, 5.12e-01,
        5.37e-01, 5.62e-01,
        /* --- lambda =  759.4 [nm] --                     -------------- */
        2.10e-01, 2.53e-01, 2.93e-01, 3.32e-01, 3.69e-01, 4.06e-01, 4.41e-01,
        4.75e-01, 5.09e-01, 5.43e-01, 5.76e-01, 6.08e-01, 6.40e-01, 6.72e-01,
        7.03e-01, 7.34e-01,
        /* --- lambda =  911.3 [nm] --                     -------------- */
        2.98e-01, 3.59e-01, 4.16e-01, 4.70e-01, 5.22e-01, 5.73e-01, 6.21e-01,
        6.68e-01, 7.15e-01, 7.60e-01, 8.04e-01, 8.47e-01, 8.90e-01, 9.32e-01,
        9.73e-01, 1.01e+00,
        /* --- lambda = 1013.0 [nm] --                     -------------- */
        3.65e-01, 4.39e-01, 5.09e-01, 5.75e-01, 6.39e-01, 7.00e-01, 7.58e-01,
        8.15e-01, 8.71e-01, 9.25e-01, 9.77e-01, 1.03e+00, 1.08e+00, 1.13e+00,
        1.18e+00, 1.23e+00,
        /* --- lambda = 1139.0 [nm] --                     -------------- */
        4.58e-01, 5.50e-01, 6.37e-01, 7.21e-01, 8.00e-01, 8.76e-01, 9.49e-01,
        1.02e+00, 1.09e+00, 1.15e+00, 1.22e+00, 1.28e+00, 1.34e+00, 1.40e+00,
        1.46e+00, 1.52e+00,
        /* --- lambda = 1302.0 [nm] --                     -------------- */
        5.92e-01, 7.11e-01, 8.24e-01, 9.31e-01, 1.03e+00, 1.13e+00, 1.23e+00,
        1.32e+00, 1.40e+00, 1.49e+00, 1.57e+00, 1.65e+00, 1.73e+00, 1.80e+00,
        1.88e+00, 1.95e+00,
        /* --- lambda = 1519.0 [nm] --                     -------------- */
        7.98e-01, 9.58e-01, 1.11e+00, 1.25e+00, 1.39e+00, 1.52e+00, 1.65e+00,
        1.77e+00, 1.89e+00, 2.00e+00, 2.11e+00, 2.21e+00, 2.32e+00, 2.42e+00,
        2.51e+00, 2.61e+00,
        /* --- lambda = 1823.0 [nm] --                     -------------- */
        1.14e+00, 1.36e+00, 1.58e+00, 1.78e+00, 1.98e+00, 2.17e+00, 2.34e+00,
        2.52e+00, 2.68e+00, 2.84e+00, 3.00e+00, 3.15e+00, 3.29e+00, 3.43e+00,
        3.57e+00, 3.70e+00,
        /* --- lambda = 2278.0 [nm] --                     -------------- */
        1.77e+00, 2.11e+00, 2.44e+00, 2.75e+00, 3.05e+00, 3.34e+00, 3.62e+00,
        3.89e+00, 4.14e+00, 4.39e+00, 4.63e+00, 4.86e+00, 5.08e+00, 5.30e+00,
        5.51e+00, 5.71e+00,
        /* --- lambda = 3038.0 [nm] --                     -------------- */
        3.10e+00, 3.71e+00, 4.29e+00, 4.84e+00, 5.37e+00, 5.87e+00, 6.36e+00,
        6.83e+00, 7.28e+00, 7.72e+00, 8.14e+00, 8.55e+00, 8.95e+00, 9.33e+00,
        9.71e+00, 1.01e+01,
        /* --- lambda = 4556.0 [nm] --                     -------------- */
        6.92e+00, 8.27e+00, 9.56e+00, 1.08e+01, 1.19e+01, 1.31e+01, 1.42e+01,
        1.52e+01, 1.62e+01, 1.72e+01, 1.82e+01, 1.91e+01, 2.00e+01, 2.09e+01,
        2.17e+01, 2.25e+01,
        /* --- lambda = 9113.0 [nm] --                     -------------- */
        2.75e+01, 3.29e+01, 3.80e+01, 4.28e+01, 4.75e+01, 5.19e+01, 5.62e+01,
        6.04e+01, 6.45e+01, 6.84e+01, 7.23e+01, 7.60e+01, 7.97e+01, 8.32e+01,
        8.67e+01, 9.01e+01};

    static constexpr int NJOHN = 6;
    static constexpr double A[NJOHN] = {0.000,    2483.346, -3449.889,
                                        2200.040, -696.271, 88.283};
    static constexpr double B[NJOHN] = {0.000,    285.827,   -1158.382,
                                        2427.719, -1841.400, 444.517};
    static constexpr double C[NJOHN] = {0.000,      -2054.291, 8746.523,
                                        -13651.105, 8624.970,  -1863.864};
    static constexpr double D[NJOHN] = {0.000,     2827.776,   -11485.632,
                                        16755.524, -10051.530, 2095.288};
    static constexpr double E[NJOHN] = {0.000,     -1341.537, 5303.609,
                                        -7510.494, 4400.067,  -901.788};
    static constexpr double F[NJOHN] = {0.000,    208.952,  -812.939,
                                1132.738, -655.020, 132.985};

        bool compute;
        F64Arr thetaIndex;
        const Atmosphere& atmos;
        F64View hMinus;
        F64View2D h;
        SplineInterpolator bfInterp;

        HMinusOpacity(const Atmosphere& a, F64View hMinPops, F64View2D hPops)
            : compute(a.Nspace == hMinPops.shape(0)),
              atmos(a),
              hMinus(hMinPops),
              h(hPops),
                               // SUE ME
              bfInterp(F64View(const_cast<double*>(lambdaBF), NBF), F64View(const_cast<double*>(alphaBF), NBF))
        {
            namespace C = Constants;
            const int Nspace = atmos.Nspace;
            thetaIndex = F64Arr(Nspace);

            for (int k = 0; k < Nspace; ++k)
            {
                f64 theta = C::Theta0 / atmos.temperature(k);
                if (theta <= thetaFF[0])
                    thetaIndex(k) = 0.0;
                else if (theta >= thetaFF[NTHETA-1])
                    thetaIndex(k) = NTHETA - 1;
                else
                {
                    int index = hunt(NTHETA, thetaFF, theta);
                    thetaIndex(k) = (f64)index + (theta - thetaFF[index]) / (thetaFF[index+1] - thetaFF[index]);
                }
            }
        }

        bool hminus_bf(f64 lambda, F64View chi, F64View eta)
        {
            /* --- H-minus Bound-Free coefficients (in units of 1.0E-21 m^2).

            From: S. Geltman (1962), ApJ 136, 935-945
            Also: Mihalas (1978), p. 102 --                     -------------- */
            namespace C = Constants;
            if (!compute)
                return false;
            if ((lambda <= lambdaBF[0]) || (lambda >= lambdaBF[NBF-1]))
                return false;

            f64 alpha = bfInterp.eval(lambda);
            alpha *= 1e-21;

            const f64 hc_kla = C::HC / (C::KBoltzmann * C::NM_TO_M * lambda);
            const f64 twohnu3_c2 = (2.0 * C::HC) / cube(C::NM_TO_M * lambda);

            for (int k = 0; k < atmos.Nspace; ++k)
            {
                f64 stimEmis = exp(-hc_kla / atmos.temperature(k));
                chi(k) = hMinus(k) * (1.0 - stimEmis) * alpha;
                eta(k) = hMinus(k) * twohnu3_c2 * stimEmis * alpha;
            }
            return true;
        }

        bool hminus_ff(f64 lambda, F64View chi)
        {
            /* --- H-minus Free-Free coefficients (in units of 1.0E-29 m^5/J)

            From: J. L. Stilley and J. Callaway (1970), ApJ 160, 245-260
            Also: D. Mihalas (1978), p. 102
            R. Mathisen (1984), Master's thesis, Inst. Theor.
              Astroph., University of Oslo. p. 17 */

            if (!compute)
                return false;

            if (lambda >= lambdaFF[NFF-1])
                return hminus_ff_long(lambda, chi);

            namespace C = Constants;
            int index = hunt(NFF, lambdaFF, lambda);
            f64 lambdaIndex = (f64)index + (lambda - lambdaFF[index]) / (lambdaFF[index + 1] - lambdaFF[index]);

            for (int k = 0; k < atmos.Nspace; ++k)
            {
                f64 pe = atmos.ne(k) * C::KBoltzmann * atmos.temperature(k);
                f64 kappa = bilinear(NTHETA, NFF, kappaFF, thetaIndex(k), lambdaIndex);
                chi(k) = (h(0, k) * 1.0e-29) * pe * kappa;
            }
            return true;
        }

        bool hminus_ff_long(f64 lambda, F64View chi)
        {
            /* --- H-minus Free-Free opacity. Parametrization for long wavelengths
            as given by T. L. John (1988), A&A 193, 189-192 (see table 3a).
            His results are based on calculations by K. L. Bell and
            K. A. Berrington (1987), J. Phys. B 20, 801-806. -- -------- */

            namespace Cst = Constants;
            f64 Clambda[NJOHN];

            constexpr f64 Ck = Cst::KBoltzmann * Cst::Theta0 * 1.0e-32;
            f64 lambdaMu = lambda / Cst::MICRON_TO_NM;
            f64 lambdaInv = 1.0 / lambdaMu;
            for (int n = 1; n < NJOHN; ++n)
            {
                Clambda[n] = square(lambdaMu) * A[n] + B[n] + lambdaInv
                                * (C[n] + lambdaInv * (D[n] + lambdaInv * (E[n] + lambdaInv * F[n])));
            }

            for (int k = 0; k < atmos.Nspace; ++k)
            {
                chi(k) = 0.0;
                f64 thetaN = 1.0;
                f64 sqrtTheta = sqrt(Cst::Theta0 / atmos.temperature(k));
                for (int n = 1; n < NJOHN; ++n)
                {
                    thetaN *= sqrtTheta;
                    chi(k) += thetaN * Clambda[n];
                }
                chi(k) *= h(0,k) * (atmos.ne(k) * Ck);
            }
            return true;
        }
};


#define NTOH 15
#define NEOH 130

bool OH_bf_opac(const Atmosphere& atmos, f64 lambda, F64View OH, F64View chi, F64View eta) {
  int index = 0;
  int index2 = 0;
  double Eev, e_index, t_index, hc_kla, twohnu3_c2, kappa, stimEmis;

  static constexpr double TOH[NTOH] = {2000.00, 2500.00, 3000.00, 3500.00, 4000.00,
                             4500.00, 5000.00, 5500.00, 6000.00, 6500.00,
                             7000.00, 7500.00, 8000.00, 8500.00, 9000.00};

  static constexpr double EOH[NEOH] = {
      2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0,  3.1,  3.2,
      3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4.0,  4.1,  4.2,  4.3,  4.4,
      4.5,  4.6,  4.7,  4.8,  4.9,  5.0,  5.1,  5.2,  5.3,  5.4,  5.5,  5.6,
      5.7,  5.8,  5.9,  6.0,  6.1,  6.2,  6.3,  6.4,  6.5,  6.6,  6.7,  6.8,
      6.9,  7.0,  7.1,  7.2,  7.3,  7.4,  7.5,  7.6,  7.7,  7.8,  7.9,  8.0,
      8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,  8.8,  8.9,  9.0,  9.1,  9.2,
      9.3,  9.4,  9.5,  9.6,  9.7,  9.8,  9.9,  10.0, 10.1, 10.2, 10.3, 10.4,
      10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6,
      11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8,
      12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0,
      14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0};

  static constexpr double OH_cross[NEOH][NTOH] = {
      {-30.855, -29.121, -27.976, -27.166, -26.566, -26.106, -25.742, -25.448,
       -25.207, -25.006, -24.836, -24.691, -24.566, -24.457, -24.363},
      {-30.494, -28.760, -27.615, -26.806, -26.206, -25.745, -25.381, -25.088,
       -24.846, -24.645, -24.475, -24.330, -24.205, -24.097, -24.002},
      {-30.157, -28.425, -27.280, -26.472, -25.872, -25.411, -25.048, -24.754,
       -24.513, -24.312, -24.142, -23.997, -23.872, -23.764, -23.669},
      {-29.848, -28.117, -26.974, -26.165, -25.566, -25.105, -24.742, -24.448,
       -24.207, -24.006, -23.836, -23.692, -23.567, -23.458, -23.364},
      {-29.567, -27.837, -26.693, -25.885, -25.286, -24.826, -24.462, -24.169,
       -23.928, -23.727, -23.557, -23.412, -23.287, -23.179, -23.084},
      {-29.307, -27.578, -26.436, -25.628, -25.029, -24.569, -24.205, -23.912,
       -23.671, -23.470, -23.300, -23.155, -23.031, -22.922, -22.828},
      {-29.068, -27.341, -26.199, -25.391, -24.792, -24.332, -23.969, -23.676,
       -23.435, -23.234, -23.064, -22.920, -22.795, -22.687, -22.592},
      {-28.820, -27.115, -25.978, -25.172, -24.574, -24.115, -23.752, -23.459,
       -23.218, -23.017, -22.848, -22.703, -22.579, -22.470, -22.376},
      {-28.540, -26.891, -25.768, -24.968, -24.372, -23.914, -23.552, -23.259,
       -23.019, -22.818, -22.649, -22.504, -22.380, -22.272, -22.177},
      {-28.275, -26.681, -25.574, -24.779, -24.186, -23.729, -23.368, -23.076,
       -22.836, -22.636, -22.467, -22.322, -22.198, -22.090, -21.996},
      {-27.993, -26.470, -25.388, -24.602, -24.014, -23.560, -23.200, -22.909,
       -22.669, -22.470, -22.301, -22.157, -22.033, -21.925, -21.831},
      {-27.698, -26.252, -25.204, -24.433, -23.851, -23.401, -23.043, -22.754,
       -22.515, -22.316, -22.148, -22.005, -21.881, -21.773, -21.679},
      {-27.398, -26.026, -25.019, -24.267, -23.696, -23.251, -22.896, -22.609,
       -22.372, -22.174, -22.007, -21.864, -21.741, -21.634, -21.540},
      {-27.100, -25.791, -24.828, -24.102, -23.543, -23.106, -22.756, -22.472,
       -22.238, -22.041, -21.875, -21.733, -21.611, -21.504, -21.411},
      {-26.807, -25.549, -24.631, -23.933, -23.391, -22.964, -22.621, -22.341,
       -22.109, -21.915, -21.751, -21.610, -21.488, -21.383, -21.290},
      {-26.531, -25.310, -24.431, -23.761, -23.238, -22.823, -22.488, -22.214,
       -21.986, -21.795, -21.633, -21.494, -21.374, -21.269, -21.178},
      {-26.239, -25.066, -24.225, -23.585, -23.082, -22.681, -22.356, -22.089,
       -21.866, -21.679, -21.520, -21.383, -21.265, -21.162, -21.072},
      {-25.945, -24.824, -24.017, -23.405, -22.923, -22.538, -22.223, -21.964,
       -21.748, -21.565, -21.410, -21.276, -21.160, -21.059, -20.970},
      {-25.663, -24.587, -23.810, -23.222, -22.761, -22.391, -22.088, -21.838,
       -21.629, -21.452, -21.300, -21.170, -21.057, -20.958, -20.872},
      {-25.372, -24.350, -23.603, -23.038, -22.596, -22.241, -21.950, -21.710,
       -21.508, -21.337, -21.190, -21.064, -20.954, -20.858, -20.774},
      {-25.076, -24.111, -23.396, -22.853, -22.429, -22.088, -21.809, -21.578,
       -21.384, -21.220, -21.078, -20.957, -20.851, -20.758, -20.676},
      {-24.779, -23.870, -23.189, -22.669, -22.261, -21.934, -21.667, -21.445,
       -21.259, -21.101, -20.965, -20.848, -20.746, -20.656, -20.578},
      {-24.486, -23.629, -22.983, -22.486, -22.095, -21.781, -21.524, -21.311,
       -21.132, -20.980, -20.850, -20.737, -20.639, -20.553, -20.478},
      {-24.183, -23.382, -22.774, -22.302, -21.928, -21.627, -21.381, -21.177,
       -21.005, -20.859, -20.734, -20.625, -20.531, -20.449, -20.376},
      {-23.867, -23.127, -22.561, -22.116, -21.761, -21.474, -21.238, -21.043,
       -20.878, -20.738, -20.617, -20.513, -20.423, -20.344, -20.274},
      {-23.538, -22.862, -22.340, -21.926, -21.592, -21.320, -21.096, -20.909,
       -20.751, -20.617, -20.502, -20.402, -20.315, -20.239, -20.172},
      {-23.234, -22.604, -22.120, -21.734, -21.422, -21.166, -20.953, -20.776,
       -20.625, -20.497, -20.387, -20.291, -20.208, -20.135, -20.071},
      {-22.934, -22.347, -21.898, -21.541, -21.250, -21.010, -20.811, -20.643,
       -20.500, -20.378, -20.273, -20.182, -20.102, -20.033, -19.971},
      {-22.637, -22.092, -21.676, -21.345, -21.075, -20.853, -20.666, -20.508,
       -20.374, -20.259, -20.159, -20.073, -19.997, -19.931, -19.872},
      {-22.337, -21.835, -21.452, -21.147, -20.899, -20.693, -20.520, -20.373,
       -20.247, -20.139, -20.046, -19.964, -19.892, -19.830, -19.774},
      {-22.049, -21.584, -21.230, -20.950, -20.721, -20.531, -20.372, -20.236,
       -20.119, -20.019, -19.931, -19.855, -19.788, -19.729, -19.676},
      {-21.768, -21.337, -21.011, -20.754, -20.544, -20.370, -20.223, -20.098,
       -19.991, -19.898, -19.817, -19.746, -19.683, -19.628, -19.579},
      {-21.494, -21.096, -20.796, -20.559, -20.367, -20.208, -20.074, -19.960,
       -19.861, -19.776, -19.701, -19.636, -19.578, -19.527, -19.482},
      {-21.233, -20.861, -20.585, -20.368, -20.193, -20.048, -19.926, -19.821,
       -19.732, -19.654, -19.586, -19.526, -19.473, -19.426, -19.384},
      {-20.983, -20.635, -20.380, -20.181, -20.021, -19.889, -19.778, -19.683,
       -19.602, -19.531, -19.469, -19.415, -19.367, -19.324, -19.286},
      {-20.743, -20.418, -20.182, -19.999, -19.853, -19.733, -19.633, -19.547,
       -19.474, -19.410, -19.354, -19.305, -19.261, -19.223, -19.189},
      {-20.515, -20.210, -19.991, -19.824, -19.690, -19.581, -19.490, -19.413,
       -19.347, -19.290, -19.240, -19.196, -19.157, -19.122, -19.092},
      {-20.297, -20.011, -19.808, -19.654, -19.532, -19.434, -19.352, -19.282,
       -19.223, -19.172, -19.127, -19.088, -19.054, -19.023, -18.996},
      {-20.090, -19.822, -19.633, -19.491, -19.381, -19.291, -19.218, -19.156,
       -19.103, -19.057, -19.018, -18.983, -18.952, -18.925, -18.901},
      {-19.893, -19.642, -19.467, -19.337, -19.236, -19.155, -19.089, -19.034,
       -18.987, -18.946, -18.912, -18.881, -18.854, -18.831, -18.810},
      {-19.705, -19.472, -19.309, -19.190, -19.098, -19.025, -18.966, -18.917,
       -18.876, -18.840, -18.810, -18.783, -18.760, -18.739, -18.721},
      {-19.527, -19.310, -19.161, -19.051, -18.968, -18.903, -18.851, -18.807,
       -18.771, -18.740, -18.713, -18.690, -18.670, -18.653, -18.637},
      {-19.357, -19.159, -19.022, -18.922, -18.847, -18.789, -18.743, -18.704,
       -18.673, -18.646, -18.623, -18.603, -18.586, -18.571, -18.558},
      {-19.195, -19.016, -18.892, -18.803, -18.736, -18.684, -18.643, -18.610,
       -18.583, -18.560, -18.540, -18.523, -18.509, -18.496, -18.485},
      {-19.042, -18.883, -18.772, -18.693, -18.634, -18.589, -18.553, -18.525,
       -18.501, -18.481, -18.465, -18.451, -18.438, -18.428, -18.419},
      {-18.894, -18.758, -18.662, -18.593, -18.542, -18.503, -18.473, -18.448,
       -18.428, -18.412, -18.398, -18.386, -18.376, -18.367, -18.359},
      {-18.752, -18.639, -18.559, -18.501, -18.458, -18.426, -18.400, -18.380,
       -18.363, -18.350, -18.338, -18.328, -18.320, -18.313, -18.306},
      {-18.611, -18.523, -18.460, -18.415, -18.381, -18.355, -18.334, -18.318,
       -18.304, -18.293, -18.284, -18.276, -18.269, -18.263, -18.258},
      {-18.471, -18.408, -18.362, -18.329, -18.304, -18.285, -18.269, -18.257,
       -18.247, -18.238, -18.231, -18.224, -18.219, -18.214, -18.210},
      {-18.330, -18.290, -18.261, -18.239, -18.223, -18.211, -18.201, -18.192,
       -18.185, -18.179, -18.174, -18.169, -18.165, -18.162, -18.159},
      {-18.190, -18.168, -18.154, -18.143, -18.135, -18.129, -18.124, -18.120,
       -18.116, -18.112, -18.109, -18.106, -18.104, -18.102, -18.100},
      {-18.055, -18.047, -18.043, -18.042, -18.040, -18.039, -18.039, -18.038,
       -18.037, -18.036, -18.035, -18.034, -18.033, -18.033, -18.032},
      {-17.929, -17.931, -17.935, -17.939, -17.943, -17.946, -17.948, -17.950,
       -17.952, -17.953, -17.955, -17.956, -17.957, -17.958, -17.959},
      {-17.818, -17.826, -17.834, -17.842, -17.849, -17.855, -17.860, -17.865,
       -17.869, -17.872, -17.875, -17.878, -17.881, -17.883, -17.886},
      {-17.724, -17.736, -17.747, -17.758, -17.767, -17.775, -17.782, -17.788,
       -17.793, -17.798, -17.803, -17.807, -17.811, -17.815, -17.819},
      {-17.651, -17.665, -17.678, -17.690, -17.701, -17.710, -17.718, -17.725,
       -17.732, -17.738, -17.744, -17.749, -17.755, -17.760, -17.765},
      {-17.601, -17.615, -17.629, -17.642, -17.653, -17.663, -17.672, -17.680,
       -17.688, -17.695, -17.701, -17.708, -17.714, -17.720, -17.726},
      {-17.572, -17.587, -17.602, -17.614, -17.626, -17.636, -17.645, -17.654,
       -17.662, -17.670, -17.677, -17.684, -17.691, -17.698, -17.704},
      {-17.565, -17.581, -17.595, -17.607, -17.619, -17.629, -17.638, -17.647,
       -17.656, -17.664, -17.671, -17.679, -17.686, -17.693, -17.700},
      {-17.580, -17.594, -17.608, -17.620, -17.630, -17.640, -17.650, -17.658,
       -17.667, -17.675, -17.682, -17.690, -17.697, -17.704, -17.711},
      {-17.613, -17.626, -17.639, -17.649, -17.659, -17.669, -17.677, -17.686,
       -17.694, -17.701, -17.709, -17.716, -17.723, -17.730, -17.737},
      {-17.663, -17.675, -17.685, -17.695, -17.703, -17.711, -17.719, -17.727,
       -17.734, -17.741, -17.748, -17.755, -17.761, -17.768, -17.774},
      {-17.728, -17.737, -17.745, -17.752, -17.759, -17.766, -17.772, -17.778,
       -17.785, -17.791, -17.797, -17.803, -17.808, -17.814, -17.820},
      {-17.803, -17.809, -17.814, -17.818, -17.823, -17.828, -17.832, -17.837,
       -17.842, -17.847, -17.852, -17.856, -17.861, -17.866, -17.871},
      {-17.884, -17.886, -17.888, -17.889, -17.891, -17.893, -17.896, -17.899,
       -17.902, -17.905, -17.908, -17.912, -17.915, -17.919, -17.922},
      {-17.966, -17.964, -17.961, -17.959, -17.958, -17.958, -17.958, -17.959,
       -17.960, -17.961, -17.963, -17.964, -17.966, -17.968, -17.970},
      {-18.040, -18.034, -18.028, -18.023, -18.019, -18.016, -18.013, -18.012,
       -18.010, -18.010, -18.009, -18.009, -18.009, -18.009, -18.010},
      {-18.096, -18.087, -18.078, -18.071, -18.065, -18.059, -18.055, -18.051,
       -18.047, -18.045, -18.042, -18.040, -18.039, -18.037, -18.036},
      {-18.125, -18.115, -18.105, -18.097, -18.089, -18.082, -18.076, -18.070,
       -18.065, -18.061, -18.057, -18.053, -18.051, -18.048, -18.046},
      {-18.120, -18.112, -18.103, -18.095, -18.087, -18.079, -18.072, -18.066,
       -18.060, -18.055, -18.050, -18.046, -18.042, -18.039, -18.036},
      {-18.083, -18.078, -18.071, -18.064, -18.057, -18.050, -18.044, -18.037,
       -18.032, -18.026, -18.022, -18.017, -18.014, -18.010, -18.007},
      {-18.025, -18.022, -18.017, -18.012, -18.006, -18.000, -17.994, -17.989,
       -17.984, -17.979, -17.975, -17.971, -17.968, -17.965, -17.963},
      {-17.957, -17.955, -17.952, -17.948, -17.943, -17.938, -17.934, -17.929,
       -17.925, -17.922, -17.918, -17.916, -17.913, -17.911, -17.910},
      {-17.890, -17.889, -17.886, -17.882, -17.879, -17.875, -17.871, -17.867,
       -17.864, -17.862, -17.860, -17.858, -17.857, -17.856, -17.855},
      {-17.831, -17.829, -17.826, -17.822, -17.819, -17.815, -17.812, -17.810,
       -17.807, -17.806, -17.804, -17.803, -17.803, -17.803, -17.803},
      {-17.786, -17.782, -17.777, -17.773, -17.769, -17.766, -17.763, -17.761,
       -17.759, -17.758, -17.757, -17.757, -17.757, -17.758, -17.759},
      {-17.753, -17.747, -17.741, -17.735, -17.731, -17.727, -17.724, -17.722,
       -17.721, -17.720, -17.720, -17.720, -17.721, -17.722, -17.724},
      {-17.733, -17.724, -17.716, -17.709, -17.703, -17.699, -17.696, -17.694,
       -17.693, -17.692, -17.692, -17.693, -17.694, -17.695, -17.697},
      {-17.723, -17.711, -17.700, -17.691, -17.685, -17.680, -17.676, -17.674,
       -17.673, -17.672, -17.673, -17.673, -17.675, -17.676, -17.678},
      {-17.718, -17.702, -17.689, -17.679, -17.672, -17.667, -17.663, -17.660,
       -17.659, -17.659, -17.659, -17.660, -17.661, -17.663, -17.665},
      {-17.713, -17.695, -17.681, -17.670, -17.662, -17.656, -17.653, -17.650,
       -17.649, -17.649, -17.649, -17.650, -17.651, -17.653, -17.655},
      {-17.705, -17.686, -17.671, -17.660, -17.652, -17.647, -17.643, -17.641,
       -17.640, -17.640, -17.640, -17.641, -17.643, -17.645, -17.647},
      {-17.690, -17.671, -17.657, -17.647, -17.640, -17.635, -17.632, -17.630,
       -17.630, -17.630, -17.631, -17.632, -17.634, -17.636, -17.639},
      {-17.667, -17.649, -17.637, -17.629, -17.623, -17.619, -17.618, -17.617,
       -17.617, -17.618, -17.619, -17.621, -17.623, -17.626, -17.628},
      {-17.635, -17.621, -17.611, -17.605, -17.601, -17.600, -17.599, -17.599,
       -17.601, -17.602, -17.604, -17.607, -17.609, -17.612, -17.615},
      {-17.596, -17.585, -17.579, -17.576, -17.575, -17.575, -17.576, -17.578,
       -17.580, -17.582, -17.585, -17.588, -17.591, -17.595, -17.598},
      {-17.550, -17.544, -17.542, -17.542, -17.544, -17.546, -17.548, -17.552,
       -17.555, -17.558, -17.562, -17.566, -17.570, -17.573, -17.577},
      {-17.501, -17.500, -17.501, -17.504, -17.508, -17.513, -17.517, -17.521,
       -17.526, -17.530, -17.535, -17.539, -17.544, -17.548, -17.553},
      {-17.449, -17.452, -17.457, -17.463, -17.470, -17.476, -17.482, -17.488,
       -17.493, -17.499, -17.504, -17.509, -17.514, -17.519, -17.524},
      {-17.396, -17.403, -17.412, -17.420, -17.429, -17.437, -17.444, -17.451,
       -17.458, -17.464, -17.470, -17.476, -17.481, -17.487, -17.492},
      {-17.344, -17.355, -17.366, -17.377, -17.387, -17.396, -17.405, -17.413,
       -17.420, -17.427, -17.434, -17.440, -17.446, -17.452, -17.458},
      {-17.295, -17.307, -17.321, -17.333, -17.345, -17.355, -17.365, -17.373,
       -17.382, -17.389, -17.397, -17.404, -17.410, -17.417, -17.423},
      {-17.249, -17.264, -17.278, -17.292, -17.304, -17.316, -17.326, -17.335,
       -17.344, -17.352, -17.360, -17.368, -17.375, -17.382, -17.389},
      {-17.209, -17.225, -17.241, -17.255, -17.268, -17.280, -17.291, -17.301,
       -17.310, -17.319, -17.327, -17.335, -17.343, -17.350, -17.357},
      {-17.177, -17.194, -17.210, -17.225, -17.239, -17.251, -17.262, -17.272,
       -17.282, -17.291, -17.300, -17.308, -17.316, -17.324, -17.331},
      {-17.154, -17.172, -17.189, -17.204, -17.218, -17.230, -17.242, -17.252,
       -17.262, -17.272, -17.280, -17.289, -17.298, -17.306, -17.314},
      {-17.144, -17.162, -17.179, -17.194, -17.208, -17.220, -17.232, -17.242,
       -17.253, -17.262, -17.271, -17.280, -17.289, -17.297, -17.306},
      {-17.146, -17.164, -17.181, -17.196, -17.210, -17.222, -17.234, -17.245,
       -17.255, -17.265, -17.274, -17.283, -17.292, -17.301, -17.309},
      {-17.163, -17.180, -17.197, -17.212, -17.225, -17.237, -17.249, -17.260,
       -17.270, -17.280, -17.289, -17.298, -17.307, -17.316, -17.325},
      {-17.193, -17.211, -17.227, -17.241, -17.254, -17.266, -17.277, -17.288,
       -17.298, -17.308, -17.317, -17.327, -17.336, -17.345, -17.353},
      {-17.239, -17.256, -17.271, -17.284, -17.297, -17.309, -17.320, -17.330,
       -17.340, -17.350, -17.359, -17.369, -17.378, -17.387, -17.395},
      {-17.299, -17.315, -17.329, -17.342, -17.354, -17.365, -17.376, -17.386,
       -17.396, -17.405, -17.415, -17.424, -17.433, -17.442, -17.451},
      {-17.373, -17.388, -17.402, -17.414, -17.425, -17.436, -17.446, -17.456,
       -17.466, -17.475, -17.484, -17.493, -17.502, -17.511, -17.520},
      {-17.462, -17.476, -17.489, -17.500, -17.511, -17.521, -17.531, -17.541,
       -17.550, -17.559, -17.569, -17.578, -17.587, -17.595, -17.604},
      {-17.567, -17.581, -17.592, -17.603, -17.613, -17.623, -17.632, -17.641,
       -17.651, -17.660, -17.669, -17.678, -17.686, -17.695, -17.704},
      {-17.689, -17.701, -17.712, -17.722, -17.732, -17.741, -17.750, -17.759,
       -17.768, -17.777, -17.786, -17.795, -17.803, -17.812, -17.821},
      {-17.829, -17.840, -17.851, -17.860, -17.869, -17.878, -17.887, -17.896,
       -17.904, -17.913, -17.922, -17.930, -17.939, -17.948, -17.956},
      {-17.988, -18.000, -18.010, -18.019, -18.028, -18.036, -18.045, -18.053,
       -18.062, -18.070, -18.079, -18.087, -18.096, -18.104, -18.112},
      {-18.171, -18.183, -18.192, -18.201, -18.210, -18.218, -18.227, -18.235,
       -18.243, -18.252, -18.260, -18.268, -18.277, -18.285, -18.293},
      {-18.381, -18.393, -18.403, -18.413, -18.422, -18.430, -18.438, -18.447,
       -18.455, -18.463, -18.471, -18.479, -18.487, -18.495, -18.503},
      {-18.625, -18.638, -18.650, -18.660, -18.669, -18.678, -18.687, -18.695,
       -18.703, -18.711, -18.719, -18.726, -18.734, -18.742, -18.750},
      {-18.912, -18.929, -18.943, -18.955, -18.966, -18.975, -18.984, -18.993,
       -19.001, -19.008, -19.016, -19.023, -19.031, -19.038, -19.045},
      {-19.260, -19.283, -19.303, -19.320, -19.333, -19.345, -19.355, -19.364,
       -19.372, -19.380, -19.387, -19.394, -19.400, -19.407, -19.413},
      {-19.704, -19.740, -19.771, -19.796, -19.816, -19.832, -19.845, -19.855,
       -19.863, -19.870, -19.876, -19.882, -19.887, -19.892, -19.897},
      {-20.339, -20.386, -20.424, -20.454, -20.476, -20.492, -20.502, -20.509,
       -20.513, -20.516, -20.518, -20.520, -20.521, -20.523, -20.524},
      {-21.052, -21.075, -21.093, -21.105, -21.114, -21.120, -21.123, -21.125,
       -21.126, -21.127, -21.128, -21.130, -21.131, -21.133, -21.135},
      {-21.174, -21.203, -21.230, -21.255, -21.278, -21.299, -21.320, -21.339,
       -21.357, -21.375, -21.392, -21.408, -21.424, -21.439, -21.454},
      {-21.285, -21.317, -21.346, -21.372, -21.395, -21.416, -21.435, -21.452,
       -21.468, -21.483, -21.497, -21.511, -21.524, -21.536, -21.548},
      {-21.396, -21.429, -21.459, -21.486, -21.511, -21.532, -21.551, -21.569,
       -21.585, -21.600, -21.614, -21.627, -21.640, -21.652, -21.663},
      {-21.516, -21.549, -21.580, -21.609, -21.635, -21.658, -21.678, -21.696,
       -21.713, -21.728, -21.742, -21.755, -21.767, -21.779, -21.790},
      {-21.651, -21.681, -21.711, -21.738, -21.763, -21.785, -21.804, -21.821,
       -21.837, -21.851, -21.864, -21.876, -21.887, -21.898, -21.908},
      {-21.810, -21.831, -21.853, -21.874, -21.893, -21.910, -21.925, -21.938,
       -21.950, -21.961, -21.971, -21.980, -21.989, -21.998, -22.006},
      {-22.009, -22.016, -22.026, -22.037, -22.048, -22.058, -22.066, -22.074,
       -22.081, -22.088, -22.094, -22.099, -22.105, -22.111, -22.117},
      {-22.353, -22.317, -22.296, -22.284, -22.276, -22.270, -22.266, -22.262,
       -22.260, -22.258, -22.257, -22.257, -22.257, -22.258, -22.259},
      {-22.705, -22.609, -22.552, -22.515, -22.488, -22.468, -22.451, -22.438,
       -22.427, -22.418, -22.410, -22.405, -22.400, -22.397, -22.395},
      {-22.889, -22.791, -22.731, -22.690, -22.659, -22.634, -22.612, -22.594,
       -22.579, -22.566, -22.555, -22.546, -22.539, -22.533, -22.528},
      {-23.211, -23.109, -23.041, -22.989, -22.945, -22.906, -22.872, -22.842,
       -22.816, -22.793, -22.774, -22.757, -22.743, -22.732, -22.722},
      {-25.312, -24.669, -24.250, -23.959, -23.746, -23.587, -23.463, -23.366,
       -23.288, -23.225, -23.173, -23.131, -23.095, -23.066, -23.041},
      {-25.394, -24.752, -24.333, -24.041, -23.829, -23.669, -23.546, -23.449,
       -23.371, -23.308, -23.256, -23.214, -23.178, -23.149, -23.124},
      {-25.430, -24.787, -24.369, -24.077, -23.865, -23.705, -23.582, -23.484,
       -23.407, -23.344, -23.292, -23.249, -23.214, -23.185, -23.160}};

  /* --- Return immediately if OH does not exist, or energy is
         outside tabulated range --                    -------------- */


    if (OH.shape(0) != atmos.Nspace)
        return false;


    namespace C = Constants;
    Eev = C::HC / (lambda * C::NM_TO_M) / C::EV;
    if (Eev < EOH[0] || Eev > EOH[NEOH - 1])
        return false;

    // Hunt(NEOH, EOH, Eev, &index);
    // auto last = EOH + NEOH;
    // auto it = std::upper_bound(EOH, last, Eev) - 1;
    // index = it - EOH;
    index = hunt(NEOH, EOH, Eev);
    e_index = (double)index + (Eev - EOH[index]) / (EOH[index + 1] - EOH[index]);

    hc_kla = C::HC / (C::KBoltzmann * C::NM_TO_M * lambda);
    twohnu3_c2 = (2.0 * C::HC) / cube(C::NM_TO_M * lambda);

    for (int k = 0; k < atmos.Nspace; k++) {
        if (atmos.temperature(k) < TOH[0] || atmos.temperature(k) > TOH[NTOH - 1]) {
            chi(k) = 0.0;
            eta(k) = 0.0;
        } else {

            // Hunt(NTOH, TOH, atmos.T[k], &index2);
            // auto last = TOH + NTOH;
            // auto it = std::upper_bound(TOH, last, atmos.temperature(k)) - 1;
            // index2 = it - TOH;
            index2 = hunt(NTOH, TOH, atmos.temperature(k));
            t_index = (double)index2 +
                        (atmos.temperature(k) - TOH[index2]) / (TOH[index2 + 1] - TOH[index2]);

            kappa = exp(C::Log10 * bilinear(NTOH, NEOH, OH_cross[0], t_index, e_index)) *
                    square(C::CM_TO_M);
            stimEmis = exp(-hc_kla / atmos.temperature[k]);
            chi(k) = OH(k) * (1.0 - stimEmis) * kappa;
            eta(k) = OH(k) * twohnu3_c2 * stimEmis * kappa;
        }
    }
    return true;
}

#define NTCH 15
#define NECH 105

bool CH_bf_opac(const Atmosphere& atmos, f64 lambda, F64View CH, F64View chi, F64View eta) {
  int index = 0;
  int index2 = 0;
  double Eev, e_index, t_index, hc_kla, twohnu3_c2, kappa, stimEmis;

  static constexpr double TCH[NTCH] = {2000.00, 2500.00, 3000.00, 3500.00, 4000.00,
                             4500.00, 5000.00, 5500.00, 6000.00, 6500.00,
                             7000.00, 7500.00, 8000.00, 8500.00, 9000.00};

  static constexpr double ECH[NECH] = {
      0.1, 0.2,  0.3,  0.4,  0.5,  0.6,  0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
      1.5, 1.6,  1.7,  1.8,  1.9,  2.0,  2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8,
      2.9, 3.0,  3.1,  3.2,  3.3,  3.4,  3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2,
      4.3, 4.4,  4.5,  4.6,  4.7,  4.8,  4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6,
      5.7, 5.8,  5.9,  6.0,  6.1,  6.2,  6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0,
      7.1, 7.2,  7.3,  7.4,  7.5,  7.6,  7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4,
      8.5, 8.6,  8.7,  8.8,  8.9,  9.0,  9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8,
      9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5};

  static constexpr double CH_cross[NECH][NTCH] = {
      {-38.000, -38.000, -38.000, -38.000, -38.000, -38.000, -38.000, -38.000,
       -38.000, -38.000, -38.000, -38.000, -38.000, -38.000, -38.000},
      {-32.727, -31.151, -30.133, -29.432, -28.925, -28.547, -28.257, -28.030,
       -27.848, -27.701, -27.580, -27.479, -27.395, -27.322, -27.261},
      {-31.588, -30.011, -28.993, -28.290, -27.784, -27.405, -27.115, -26.887,
       -26.705, -26.558, -26.437, -26.336, -26.251, -26.179, -26.117},
      {-30.407, -28.830, -27.811, -27.108, -26.601, -26.223, -25.932, -25.705,
       -25.523, -25.376, -25.255, -25.154, -25.069, -24.997, -24.935},
      {-29.513, -27.937, -26.920, -26.218, -25.712, -25.334, -25.043, -24.816,
       -24.635, -24.487, -24.366, -24.266, -24.181, -24.109, -24.047},
      {-28.910, -27.341, -26.327, -25.628, -25.123, -24.746, -24.457, -24.230,
       -24.049, -23.902, -23.782, -23.681, -23.597, -23.525, -23.464},
      {-28.517, -26.961, -25.955, -25.261, -24.760, -24.385, -24.098, -23.873,
       -23.694, -23.548, -23.429, -23.329, -23.245, -23.174, -23.113},
      {-28.213, -26.675, -25.680, -24.993, -24.497, -24.127, -23.843, -23.620,
       -23.443, -23.299, -23.181, -23.082, -22.999, -22.929, -22.869},
      {-27.942, -26.427, -25.446, -24.769, -24.280, -23.915, -23.635, -23.416,
       -23.241, -23.100, -22.983, -22.887, -22.805, -22.736, -22.677},
      {-27.706, -26.210, -25.241, -24.572, -24.088, -23.728, -23.451, -23.235,
       -23.063, -22.923, -22.808, -22.713, -22.633, -22.565, -22.507},
      {-27.475, -26.000, -25.043, -24.382, -23.905, -23.548, -23.275, -23.062,
       -22.891, -22.753, -22.640, -22.546, -22.467, -22.400, -22.343},
      {-27.221, -25.783, -24.844, -24.193, -23.723, -23.372, -23.102, -22.892,
       -22.724, -22.588, -22.476, -22.384, -22.306, -22.240, -22.184},
      {-26.863, -25.506, -24.607, -23.979, -23.523, -23.182, -22.919, -22.714,
       -22.550, -22.417, -22.309, -22.218, -22.142, -22.078, -22.023},
      {-26.685, -25.347, -24.457, -23.835, -23.382, -23.044, -22.784, -22.580,
       -22.418, -22.286, -22.178, -22.089, -22.014, -21.950, -21.896},
      {-26.085, -24.903, -24.105, -23.538, -23.120, -22.805, -22.561, -22.370,
       -22.217, -22.093, -21.991, -21.906, -21.835, -21.775, -21.723},
      {-25.902, -24.727, -23.936, -23.376, -22.964, -22.654, -22.415, -22.227,
       -22.076, -21.955, -21.855, -21.772, -21.702, -21.644, -21.593},
      {-25.215, -24.196, -23.510, -23.019, -22.655, -22.378, -22.163, -21.992,
       -21.855, -21.744, -21.653, -21.577, -21.513, -21.459, -21.412},
      {-24.914, -23.937, -23.284, -22.820, -22.475, -22.212, -22.007, -21.845,
       -21.715, -21.609, -21.522, -21.449, -21.388, -21.336, -21.292},
      {-24.519, -23.637, -23.039, -22.606, -22.281, -22.030, -21.834, -21.678,
       -21.552, -21.450, -21.365, -21.295, -21.236, -21.185, -21.142},
      {-24.086, -23.222, -22.650, -22.246, -21.948, -21.722, -21.546, -21.407,
       -21.296, -21.205, -21.131, -21.070, -21.018, -20.974, -20.937},
      {-23.850, -23.018, -22.472, -22.088, -21.805, -21.590, -21.422, -21.289,
       -21.182, -21.095, -21.024, -20.964, -20.914, -20.872, -20.835},
      {-23.136, -22.445, -21.994, -21.676, -21.440, -21.259, -21.117, -21.004,
       -20.912, -20.837, -20.775, -20.723, -20.679, -20.642, -20.611},
      {-23.199, -22.433, -21.927, -21.573, -21.314, -21.119, -20.969, -20.851,
       -20.758, -20.682, -20.621, -20.571, -20.529, -20.493, -20.463},
      {-22.696, -22.020, -21.585, -21.286, -21.071, -20.912, -20.791, -20.697,
       -20.622, -20.563, -20.514, -20.475, -20.442, -20.414, -20.391},
      {-22.119, -21.557, -21.194, -20.943, -20.761, -20.624, -20.518, -20.434,
       -20.367, -20.313, -20.268, -20.231, -20.201, -20.175, -20.153},
      {-21.855, -21.300, -20.931, -20.673, -20.485, -20.344, -20.235, -20.151,
       -20.084, -20.031, -19.988, -19.953, -19.924, -19.900, -19.880},
      {-21.126, -20.673, -20.382, -20.184, -20.044, -19.943, -19.868, -19.811,
       -19.769, -19.736, -19.710, -19.690, -19.674, -19.662, -19.652},
      {-20.502, -20.150, -19.922, -19.766, -19.657, -19.578, -19.520, -19.478,
       -19.446, -19.422, -19.404, -19.390, -19.379, -19.371, -19.365},
      {-20.030, -19.724, -19.530, -19.399, -19.309, -19.245, -19.199, -19.166,
       -19.142, -19.125, -19.112, -19.103, -19.096, -19.091, -19.088},
      {-19.640, -19.364, -19.189, -19.074, -18.996, -18.943, -18.906, -18.881,
       -18.863, -18.852, -18.844, -18.839, -18.837, -18.836, -18.836},
      {-19.333, -19.092, -18.939, -18.838, -18.770, -18.725, -18.695, -18.675,
       -18.662, -18.655, -18.651, -18.649, -18.649, -18.651, -18.653},
      {-19.070, -18.880, -18.756, -18.674, -18.621, -18.585, -18.562, -18.548,
       -18.540, -18.536, -18.536, -18.537, -18.539, -18.542, -18.546},
      {-18.851, -18.708, -18.617, -18.558, -18.521, -18.498, -18.484, -18.477,
       -18.475, -18.476, -18.478, -18.482, -18.487, -18.493, -18.498},
      {-18.709, -18.599, -18.533, -18.494, -18.471, -18.459, -18.454, -18.454,
       -18.457, -18.462, -18.469, -18.476, -18.483, -18.490, -18.498},
      {-18.656, -18.572, -18.524, -18.497, -18.485, -18.480, -18.482, -18.486,
       -18.493, -18.501, -18.510, -18.519, -18.527, -18.536, -18.544},
      {-18.670, -18.613, -18.582, -18.566, -18.561, -18.562, -18.568, -18.575,
       -18.583, -18.592, -18.601, -18.610, -18.619, -18.627, -18.635},
      {-18.728, -18.700, -18.687, -18.683, -18.685, -18.691, -18.698, -18.706,
       -18.715, -18.723, -18.731, -18.739, -18.745, -18.752, -18.758},
      {-18.839, -18.835, -18.836, -18.842, -18.849, -18.857, -18.865, -18.872,
       -18.878, -18.883, -18.888, -18.892, -18.895, -18.898, -18.900},
      {-19.034, -19.041, -19.049, -19.057, -19.064, -19.069, -19.071, -19.071,
       -19.070, -19.068, -19.065, -19.061, -19.058, -19.054, -19.051},
      {-19.372, -19.378, -19.382, -19.380, -19.372, -19.359, -19.341, -19.321,
       -19.300, -19.280, -19.261, -19.243, -19.227, -19.212, -19.199},
      {-19.780, -19.777, -19.763, -19.732, -19.686, -19.631, -19.573, -19.517,
       -19.465, -19.419, -19.379, -19.344, -19.314, -19.288, -19.265},
      {-20.151, -20.133, -20.087, -20.009, -19.911, -19.810, -19.715, -19.631,
       -19.559, -19.497, -19.446, -19.402, -19.365, -19.333, -19.306},
      {-20.525, -20.454, -20.312, -20.138, -19.970, -19.825, -19.705, -19.607,
       -19.528, -19.464, -19.411, -19.367, -19.330, -19.300, -19.274},
      {-20.869, -20.655, -20.366, -20.104, -19.894, -19.731, -19.604, -19.505,
       -19.426, -19.363, -19.312, -19.271, -19.236, -19.208, -19.184},
      {-21.179, -20.768, -20.380, -20.081, -19.856, -19.686, -19.556, -19.454,
       -19.375, -19.311, -19.260, -19.218, -19.184, -19.155, -19.131},
      {-21.167, -20.601, -20.206, -19.925, -19.719, -19.565, -19.447, -19.355,
       -19.283, -19.226, -19.180, -19.143, -19.112, -19.087, -19.066},
      {-20.918, -20.348, -19.976, -19.720, -19.536, -19.401, -19.299, -19.220,
       -19.159, -19.112, -19.073, -19.043, -19.018, -18.998, -18.981},
      {-20.753, -20.204, -19.847, -19.602, -19.427, -19.299, -19.203, -19.129,
       -19.072, -19.028, -18.993, -18.965, -18.942, -18.924, -18.909},
      {-20.456, -19.987, -19.677, -19.460, -19.302, -19.186, -19.098, -19.030,
       -18.978, -18.937, -18.904, -18.878, -18.857, -18.841, -18.827},
      {-20.154, -19.734, -19.461, -19.272, -19.136, -19.035, -18.960, -18.902,
       -18.858, -18.824, -18.797, -18.775, -18.759, -18.745, -18.735},
      {-19.941, -19.544, -19.288, -19.114, -18.992, -18.903, -18.837, -18.788,
       -18.751, -18.723, -18.701, -18.684, -18.671, -18.661, -18.654},
      {-19.657, -19.321, -19.104, -18.956, -18.853, -18.779, -18.724, -18.684,
       -18.655, -18.632, -18.615, -18.602, -18.592, -18.585, -18.579},
      {-19.388, -19.109, -18.930, -18.810, -18.725, -18.664, -18.620, -18.586,
       -18.562, -18.543, -18.529, -18.518, -18.510, -18.503, -18.498},
      {-19.201, -18.953, -18.794, -18.686, -18.611, -18.556, -18.515, -18.485,
       -18.462, -18.446, -18.433, -18.423, -18.416, -18.410, -18.406},
      {-18.923, -18.719, -18.588, -18.500, -18.439, -18.396, -18.365, -18.344,
       -18.328, -18.318, -18.311, -18.307, -18.304, -18.303, -18.302},
      {-18.614, -18.458, -18.361, -18.298, -18.258, -18.232, -18.216, -18.206,
       -18.202, -18.201, -18.202, -18.205, -18.208, -18.213, -18.218},
      {-18.419, -18.295, -18.222, -18.178, -18.153, -18.139, -18.132, -18.131,
       -18.133, -18.138, -18.143, -18.150, -18.157, -18.164, -18.172},
      {-18.296, -18.201, -18.148, -18.118, -18.101, -18.094, -18.091, -18.093,
       -18.096, -18.101, -18.107, -18.113, -18.120, -18.126, -18.132},
      {-18.021, -17.992, -17.977, -17.970, -17.967, -17.968, -17.970, -17.974,
       -17.978, -17.983, -17.989, -17.994, -18.000, -18.005, -18.011},
      {-17.694, -17.686, -17.686, -17.691, -17.698, -17.708, -17.718, -17.729,
       -17.740, -17.750, -17.761, -17.771, -17.781, -17.790, -17.798},
      {-17.374, -17.384, -17.400, -17.420, -17.440, -17.462, -17.483, -17.503,
       -17.523, -17.541, -17.558, -17.575, -17.590, -17.603, -17.616},
      {-17.169, -17.199, -17.230, -17.262, -17.293, -17.323, -17.351, -17.378,
       -17.404, -17.427, -17.449, -17.469, -17.488, -17.505, -17.520},
      {-17.151, -17.184, -17.217, -17.250, -17.282, -17.313, -17.342, -17.369,
       -17.395, -17.418, -17.440, -17.461, -17.480, -17.497, -17.513},
      {-17.230, -17.260, -17.290, -17.320, -17.348, -17.375, -17.401, -17.425,
       -17.448, -17.469, -17.489, -17.508, -17.525, -17.541, -17.556},
      {-17.379, -17.403, -17.425, -17.446, -17.467, -17.486, -17.505, -17.524,
       -17.541, -17.558, -17.574, -17.588, -17.602, -17.615, -17.627},
      {-17.596, -17.604, -17.609, -17.612, -17.616, -17.622, -17.628, -17.636,
       -17.644, -17.652, -17.661, -17.670, -17.679, -17.687, -17.695},
      {-17.846, -17.823, -17.795, -17.770, -17.750, -17.735, -17.725, -17.719,
       -17.716, -17.715, -17.716, -17.719, -17.722, -17.726, -17.730},
      {-18.089, -18.015, -17.942, -17.882, -17.836, -17.802, -17.777, -17.760,
       -17.748, -17.740, -17.736, -17.734, -17.733, -17.734, -17.736},
      {-18.299, -18.156, -18.038, -17.947, -17.881, -17.833, -17.798, -17.774,
       -17.757, -17.745, -17.738, -17.733, -17.730, -17.729, -17.729},
      {-18.441, -18.243, -18.096, -17.991, -17.915, -17.860, -17.821, -17.792,
       -17.772, -17.757, -17.746, -17.738, -17.733, -17.730, -17.728},
      {-18.474, -18.262, -18.111, -18.004, -17.926, -17.869, -17.826, -17.795,
       -17.771, -17.753, -17.740, -17.730, -17.722, -17.717, -17.713},
      {-18.387, -18.191, -18.053, -17.952, -17.878, -17.823, -17.782, -17.752,
       -17.729, -17.711, -17.698, -17.689, -17.681, -17.676, -17.672},
      {-18.161, -17.990, -17.874, -17.793, -17.736, -17.696, -17.668, -17.648,
       -17.634, -17.625, -17.619, -17.616, -17.614, -17.614, -17.615},
      {-17.908, -17.774, -17.690, -17.637, -17.604, -17.583, -17.572, -17.567,
       -17.566, -17.568, -17.571, -17.576, -17.581, -17.587, -17.593},
      {-17.681, -17.589, -17.540, -17.515, -17.506, -17.505, -17.511, -17.520,
       -17.530, -17.542, -17.554, -17.566, -17.578, -17.589, -17.600},
      {-17.647, -17.606, -17.584, -17.575, -17.573, -17.576, -17.582, -17.589,
       -17.597, -17.605, -17.614, -17.623, -17.631, -17.639, -17.646},
      {-17.300, -17.291, -17.291, -17.297, -17.307, -17.319, -17.333, -17.347,
       -17.361, -17.375, -17.389, -17.402, -17.415, -17.427, -17.438},
      {-16.786, -16.802, -16.825, -16.853, -16.883, -16.914, -16.944, -16.974,
       -17.003, -17.030, -17.055, -17.079, -17.101, -17.122, -17.141},
      {-16.489, -16.533, -16.579, -16.625, -16.670, -16.713, -16.754, -16.793,
       -16.830, -16.864, -16.896, -16.925, -16.952, -16.977, -17.000},
      {-16.694, -16.724, -16.756, -16.789, -16.823, -16.856, -16.888, -16.919,
       -16.949, -16.976, -17.002, -17.026, -17.048, -17.069, -17.088},
      {-16.935, -16.951, -16.971, -16.993, -17.016, -17.040, -17.064, -17.088,
       -17.111, -17.132, -17.153, -17.172, -17.190, -17.206, -17.222},
      {-17.200, -17.208, -17.220, -17.235, -17.251, -17.269, -17.286, -17.304,
       -17.322, -17.338, -17.354, -17.369, -17.384, -17.397, -17.409},
      {-17.597, -17.591, -17.589, -17.590, -17.594, -17.600, -17.608, -17.617,
       -17.626, -17.635, -17.645, -17.654, -17.662, -17.671, -17.679},
      {-18.166, -18.134, -18.107, -18.085, -18.068, -18.056, -18.047, -18.041,
       -18.038, -18.036, -18.035, -18.035, -18.036, -18.038, -18.039},
      {-19.000, -18.917, -18.838, -18.770, -18.714, -18.669, -18.632, -18.603,
       -18.579, -18.560, -18.545, -18.532, -18.522, -18.514, -18.507},
      {-20.313, -19.982, -19.754, -19.592, -19.472, -19.380, -19.309, -19.253,
       -19.208, -19.172, -19.143, -19.119, -19.099, -19.083, -19.069},
      {-19.751, -19.611, -19.520, -19.461, -19.423, -19.398, -19.382, -19.372,
       -19.366, -19.364, -19.363, -19.364, -19.366, -19.368, -19.371},
      {-19.581, -19.431, -19.337, -19.277, -19.240, -19.218, -19.207, -19.202,
       -19.203, -19.207, -19.212, -19.220, -19.228, -19.236, -19.245},
      {-19.685, -19.506, -19.389, -19.311, -19.258, -19.222, -19.199, -19.184,
       -19.175, -19.170, -19.168, -19.169, -19.171, -19.174, -19.177},
      {-19.977, -19.756, -19.606, -19.501, -19.425, -19.370, -19.330, -19.300,
       -19.278, -19.262, -19.250, -19.241, -19.235, -19.230, -19.227},
      {-20.445, -20.158, -19.958, -19.815, -19.711, -19.633, -19.574, -19.528,
       -19.493, -19.465, -19.442, -19.425, -19.410, -19.398, -19.389},
      {-20.980, -20.625, -20.391, -20.229, -20.110, -20.020, -19.949, -19.892,
       -19.846, -19.807, -19.775, -19.748, -19.724, -19.704, -19.687},
      {-21.404, -21.023, -20.771, -20.594, -20.461, -20.358, -20.274, -20.205,
       -20.148, -20.099, -20.058, -20.022, -19.991, -19.965, -19.942},
      {-21.309, -20.970, -20.753, -20.603, -20.495, -20.412, -20.348, -20.295,
       -20.252, -20.215, -20.185, -20.158, -20.135, -20.115, -20.098},
      {-21.221, -20.906, -20.707, -20.574, -20.480, -20.412, -20.361, -20.322,
       -20.292, -20.268, -20.249, -20.233, -20.221, -20.210, -20.201},
      {-21.441, -21.097, -20.878, -20.728, -20.623, -20.546, -20.489, -20.446,
       -20.413, -20.387, -20.368, -20.352, -20.340, -20.330, -20.322},
      {-21.668, -21.305, -21.071, -20.911, -20.797, -20.713, -20.650, -20.602,
       -20.565, -20.536, -20.514, -20.496, -20.481, -20.470, -20.460},
      {-21.926, -21.556, -21.316, -21.150, -21.031, -20.942, -20.874, -20.822,
       -20.782, -20.750, -20.724, -20.704, -20.687, -20.674, -20.663},
      {-22.319, -21.937, -21.686, -21.510, -21.380, -21.282, -21.206, -21.147,
       -21.099, -21.061, -21.031, -21.006, -20.985, -20.968, -20.954},
      {-22.969, -22.561, -22.288, -22.092, -21.945, -21.832, -21.743, -21.672,
       -21.616, -21.570, -21.533, -21.503, -21.477, -21.457, -21.439},
      {-24.001, -23.527, -23.199, -22.957, -22.772, -22.629, -22.516, -22.427,
       -22.355, -22.297, -22.250, -22.212, -22.180, -22.153, -22.131},
      {-24.233, -23.774, -23.477, -23.273, -23.128, -23.022, -22.943, -22.883,
       -22.837, -22.802, -22.774, -22.752, -22.735, -22.721, -22.710},
      {-24.550, -23.913, -23.521, -23.266, -23.094, -22.976, -22.893, -22.836,
       -22.796, -22.768, -22.750, -22.737, -22.730, -22.726, -22.725},
      {-24.301, -23.665, -23.274, -23.019, -22.848, -22.730, -22.648, -22.591,
       -22.552, -22.525, -22.507, -22.495, -22.489, -22.485, -22.485},
      {-24.519, -23.883, -23.491, -23.237, -23.065, -22.948, -22.866, -22.809,
       -22.770, -22.743, -22.724, -22.713, -22.706, -22.703, -22.702}};

  /* --- Return immediately if CH does not exist, or energy is
         outside tabulated range --                    -------------- */
    namespace C = Constants;

    if (CH.shape(0) != atmos.Nspace)
        return false;
    Eev = C::HC / (lambda * C::NM_TO_M) / C::EV;
    if (Eev < ECH[0] || Eev > ECH[NECH - 1])
        return false;

    // Hunt(NECH, ECH, Eev, &index);
    // auto last = ECH + NECH;
    // auto it = std::upper_bound(ECH, last, Eev) - 1;
    // index = it - ECH;
    index = hunt(NECH, ECH, Eev);
    e_index = (double)index + (Eev - ECH[index]) / (ECH[index + 1] - ECH[index]);

    hc_kla = C::HC / (C::KBoltzmann * C::NM_TO_M * lambda);
    twohnu3_c2 = (2.0 * C::HC) / cube(C::NM_TO_M * lambda);

    for (int k = 0; k < atmos.Nspace; k++) {
        if (atmos.temperature(k) < TCH[0] || atmos.temperature(k) > TCH[NTCH - 1]) {
            chi(k) = 0.0;
            eta(k) = 0.0;
        } else {
            // Hunt(NTCH, TCH, atmos.T[k], &index2);
            // auto last = TCH + NTCH;
            // auto it = std::upper_bound(TCH, last, atmos.temperature(k));
            // index2 = it - TCH;
            index2 = hunt(NTCH, TCH, atmos.temperature(k));
            t_index = (double)index2 +
                        (atmos.temperature(k) - TCH[index2]) / (TCH[index2 + 1] - TCH[index2]);

            kappa = exp(C::Log10 * bilinear(NTCH, NECH, CH_cross[0], t_index, e_index)) *
                    square(C::CM_TO_M);
            stimEmis = exp(-hc_kla / atmos.temperature(k));
            chi(k) = CH(k) * (1.0 - stimEmis) * kappa;
            eta(k) = CH(k) * twohnu3_c2 * stimEmis * kappa;
        }
    }
    return true;
}

void basic_background(BackgroundData* bd)
{
    JasUnpack((*bd), chPops, ohPops, h2Pops, hMinusPops, hPops);
    JasUnpack((*bd), wavelength, chi, eta, scatt);
    const auto& atmos = *bd->atmos;

    chi.fill(0.0);
    eta.fill(0.0);
    scatt.fill(0.0);

    const int Nlambda = wavelength.shape(0);

    F64Arr chiAccum(atmos.Nspace);
    F64Arr etaAccum(atmos.Nspace);
    F64Arr scaAccum(atmos.Nspace);

    thomson_scattering(atmos, scaAccum);

    for (int la = 0; la < Nlambda; ++la)
        for (int k = 0; k < atmos.Nspace; ++k)
            scatt(la, k) += scaAccum(k);

    F64Arr Bnu(atmos.Nspace);

    auto hMinus = HMinusOpacity(atmos, hMinusPops, hPops);
    auto h2 = H2Opacity(atmos, h2Pops, hPops);
    for (int la = 0; la < Nlambda; ++la)
    {
        f64 lambda = wavelength(la);
        auto etaLa = eta(la);
        auto chiLa = chi(la);
        auto scaLa = scatt(la);

        planck_nu(atmos.Nspace, atmos.temperature.data, lambda, Bnu.data.data());

        if (hMinus.hminus_bf(lambda, chiAccum, etaAccum))
        {
            for (int k = 0; k < atmos.Nspace; ++k)
            {
                chiLa(k) += chiAccum(k);
                etaLa(k) += etaAccum(k);
            }
        }
        if (hMinus.hminus_ff(lambda, chiAccum))
        {
            for (int k = 0; k < atmos.Nspace; ++k)
            {
                chiLa(k) += chiAccum(k);
                etaLa(k) += chiAccum(k) * Bnu(k);
            }
        }

        // NOTE(cmo): Do H- fudge here if doing

        if (OH_bf_opac(atmos, lambda, ohPops, chiAccum, etaAccum))
        {
            for (int k = 0; k < atmos.Nspace; ++k)
            {
                chiLa(k) += chiAccum(k);
                etaLa(k) += etaAccum(k);
            }
        }
        if (CH_bf_opac(atmos, lambda, chPops, chiAccum, etaAccum))
        {
            for (int k = 0; k < atmos.Nspace; ++k)
            {
                chiLa(k) += chiAccum(k);
                etaLa(k) += etaAccum(k);
            }
        }

        if (hydrogen_ff(atmos, lambda, hPops, chiAccum))
        {
            for (int k = 0; k < atmos.Nspace; ++k)
            {
                chiLa(k) += chiAccum(k);
                etaLa(k) += chiAccum(k) * Bnu(k);
            }
        }

        if (h2.h2plus_ff(lambda, chiAccum))
        {
            for (int k = 0; k < atmos.Nspace; ++k)
            {
                chiLa(k) += chiAccum(k);
                etaLa(k) += chiAccum(k) * Bnu(k);
            }
        }
        if (h2.h2minus_ff(lambda, chiAccum))
        {
            for (int k = 0; k < atmos.Nspace; ++k)
            {
                chiLa(k) += chiAccum(k);
                etaLa(k) += chiAccum(k) * Bnu(k);
            }
        }
        if (h2.rayleigh_H2(lambda, scaAccum))
        {
            for (int k = 0; k < atmos.Nspace; ++k)
                scaLa(k) += scaAccum(k);
        }

    }

    // NOTE(cmo): Still needed: Rayleigh for H, Rayleigh for He, bf opacities from atomic models 
    // NOTE(cmo): At end, add sca * scaFudge to chi
}