#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <Eigen/Dense>
#include "matplotlibcpp.h" // External library for plotting

namespace plt = matplotlibcpp;


class WaveSolver {
public:
    WaveSolver(double a, double b, double T, int n, double mu, const std::vector<double>& bl, const std::vector<double>& br)
        : a(a), b(b), T(T), n(n), mu(mu), bl(bl), br(br) {
        dx = (b - a) / (n - 1);
        dt = 0.99 * dx;
        m = static_cast<int>(std::ceil(T / dt));
        x.resize(n);
        t.resize(m);
        for (int i = 0; i < n; ++i) x[i] = a + i * dx;
        for (int j = 0; j < m; ++j) t[j] = j * dt;
    }

    static double u0(double x) {
        if (x > 0.25 && x < 0.75)
            return 1.0;
        else if (std::abs(x - 0.25) < 1e-8 || std::abs(x - 0.75) < 1e-8)
            return 0.5;
        return 0.0;
    }

    static double u0t(double x) {
        return 0.0;
    }

    Eigen::MatrixXd wave_direct(const std::vector<double>& u0, const std::vector<double>& u0t) {
        Eigen::MatrixXd u = Eigen::MatrixXd::Zero(n, m);
        double s = (dt * dt) / (dx * dx);

        // Initialize the wave
        for (int i = 1; i < n - 1; ++i) {
            u(i, 0) = WaveSolver::u0(x[i]);
        }
        for (int j = 0; j < m; ++j) {
            u(0, j) = bl[j];
            u(n - 1, j) = br[j];
        }

        // Compute first time step
        for (int i = 1; i < n - 1; ++i) {
            u(i, 1) = u(i, 0) + dt * WaveSolver::u0t(x[i]);
        }

        // Update the wave over time
        for (int j = 2; j < m; ++j) {
            for (int i = 1; i < n - 1; ++i) {
                u(i, j) = s * (u(i - 1, j - 1) + u(i + 1, j - 1))
                    + 2 * (1 - s) * u(i, j - 1)
                    - u(i, j - 2);
            }
        }

        return u;
    }

    Eigen::VectorXd solve() {
        std::vector<double> u01(n - 2), u01t(n - 2);
        for (int i = 0; i < n - 2; ++i) {
            u01[i] = WaveSolver::u0(x[i + 1]);
            u01t[i] = WaveSolver::u0t(x[i + 1]);
        }

        Eigen::VectorXd f(2 * (n - 2));
        for (int i = 0; i < n - 2; ++i) {
            f[i] = u01t[i];
            f[i + n - 2] = -u01[i];
        }

        // Define residual function for optimization (mocking fsolve functionality)
        auto residual = [&](const Eigen::VectorXd& X) -> Eigen::VectorXd {
            Eigen::VectorXd Y = lambda_function(X.head(n - 2), X.tail(n - 2));
            return Y - f;
            };

        Eigen::VectorXd initial_guess = Eigen::VectorXd::Constant(2 * (n - 2), 2.0);
        Eigen::VectorXd Y = fsolve(residual, initial_guess); // Custom fsolve equivalent
        return Y;
    }

private:
    double a, b, T, dx, dt, mu;
    int n, m;
    std::vector<double> x, t, bl, br;

    Eigen::VectorXd lambda_function(const Eigen::VectorXd& u0, const Eigen::VectorXd& u0t) {
        // Mocking the lambda function, full implementation depends on wave_direct and additional logic
        return Eigen::VectorXd::Zero(2 * (n - 2)); // Placeholder
    }

    Eigen::VectorXd fsolve(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& residual,
        const Eigen::VectorXd& guess) {
        // Mocking fsolve (simple iterative solver could go here)
        return guess; // Placeholder
    }
};

int main() {
    double a = 0.0, b = 1.0, T = 3.0, mu = 1.0;
    int n = 161;
    double dx = (b - a) / (n - 1);
    double dt = 0.99 * dx;
    int m = static_cast<int>(std::ceil(T / dt));

    std::vector<double> bl(m, 0.0), br(m, 0.0);

    WaveSolver wave_solver(a, b, T, n, mu, bl, br);

    Eigen::VectorXd Y = wave_solver.solve();

    Eigen::MatrixXd u = wave_solver.wave_direct(std::vector<double>(n - 2, 0.0), std::vector<double>(n - 2, 0.0));
    Eigen::VectorXd v = -1 / dx * u.row(n - 2);

    std::vector<double> t(m);
    for (int j = 0; j < m; ++j) t[j] = j * dt;

    plt::plot(t, std::vector<double>(v.data(), v.data() + v.size()));
    plt::legend({ "control" });
    plt::xlabel("time");
    plt::ylabel("value");
    plt::title("The discrete control");
    plt::show();

    return 0;
}