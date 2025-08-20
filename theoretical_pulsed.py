import tqdm
import os
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh, eigs
import numpy as np
import numexpr as ne
import matplotlib.image as mpimg
from matplotlib import gridspec
from scipy.integrate import cumulative_trapezoid
import ipywidgets as widgets
from ipywidgets import interact


class TheoreticalPulsedSqueezing:
    """
    Simulate pulsed squeezing in an optical cavity with various input pulse shapes.
    
    Attributes:
        pump_power: Pump power (W) for squeezing calculation.
        simulation_time: Total simulation time (ns).
        N_points: Number of time points in simulation.
        t: Time array.
        dt: Time step.
        tau_cav: Cavity round trip time.
        L_s, L_p: Intracavity losses for signal and pump.
        T1_s, T1_p: Mirror transmissions.
        gamma_s, gamma_p: Total damping rates for signal and pump.
        betamax: Maximum amplitude coefficient.
    """
    def __init__(self, pump_power, simulation_time, N_points, R_s = 94.5, R_p = 73.4, L_s = 0.2, L_p = 0.2, tau_cav = 222e-3):
        # Simulation parameters
        self.pump_power = pump_power**.5
        self.simulation_time = simulation_time
        self.N_points = N_points
        self.t = np.linspace(0,self.simulation_time,self.N_points)
        self.dt = self.simulation_time/(self.N_points-1)

        
        # Cavity parameters
        self.tau_cav = tau_cav # cavity round trip [ns] 
        self.L_s = L_s / 100 # intra cavity loss @1550 [%/100]
        self.L_p = L_p / 100 # intra cavity loss @775 [%/100]
        self.T1_p = 1-R_p/100 # transmission coefficient @775nm
        self.T1_s = 1-R_s/100 # transmission coefficient @1550nm

        # Damping rates
        self.gammaL_s = (1-np.sqrt(1-self.L_s))/self.tau_cav
        self.gammaL_p = (1-np.sqrt(1-self.L_p))/self.tau_cav

        self.gamma_p_M1 = (1-np.sqrt(1-self.T1_p))/self.tau_cav
        self.gamma_p = self.gamma_p_M1 + self.gammaL_p
        self.gamma_s_M1 = (1-np.sqrt(1-self.T1_s))/self.tau_cav
        self.gamma_s = self.gamma_s_M1 + self.gammaL_s

        # Maximum amplitude coefficient for intra-cavity pump field
        self.betamax = np.sqrt(2*self.gamma_p_M1)/self.gamma_p
    
    def simulate_trapezoid_pulse(self, t_start=3, t_rise=2, t_plateau=3, t_fall=2, noise = 0.0):
        """
        Generate a trapezoidal pulse and compute associated matrices.
        """
        t1 = t_start
        t2 = t1 + t_rise
        t3 = t2 + t_plateau
        t4 = t3 + t_fall

        self.input_pulse =  np.piecewise(
            self.t,
            [self.t < t1,
            (t1 <= self.t) & (self.t < t2),
            (t2 <= self.t) & (self.t < t3),
            (t3 <= self.t) & (self.t < t4),
            self.t >= t4],
            [0,
            lambda t: (t - t1) / t_rise,       
            1,                              
            lambda t: 1 - (t - t3) / t_fall,  
            0]
        )
        # Add optional Gaussian noise
        self.input_pulse += noise * np.random.randn(self.N_points)

        # Solve dynamics and compute matrices
        self.solve_beta_general()
        self.compute_beta_integral_matrix_general()
        self.create_matrix_NM_fast()

    def simulate_gaussian_pulse(self, t_center=5, sigma=1, noise = 0):
        """
        Generate a Gaussian pulse centered at t_center with standard deviation sigma.
        """
        self.input_pulse = np.exp(-0.5 * ((self.t - t_center)/sigma)**2) + noise * np.random.randn(self.N_points)

        self.solve_beta_general()
        self.compute_beta_integral_matrix_general()
        self.create_matrix_NM_fast()

    def simulate_square_pulse(self, start_pulse=3, pulse_length=5):
        """
        Generate a square pulse with given start and duration.
        """
        self.start_pulse = start_pulse
        self.pulse_length = pulse_length

        start_idx = int(np.searchsorted(self.t, self.start_pulse))
        end_idx = int(np.searchsorted(self.t, self.start_pulse + self.pulse_length))

        self.input_pulse = np.zeros_like(self.t, dtype=np.float64)
        self.input_pulse[start_idx:end_idx+1] = 1.0 

        self.solve_beta_square()
        self.compute_beta_integral_matrix_square()
        self.create_matrix_NM_fast()

    def simulate_custom_pulse(self, input_pulse):
        """
        Use a user-defined pulse array.
        """
        if len(input_pulse) != self.N_points:
            raise ValueError(f"input_pulse length ({len(input_pulse)}) "
                            f"does not match expected length ({self.N_points})")
    
        self.input_pulse = input_pulse
        
        self.solve_beta_general()
        self.compute_beta_integral_matrix_general()
        self.create_matrix_NM_fast()
 
    def solve_beta_general(self):
        """
        Solve pump dynamics for general pulse using exact step update.
        dβ/dt = -γ_p β + sqrt(2 γ_p_M1) * B_in(t)
        """

        g = self.gamma_p                      # total pump damping
        kappa = np.sqrt(2 * self.gamma_p_M1)  # coupling factor
        dt = self.dt

        expfac = np.exp(-g * dt)
        # handles g -> 0 gracefully
        coeff = kappa * ((1 - expfac) / g) if g != 0 else kappa * dt

        beta = np.zeros_like(self.t, dtype=float)
        beta[0] = 0.0

        for n in range(len(self.t) - 1):
            beta[n+1] = expfac * beta[n] + coeff * self.input_pulse[n]

        self.beta = beta
        return beta
    
    def solve_beta_square(self):
        """
        Exact solution for square pulse input.
        """
        A = np.sqrt(2 * self.gamma_p_M1) / self.gamma_p
        beta = np.zeros_like(self.t, dtype=np.float64)
        start_idx = np.searchsorted(self.t, self.start_pulse)
        end_idx = np.searchsorted(self.t, self.start_pulse + self.pulse_length)

        # During the pulse
        beta[start_idx:end_idx+1] = A * (1 - np.exp(-self.gamma_p * (self.t[start_idx:end_idx+1] - self.start_pulse)))

        # After the pulse
        beta[end_idx+1:] = (
            A
            * np.exp(self.gamma_p * self.start_pulse)
            * np.expm1(self.gamma_p * self.pulse_length)
            * np.exp(-self.gamma_p * self.t[end_idx+1:])
        )

        self.beta = beta
    
    def compute_beta_integral_matrix_general(self):
        """
        Compute matrix of integrated beta values for general pulse:
        I(t_i, t_j) = ∫_{t_i}^{t_j} beta(τ) dτ
        """
        # cumulative integral (same length as self.t)
        int_beta = cumulative_trapezoid(self.beta, self.t, initial=0)
        # build matrix using broadcasting
        self.matrix_beta = np.abs(int_beta[None, :] - int_beta[:, None])

    def compute_beta_integral_matrix_square(self):
        """
        Precomputed integral matrix for square pulses (fast evaluation using numexpr).
        """
        # Implementation optimized with numexpr
        a = self.start_pulse
        b = self.start_pulse + self.pulse_length
        gamma_p = self.gamma_p
        first = np.sqrt(2 * self.gamma_p_M1) / gamma_p

        N = self.N_points
        dt = self.dt
        time_points = np.arange(N, dtype=np.float64) * dt

        exp_t         = ne.evaluate("exp(-gamma_p * time_points)")
        exp_t_minus_a = ne.evaluate("exp(-gamma_p * (time_points - a))")
        exp_t_minus_b = ne.evaluate("exp(-gamma_p * (time_points - b))")
        exp_a         = ne.evaluate("exp(gamma_p * a)")
        exp_b         = ne.evaluate("exp(gamma_p * b)")

        # Initialize output
        matrix_beta = np.zeros((N, N), dtype=np.float64)

        # Upper triangle indices
        i, j = np.triu_indices(N)
        t1 = time_points[i]
        t2 = time_points[j]

        # Precompute exponentials at selected indices
        exp_t1 = exp_t[i]
        exp_t2 = exp_t[j]
        exp_t1_a = exp_t_minus_a[i]
        exp_t2_a = exp_t_minus_a[j]
        exp_t2_b = exp_t_minus_b[j]

        # Conditions
        cond1 = (t1 <= a) & (a <= t2) & (t2 < b)
        cond2 = (a <= t1) & (t1 <= b) & (a <= t2) & (t2 <= b)
        cond3 = (t1 <= a) & (t2 >= b)
        cond4 = (a <= t1) & (t1 < b) & (t2 >= b)
        cond5 = (t1 >= b) & (t2 >= b)

        # Compute beta using NumExpr
        beta_values = ne.evaluate(
            "(cond1 * ((exp_t2_a - 1)/gamma_p + t2 - a) + "
            "cond2 * ((exp_t2_a - exp_t1_a)/gamma_p + t2 - t1) + "
            "cond3 * ((exp_t2 * (exp_a - exp_b))/gamma_p + b - a) + "
            "cond4 * ((1 + exp_t2_a - exp_t1_a - exp_t2_b)/gamma_p + b - t1) + "
            "cond5 * ((exp_t2 - exp_t1) * (exp_a - exp_b)/gamma_p)) * first",
            local_dict={
                't1': t1, 't2': t2, 'a': a, 'b': b, 'gamma_p': gamma_p,
                'first': first, 'exp_a': exp_a, 'exp_b': exp_b,
                'exp_t1': exp_t1, 'exp_t2': exp_t2, 'exp_t1_a': exp_t1_a,
                'exp_t2_a': exp_t2_a, 'exp_t2_b': exp_t2_b,
                'cond1': cond1, 'cond2': cond2, 'cond3': cond3, 'cond4': cond4, 'cond5': cond5
            }
        )

        # Fill matrix
        matrix_beta[i, j] = beta_values
        matrix_beta[j, i] = beta_values

        self.matrix_beta = matrix_beta

    def calculate_all_G(self):
        # Time points (1D)
        time_points = np.arange(self.N_points) * self.dt
        gamma_s = self.gamma_s

        # Get lower triangle indices (including diagonal)
        i_idx, j_idx = np.tril_indices(self.N_points)

        # Convert to physical times
        t1c = time_points[i_idx]
        t2c = time_points[j_idx]

        # Integer indices for beta lookup
        beta_vals = self.matrix_beta[i_idx, j_idx]
        arg = self.pump_power * self.gamma_s / self.betamax * beta_vals

        # Evaluate in numexpr
        exp_term = ne.evaluate("exp(-gamma_s * (t1c - t2c))")  # t1 >= t2
        cosh_vals = ne.evaluate("cosh(arg)")
        sinh_vals = ne.evaluate("sinh(arg)")

        # Allocate output
        G11_ = np.zeros((self.N_points, self.N_points), dtype=np.float64)
        G12_ = np.zeros_like(G11_)

        # Fill only lower triangle
        G11_[i_idx, j_idx] = ne.evaluate("exp_term * cosh_vals")
        G12_[i_idx, j_idx] = ne.evaluate("exp_term * sinh_vals")

        return G11_, G12_ 
    
    def create_matrix_NM_fast(self):
        G11_, G12_ = self.calculate_all_G()
        
        # Precompute constants
        scale_outer = 2 * self.gamma_s_M1
        scale_matmul = 2 * self.gamma_s * self.dt

        # Outer product term (elementwise multiply via numexpr)
        outer_term = ne.evaluate(
            "g12_col0_col * g12_col0_row",
            local_dict={
                'g12_col0_col': G12_[:, 0][:, None],
                'g12_col0_row': G12_[:, 0][None, :]
            }
        )


        # Matrix multiply term (BLAS-optimized)
        matmul_term = np.einsum("ik,jk->ij", G12_, G12_, optimize=True)
        # Combine using numexpr to avoid large intermediate arrays
        self.N = ne.evaluate("scale_outer * (outer_term + scale_matmul * matmul_term)")

        

        
        
        outer_term = ne.evaluate(
            "g11_col0_col * g12_col0_row",
            local_dict={
                'g11_col0_col': G11_[:, 0][:, None],
                'g12_col0_row': G12_[:, 0][None, :]
            }
        )

        matmul_term = np.einsum("ik,jk->ij", G11_, G12_, optimize=True)

        self.M = ne.evaluate(
            "scale_outer * (outer_term + scale_matmul * matmul_term - G12_T)",
            local_dict={
                'scale_outer': scale_outer,
                'outer_term': outer_term,
                'scale_matmul': scale_matmul,
                'matmul_term': matmul_term,
                'G12_T': G12_.T
            }
        )
                                    

        self.eigenvalues, self.eigenvectors = eigsh(self.N, k=10, which='LM')
        self.eigenvalues, self.eigenvectors = self.eigenvalues, self.eigenvectors


        self.first_term = 2 * np.einsum('ij,ik,kj->j', self.eigenvectors, self.N, self.eigenvectors)
        self.second_term = 2 * np.einsum('ij,ik,kj->j', self.eigenvectors, self.M, self.eigenvectors)


        self.first_eigenvector = self.eigenvectors[:,-1]
        self.squeezing_modes = 10*np.log10((self.first_term - self.second_term)*self.dt + 1)
        self.antisqueezing_modes = 10*np.log10((self.first_term + self.second_term)*self.dt + 1)
        
        self.squeezing = self.squeezing_modes[-1]
        self.antisqueezing = self.antisqueezing_modes[-1]

        self.purity = self.eigenvalues[-1] / np.sum(self.eigenvalues)
        self.schmidt = np.sum(self.eigenvalues)**2 / np.sum(self.eigenvalues**2)
        


class TheoreticalPulsed1D:
    """
    1D parameter sweep for pulsed squeezing simulations.

    Allows sweeping a single system or pulse parameter while
    tracking the resulting antisqueezing and Schmidt number.
    """

    # Pretty labels for plotting
    LABELS = {
        "pump_power": r"$\frac{\text{Peak Pump Power}}{\text{Threshold Power}}$",
        "simulation_time": r"Simulation Time [ns]",
        "N_points": r"Number of Points",
        "R_s": r"Signal Mirror Reflectivity [%]",
        "R_p": r"Pump Mirror Reflectivity [%]",
        "L_s": r"Loss Signal [%]",
        "L_p": r"Loss Pump [%]",
        "tau_cav": r"Cavity Round Trip [ns]",
        "t_start": r"Pulse Start [ns]",
        "t_rise": r"Pulse Rise [ns]",
        "t_plateau": r"Pulse Plateau [ns]",
        "t_fall": r"Pulse Fall [ns]"
    }

    def __init__(self, pump_power, simulation_time, N_points,
                 R_s=94.5, R_p=73.4, L_s=0.2, L_p=0.2, tau_cav=222e-3):
        """
        Initialize the simulation class.

        Parameters
        ----------
        pump_power : float
            Peak pump power normalized to threshold.
        simulation_time : float
            Total simulation time (ns).
        N_points : int
            Number of time points in simulation.
        R_s, R_p : float
            Mirror reflectivities [%].
        L_s, L_p : float
            Signal and pump losses [%].
        tau_cav : float
            Cavity round-trip time (ns).
        """

        # Store system parameters
        self.params = {
            "pump_power": pump_power,
            "simulation_time": simulation_time,
            "N_points": N_points,
            "R_s": R_s,
            "R_p": R_p,
            "L_s": L_s,
            "L_p": L_p,
            "tau_cav": tau_cav
        }

        # Default trapezoidal pulse parameters
        self.pulse_params = {
            "t_start": 3,
            "t_rise": 0,
            "t_plateau": 5,
            "t_fall": 0
        }

        # Sweep-related attributes
        self.sweep_param = None
        self.x_values = None
        self.results = {"antisq": [], "schmidt": []}

    def set_pulse_params(self, **kwargs):
        """Update pulse parameters. Accepts any subset of 't_start', 't_rise', 't_plateau', 't_fall'."""
        for k, v in kwargs.items():
            if k not in self.pulse_params:
                raise ValueError(f"Invalid pulse parameter: {k}")
            self.pulse_params[k] = v

    def _detect_sweep_param(self):
        """
        Detect which single parameter is being swept.
        Only one parameter may be an array; others must be scalars.
        """
        sweep_candidates = []

        # Check system params
        for k, v in self.params.items():
            if hasattr(v, "__iter__") and not isinstance(v, str):
                sweep_candidates.append(k)

        # Check pulse params
        for k, v in self.pulse_params.items():
            if hasattr(v, "__iter__") and not isinstance(v, str):
                sweep_candidates.append(k)

        if len(sweep_candidates) == 0:
            raise ValueError("One parameter must be an array for sweeping.")
        elif len(sweep_candidates) > 1:
            raise ValueError("Only one parameter can be an array for sweeping.")

        self.sweep_param = sweep_candidates[0]

        # Extract x-values
        if self.sweep_param in self.params:
            self.x_values = np.array(self.params[self.sweep_param])
        else:
            self.x_values = np.array(self.pulse_params[self.sweep_param])

    def run(self):
        """
        Run the 1D sweep simulation.
        """
        self._detect_sweep_param()
        self.results = {"antisq": [], "schmidt": [], "sq" : []}

        for val in tqdm.tqdm(self.x_values, desc=f"Sweeping {self.sweep_param}"):
            sys_copy = self.params.copy()
            pulse_copy = self.pulse_params.copy()

            if self.sweep_param in sys_copy:
                sys_copy[self.sweep_param] = val
            else:
                pulse_copy[self.sweep_param] = val

            # Run the actual simulation
            b = TheoreticalPulsedSqueezing(**sys_copy)
            b.simulate_trapezoid_pulse(**pulse_copy)

            self.results["antisq"].append(b.antisqueezing)
            self.results["schmidt"].append(b.schmidt)
            self.results["sq"].append(b.squeezing)

    def plot(self):
        """Plot antisqueezing and Schmidt number versus the sweep parameter."""
        x_label = self.LABELS.get(self.sweep_param, self.sweep_param)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Antisqueezing
        axs[0].plot(self.x_values, self.results["antisq"], 'r')
        axs[0].set_title(f"Antisqueezing vs {x_label}")
        axs[0].set_xlabel(x_label)
        axs[0].set_ylabel("Variance [dB]")
        axs[0].set_ylim((0, np.max(self.results["antisq"]) + 0.5))
        axs[0].grid(True)

        # Schmidt
        axs[1].plot(self.x_values, self.results["schmidt"], 'b')
        axs[1].set_title(f"Schmidt Number vs {x_label}")
        axs[1].set_xlabel(x_label)
        axs[1].set_ylabel("Schmidt Number")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()



class PulsedSqueezingVisualizer:
    """
    Visualizer for pulsed squeezing simulations with various pulse shapes.
    """
    def __init__(self, squeezing_class):
        """
        Parameters
        ----------
        squeezing_class : class
            Class implementing the pulsed squeezing simulation.
        """
        self.squeezing_class = squeezing_class

    def _initialize_sim(self, pump_power, simulation_time, N_points,
                        R_s, R_p, L_s, L_p, tau_cav):
        """Helper to initialize the simulation."""
        return self.squeezing_class(
            pump_power=pump_power,
            simulation_time=simulation_time,
            N_points=N_points,
            R_s=R_s, R_p=R_p, L_s=L_s, L_p=L_p,
            tau_cav=tau_cav / 1000
        )
   
    def plot_gaussian(self, pump_power=1.0, simulation_time=50.0, N_points=200,
                      R_s=94.5, R_p=73.4, L_s=0.2, L_p=0.2, tau_cav=222e-3,
                      t_center=5, sigma=1, N_eigenvectors=1, noise=0):
        """Plot simulation with a Gaussian pulse."""
        self.N_eigenvectors = N_eigenvectors
        sim = self._initialize_sim(pump_power, simulation_time, N_points,
                                   R_s, R_p, L_s, L_p, tau_cav)
        sim.simulate_gaussian_pulse(t_center=t_center, sigma=sigma, noise=noise)
        self.plot_final(sim)

    def plot_square(self, pump_power=1.0, simulation_time=50.0, N_points=200,
                    R_s=94.5, R_p=73.4, L_s=0.2, L_p=0.2, tau_cav=222e-3,
                    start_pulse=3, pulse_length=5, N_eigenvectors=1):
        """Plot simulation with a square pulse."""
        self.N_eigenvectors = N_eigenvectors
        sim = self._initialize_sim(pump_power, simulation_time, N_points,
                                   R_s, R_p, L_s, L_p, tau_cav)
        sim.simulate_square_pulse(start_pulse=start_pulse, pulse_length=pulse_length)
        self.plot_final(sim)

    def plot_trapezoid(self, pump_power=1.0, simulation_time=50.0, N_points=200,
                       R_s=94.5, R_p=73.4, L_s=0.2, L_p=0.2, tau_cav=222e-3,
                       t_start=3, t_rise=2, t_plateau=3, t_fall=2,
                       N_eigenvectors=1, noise=0):
        """Plot simulation with a trapezoidal pulse."""
        self.N_eigenvectors = N_eigenvectors
        sim = self._initialize_sim(pump_power, simulation_time, N_points,
                                   R_s, R_p, L_s, L_p, tau_cav)
        sim.simulate_trapezoid_pulse(
            t_start=t_start, t_rise=t_rise, t_plateau=t_plateau, t_fall=t_fall, noise=noise
        )
        self.plot_final(sim)

    def plot_custom(self, pump_power=1.0, R_s=94.5, R_p=73.4, L_s=0.2, L_p=0.2,
                    tau_cav=222e-3, N_eigenvectors=1):
        """Plot simulation with a user-defined custom pulse."""
        self.N_eigenvectors = N_eigenvectors
        sim = self.squeezing_class(
            pump_power=pump_power,
            simulation_time=self.t[-1],
            N_points=len(self.custom_pulse),
            R_s=R_s, R_p=R_p, L_s=L_s, L_p=L_p,
            tau_cav=tau_cav / 1000
        )
        sim.simulate_custom_pulse(self.custom_pulse)
        self.plot_final(sim)

    def plot_final(self, sim):
        """Plot the final results including beta, eigenmodes, Schmidt numbers, and N-matrix."""
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        axs = axs.flatten()

        # Beta and input pulse
        axs[0].plot(sim.t, sim.beta * sim.pump_power / sim.betamax, label=r"$\beta(t)$")
        axs[0].plot(sim.t, sim.input_pulse, label=r"$\hat{B}_{\text{in}}$")
        axs[0].set_xlabel("Time [ns]")
        axs[0].legend()
        axs[0].grid(True)

        # Eigenvectors
        for i in range(self.N_eigenvectors):
            idx = np.argmax(np.abs(sim.eigenvectors[:, -i-1]) > 0.2 * sim.dt)
            if sim.eigenvectors[idx, -i-1] < 0:
                sim.eigenvectors[:, -i-1] *= -1
            axs[1].plot(sim.t, sim.eigenvectors[:, -i-1])
        axs[1].set_title("Eigenmodes")
        axs[1].set_xlim([0, sim.simulation_time])
        axs[1].set_xlabel("Time [ns]")
        axs[1].grid(True)

        # Eigenvalues (Schmidt numbers)
        axs[2].semilogy(sim.eigenvalues[::-1], color='blue')
        axs[2].set_xlim([0, len(sim.eigenvalues)-1])
        axs[2].set_xlabel("Schmidt Mode")
        axs[2].set_ylabel("Eigenvalue")
        axs[2].set_title(f"Schmidt Number = {sim.schmidt:.3f}\nAntisqueezing = {sim.antisqueezing:.3f} dB")

        # N-matrix
        axs[3].matshow(sim.N, origin='lower', extent=[sim.t[0], sim.t[-1], sim.t[0], sim.t[-1]])
        axs[3].xaxis.set_ticks_position("bottom")
        axs[3].set_xticks(np.arange(0, sim.simulation_time+1, 10))
        axs[3].set_yticks(np.arange(0, sim.simulation_time+1, 10))
        axs[3].set_xlabel(r"$t_1$ [ns]")
        axs[3].set_ylabel(r"$t_2$ [ns]")
        axs[3].set_title(r"$N(t_1, t_2)$")

        plt.tight_layout()
        plt.show()

    def interact_trapezoidal_pulse(self):
        interact(self.plot_trapezoid,
                 pump_power=widgets.FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0,
                                                description="Pump Power", 
                                                style={'description_width': '200px'},
                                                layout=widgets.Layout(width='450px')),
                 simulation_time=widgets.FloatSlider(min=10, max=200, step=1, value=50.0,
                                                    description="Simulation Time [ns]",
                                                    style={'description_width': '200px'},
                                                    layout=widgets.Layout(width='450px')),
                 N_points=widgets.IntSlider(min=100, max=10000, step=100, value=200,
                                            description="Simulation Points",
                                            style={'description_width': '200px'},
                                            layout=widgets.Layout(width='450px')),
                 R_s=widgets.FloatSlider(min=50, max=99.9, step=0.1, value=94.5,
                                         description="Signal Mirror Reflectivity [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 R_p=widgets.FloatSlider(min=0, max=99.9, step=0.1, value=73.4,
                                         description="Pump Mirror Reflectivity [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 L_s=widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.2,
                                         description="Signal Intra-cavity Loss [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 L_p=widgets.FloatSlider(min=0.0, max=10.0, step=0.01, value=0.2,
                                         description="Pump Intra-cavity Loss [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 tau_cav=widgets.FloatSlider(min=100, max=500, step=1, value=222,
                                             description="Cavity Round Trip [ps]",
                                             style={'description_width': '200px'},
                                             layout=widgets.Layout(width='450px')),
                 t_start=widgets.FloatSlider(min=0.0, max=20.0, step=0.1, value=3.0,
                                             description="Start Pulse [ns]",
                                             style={'description_width': '200px'},
                                             layout=widgets.Layout(width='450px')),
                 t_rise=widgets.FloatSlider(min=0.0, max=20.0, step=0.1, value=3.0,
                                            description="Rise Time [ns]",
                                            style={'description_width': '200px'},
                                            layout=widgets.Layout(width='450px')),
                 t_plateau=widgets.FloatSlider(min=0.0, max=200.0, step=0.1, value=3.0,
                                               description="Plateau Time [ns]",
                                               style={'description_width': '200px'},
                                               layout=widgets.Layout(width='450px')),
                 t_fall=widgets.FloatSlider(min=0.0, max=20.0, step=0.1, value=3.0,
                                            description="Fall Time [ns]",
                                            style={'description_width': '200px'},
                                            layout=widgets.Layout(width='450px')),
                 N_eigenvectors=widgets.IntSlider(min=1, max=10, step=1, value=1,
                                                  description="Eigenvectors Displayed",
                                                  style={'description_width': '200px'},
                                                  layout=widgets.Layout(width='450px')),
                 noise=widgets.FloatSlider(min=0, max=0.1, step=0.01, value=0,
                                           description="Gaussian Noise",
                                           style={'description_width': '200px'},
                                           layout=widgets.Layout(width='450px')))
        
    def interact_square_pulse(self):
        interact(self.plot_square,
                 pump_power=widgets.FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0,
                                                description="Pump Power", 
                                                style={'description_width': '200px'},
                                                layout=widgets.Layout(width='450px')),
                 simulation_time=widgets.FloatSlider(min=10, max=200, step=1, value=50.0,
                                                    description="Simulation Time [ns]",
                                                    style={'description_width': '200px'},
                                                    layout=widgets.Layout(width='450px')),
                 N_points=widgets.IntSlider(min=100, max=10000, step=100, value=200,
                                            description="Simulation Points",
                                            style={'description_width': '200px'},
                                            layout=widgets.Layout(width='450px')),
                 R_s=widgets.FloatSlider(min=50, max=99.9, step=0.1, value=94.5,
                                         description="Signal Mirror Reflectivity [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 R_p=widgets.FloatSlider(min=0, max=99.9, step=0.1, value=73.4,
                                         description="Pump Mirror Reflectivity [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 L_s=widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.2,
                                         description="Signal Intra-cavity Loss [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 L_p=widgets.FloatSlider(min=0.0, max=10.0, step=0.01, value=0.2,
                                         description="Pump Intra-cavity Loss [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 tau_cav=widgets.FloatSlider(min=100, max=500, step=1, value=222,
                                             description="Cavity Round Trip [ps]",
                                             style={'description_width': '200px'},
                                             layout=widgets.Layout(width='450px')),
                 start_pulse=widgets.FloatSlider(min=0.0, max=20.0, step=0.1, value=3.0,
                                             description="Start Pulse [ns]",
                                             style={'description_width': '200px'},
                                             layout=widgets.Layout(width='450px')),
                 pulse_length=widgets.FloatSlider(min=0.0, max=20.0, step=0.1, value=5.0,
                                            description="Pulse Length [ns]",
                                            style={'description_width': '200px'},
                                            layout=widgets.Layout(width='450px')),
                 N_eigenvectors=widgets.IntSlider(min=1, max=10, step=1, value=1,
                                            description="Eigenvectors Displayed",
                                            style={'description_width': '200px'},
                                            layout=widgets.Layout(width='450px')))
        
    def interact_custom_pulse(self, t, custom_pulse):
        self.t = t
        self.custom_pulse = custom_pulse
        interact(self.plot_custom,
                 pump_power=widgets.FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0,
                                                description="Pump Power", 
                                                style={'description_width': '200px'},
                                                layout=widgets.Layout(width='450px')),
                 R_s=widgets.FloatSlider(min=50, max=99.9, step=0.1, value=94.5,
                                         description="Signal Mirror Reflectivity [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 R_p=widgets.FloatSlider(min=0, max=99.9, step=0.1, value=73.4,
                                         description="Pump Mirror Reflectivity [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 L_s=widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.2,
                                         description="Signal Intra-cavity Loss [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 L_p=widgets.FloatSlider(min=0.0, max=10.0, step=0.01, value=0.2,
                                         description="Pump Intra-cavity Loss [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 tau_cav=widgets.FloatSlider(min=100, max=500, step=1, value=222,
                                             description="Cavity Round Trip [ps]",
                                             style={'description_width': '200px'},
                                             layout=widgets.Layout(width='450px')),
                 N_eigenvectors=widgets.IntSlider(min=1, max=10, step=1, value=1,
                                            description="Eigenvectors Displayed",
                                            style={'description_width': '200px'},
                                            layout=widgets.Layout(width='450px')))        

    def interact_gaussian_pulse(self):
        interact(self.plot_gaussian,
                 pump_power=widgets.FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0,
                                                description="Pump Power", 
                                                style={'description_width': '200px'},
                                                layout=widgets.Layout(width='450px')),
                 simulation_time=widgets.FloatSlider(min=10, max=200, step=1, value=50.0,
                                                    description="Simulation Time [ns]",
                                                    style={'description_width': '200px'},
                                                    layout=widgets.Layout(width='450px')),
                 N_points=widgets.IntSlider(min=100, max=10000, step=100, value=200,
                                            description="Simulation Points",
                                            style={'description_width': '200px'},
                                            layout=widgets.Layout(width='450px')),
                 R_s=widgets.FloatSlider(min=50, max=99.9, step=0.1, value=94.5,
                                         description="Signal Mirror Reflectivity [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 R_p=widgets.FloatSlider(min=0, max=99.9, step=0.1, value=73.4,
                                         description="Pump Mirror Reflectivity [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 L_s=widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.2,
                                         description="Signal Intra-cavity Loss [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 L_p=widgets.FloatSlider(min=0.0, max=10.0, step=0.01, value=0.2,
                                         description="Pump Intra-cavity Loss [%]",
                                         style={'description_width': '200px'},
                                         layout=widgets.Layout(width='450px')),
                 tau_cav=widgets.FloatSlider(min=100, max=500, step=1, value=222,
                                             description="Cavity Round Trip [ps]",
                                             style={'description_width': '200px'},
                                             layout=widgets.Layout(width='450px')),
                 t_center=widgets.FloatSlider(min=0.0, max=20.0, step=0.1, value=3.0,
                                             description="Center Pulse [ns]",
                                             style={'description_width': '200px'},
                                             layout=widgets.Layout(width='450px')),
                 sigma=widgets.FloatSlider(min=0.0, max=20.0, step=0.1, value=2.0,
                                            description="Gaussian STD [ns]",
                                            style={'description_width': '200px'},
                                            layout=widgets.Layout(width='450px')),
                 N_eigenvectors=widgets.IntSlider(min=1, max=10, step=1, value=1,
                                            description="Eigenvectors Displayed",
                                            style={'description_width': '200px'},
                                            layout=widgets.Layout(width='450px')),
                 noise=widgets.FloatSlider(min=0, max=0.1, step=0.01, value=0,
                                           description="Gaussian Noise",
                                           style={'description_width': '200px'},
                                           layout=widgets.Layout(width='450px')))     