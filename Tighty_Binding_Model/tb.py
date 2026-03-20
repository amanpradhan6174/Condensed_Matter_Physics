import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tabulate import tabulate

# ROOT OUTPUT DIRECTORY

BASE_DIR = Path("TB_assignment_outputs")
BASE_DIR.mkdir(exist_ok=True)


# HELPERS

def make_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def savefig_close(path, dpi=300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def estimate_zero_crossing(x, y):

   #Estimate x where y crosses 0 by linear interpolation. Returns np.nan if no sign change is found.
    
    y = np.asarray(y)
    x = np.asarray(x)
    s = np.sign(y)
    idx = np.where(np.diff(np.signbit(y)))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    x1, x2 = x[i], x[i + 1]
    y1, y2 = y[i], y[i + 1]
    if y2 == y1:
        return np.nan
    return x1 - y1 * (x2 - x1) / (y2 - y1)

def linear_gap(lower_band, upper_band):
    
  # Global band gap = min(upper) - max(lower)
  # Positive => insulator
  # Zero/negative => metal or band overlap
   
    return np.min(upper_band) - np.max(lower_band)

def band_summary(lower, upper):
    return {
        "lower_min": float(np.min(lower)),
        "lower_max": float(np.max(lower)),
        "upper_min": float(np.min(upper)),
        "upper_max": float(np.max(upper)),
        "gap": float(linear_gap(lower, upper))
    }
TOL = 1e-4
def classify_gap(g):
    if abs(g) < TOL:
        return "Critical"
    elif g > 0:
        return "Insulator"
    else:
        return "Metal"

def save_clean_table(data, csv_path, txt_path=None, extra_cols=None):
    df = pd.DataFrame(data)

    # Round numeric columns first
    df = df.round(6)

    # Add physics-based columns if needed
    if extra_cols:
        for col_name, func in extra_cols.items():
            df[col_name] = df.apply(func, axis=1)

    # Replace NaN after adding extra columns
    df = df.fillna("NaH")

    # Save CSV (real data file)
    df.to_csv(csv_path, index=False)

    # Save pretty grid table as text
    if txt_path is not None:
        grid_text = tabulate(df, headers="keys", tablefmt="grid", showindex=False)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(grid_text)

    return df

def find_critical_point(tp_vals, gaps):
    
   #Find approximate transition point where gap crosses zero
  
    for i in range(len(tp_vals)-1):
        if gaps[i] < 0 and gaps[i+1] > 0:
            return (tp_vals[i] + tp_vals[i+1]) / 2
    return None

# PART Q1: TWO-LEG LADDER

q1_dir = BASE_DIR / "Q1"
q1_plots = q1_dir / "plots"
make_dir(q1_plots)

# Use t > 0 as hopping magnitude
t1 = 1.0
k = np.linspace(-np.pi, np.pi, 1200)

# Sweep t' across the transition. For the ladder, the global gap opens around t' ~ 2t.
tp_values = np.linspace(0.0, 4.0, 81)
tp_samples = [0.0, 1.0, 2.0, 3.0]

def q1_bands(tp):
    """
    Q1 ladder Hamiltonian:
    H(k) = [[-2t cos k, -tp],
            [-tp,      -2t cos k]]
    Bands: E = -2t cos k ± tp
    """
    eps = -2.0 * t1 * np.cos(k)
    lower = eps - tp
    upper = eps + tp
    return lower, upper

q1_rows = []
q1_gap_vals = []

# Plot sample bands
plt.figure(figsize=(8, 5.5))
for tp in tp_samples:
    lower, upper = q1_bands(tp)
    plt.plot(k, lower, label=f"t' = {tp:.1f}")
    plt.plot(k, upper)
plt.axhline(0, color="k", lw=0.7)
plt.xlabel(r"$k$")
plt.ylabel("Energy")
plt.title("Q1: Two-leg ladder band structure")
plt.legend()
savefig_close(q1_plots / "q1_band_samples.png")

# Gap scan
for tp in tp_values:
    lower, upper = q1_bands(tp)
    gap = linear_gap(lower, upper)

    q1_gap_vals.append(gap)

    q1_rows.append({
        "tp": tp,
        "gap": gap,
        "phase": classify_gap(gap),
        "abs_gap": abs(gap),
        "lower_min": np.min(lower),
        "lower_max": np.max(lower),
        "upper_min": np.min(upper),
        "upper_max": np.max(upper)
    })

plt.figure(figsize=(7, 5))
plt.plot(tp_values, q1_gap_vals, marker="o", ms=3)
plt.axhline(0, color="k", ls="--", lw=1)
plt.xlabel(r"$t'$")
plt.ylabel("Global gap")
plt.title("Q1: gap vs $t'$")
plt.grid(alpha=0.3)
savefig_close(q1_plots / "q1_gap_vs_tp.png")

df=save_clean_table(
    q1_rows,
    q1_dir / "q1_gap_scan.csv",
    q1_dir / "q1_gap_scan.txt",
)


q1_crit = estimate_zero_crossing(tp_values, q1_gap_vals)

min_gap = np.min(q1_gap_vals)
max_gap = np.max(q1_gap_vals)

q1_crit_text = (
    f"Q1 ANALYSIS: TWO-LEG LADDER MODEL\n"
    f"----------------------------------\n\n"

    f"Model Description:\n"
    f"- System: Two-leg ladder\n"
    f"- Nearest-neighbour hopping along chain: t = {t1:.2f}\n"
    f"- Inter-chain hopping: t'\n\n"

    f"Numerical Results:\n"
    f"- Minimum gap observed: {min_gap:.4f}\n"
    f"- Maximum gap observed: {max_gap:.4f}\n"
    f"- Estimated critical point: t' ≈ {q1_crit:.4f}\n\n"

    f"Band Structure Interpretation:\n"
    f"- Energy dispersion: E(k) = -2t cos(k) ± t'\n"
    f"- Increasing t' splits bonding and antibonding bands\n\n"

    f"Phase Behaviour:\n"
    f"- For t' < {q1_crit:.2f}: bands overlap → METALLIC phase\n"
    f"- At t' ≈ {q1_crit:.2f}: gap closes → CRITICAL point\n"
    f"- For t' > {q1_crit:.2f}: finite gap → INSULATING phase\n\n"

    f"Physical Insight:\n"
    f"- Inter-chain coupling enhances localization\n"
    f"- The transition occurs when t' ≈ 2t\n"
    f"- No re-entrant phase is observed (single transition)\n\n"

    f"Conclusion:\n"
    f"- The ladder system exhibits a single metal–insulator transition\n"
    f"- Controlled entirely by inter-chain hopping strength\n"
)

with open(q1_dir / "analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write(q1_crit_text)


# PART Q2: DIAGONAL HOPPING

q2_dir = BASE_DIR / "Q2"
q2_plots = q2_dir / "plots"
make_dir(q2_plots)

t2 = 1.0
tpp = 0.2   # must be smaller than both |t| and |t'|
tp_values_q2 = np.linspace(0.0, 4.0, 81)
tp_samples_q2 = [0.5, 1.5, 2.5, 3.5]

def q2a_bands(tp, tpp):
    
   #Q2(a): one diagonal per plaquette.
   #Off-diagonal Bloch term:   -tp - tpp e^{-ik}
   #Hermiticity gives conjugate in lower-left.
   #Bands: E = eps ± |tp + tpp e^{-ik}|
    
    eps = -2.0 * t2 * np.cos(k)
    off_mag = np.abs(tp + tpp * np.exp(-1j * k))
    lower = eps - off_mag
    upper = eps + off_mag
    return lower, upper

def q2b_bands(tp, tpp):
    
   #Q2(b): both diagonals. Off-diagonal Bloch term:  -tp - 2 tpp cos(k)
   #Bands:  E = eps ± |tp + 2 tpp cos(k)|

    eps = -2.0 * t2 * np.cos(k)
    off_mag = np.abs(tp + 2.0 * tpp * np.cos(k))
    lower = eps - off_mag
    upper = eps + off_mag
    return lower, upper

# Q2(a)
q2a_rows = []
q2a_gap_vals = []

plt.figure(figsize=(8, 5.5))
for tp in tp_samples_q2:
    lower, upper = q2a_bands(tp, tpp)
    plt.plot(k, lower, label=f"t' = {tp:.1f}")
    plt.plot(k, upper)
plt.axhline(0, color="k", lw=0.7)
plt.xlabel(r"$k$")
plt.ylabel("Energy")
plt.title(r"Q2(a): single diagonal hopping $t''$")
plt.legend()
savefig_close(q2_plots / "q2a_band_samples.png")

for tp in tp_values_q2:
    lower, upper = q2a_bands(tp, tpp)
    gap = linear_gap(lower, upper)
    q2a_gap_vals.append(gap)
    q2a_rows.append({
        "tp": tp,
        "tpp": tpp,
        "gap": gap,
        "phase": classify_gap(gap),
        "abs_gap": abs(gap),
        "lower_min": np.min(lower),
        "lower_max": np.max(lower),
        "upper_min": np.min(upper),
        "upper_max": np.max(upper)
    })

plt.figure(figsize=(7, 5))
plt.plot(tp_values_q2, q2a_gap_vals, marker="o", ms=3)
plt.axhline(0, color="k", ls="--", lw=1)
plt.xlabel(r"$t'$")
plt.ylabel("Global gap")
plt.title(r"Q2(a): gap vs $t'$ with single diagonal hopping")
plt.grid(alpha=0.3)
savefig_close(q2_plots / "q2a_gap_vs_tp.png")

df=save_clean_table(  
    q2a_rows,
    q2_dir / "q2a_gap_scan.csv",
    q2_dir / "q2a_gap_scan.txt", 
)

# Q2(b)
q2b_rows = []
q2b_gap_vals = []

plt.figure(figsize=(8, 5.5))
for tp in tp_samples_q2:
    lower, upper = q2b_bands(tp, tpp)
    plt.plot(k, lower, label=f"t' = {tp:.1f}")
    plt.plot(k, upper)
plt.axhline(0, color="k", lw=0.7)
plt.xlabel(r"$k$")
plt.ylabel("Energy")
plt.title(r"Q2(b): crossed diagonals (both $t''$ paths)")
plt.legend()
savefig_close(q2_plots / "q2b_band_samples.png")

for tp in tp_values_q2:
    lower, upper = q2b_bands(tp, tpp)
    gap = linear_gap(lower, upper)
    q2b_gap_vals.append(gap)
    q2b_rows.append({
        "tp": tp,
        "tpp": tpp,
        "gap": gap,
        "phase": classify_gap(gap),
        "abs_gap": abs(gap),
        "lower_min": np.min(lower),
        "lower_max": np.max(lower),
        "upper_min": np.min(upper),
        "upper_max": np.max(upper)
    })

plt.figure(figsize=(7, 5))
plt.plot(tp_values_q2, q2a_gap_vals, marker="o", ms=3, label="Q2(a)")
plt.plot(tp_values_q2, q2b_gap_vals, marker="s", ms=3, label="Q2(b)")
plt.axhline(0, color="k", ls="--", lw=1)
plt.xlabel(r"$t'$")
plt.ylabel("Global gap")
plt.title(r"Q2: effect of diagonal hopping $t''$")
plt.legend()
plt.grid(alpha=0.3)
savefig_close(q2_plots / "q2_gap_comparison.png")

df=save_clean_table(
    q2b_rows,
    q2_dir / "q2b_gap_scan.csv",
    q2_dir / "q2b_gap_scan.txt",
)


q2_crit_a = estimate_zero_crossing(tp_values_q2, q2a_gap_vals)
q2_crit_b = estimate_zero_crossing(tp_values_q2, q2b_gap_vals)

min_gap_a = np.min(q2a_gap_vals)
max_gap_a = np.max(q2a_gap_vals)

min_gap_b = np.min(q2b_gap_vals)
max_gap_b = np.max(q2b_gap_vals)

q2_text = (
    f"Q2 ANALYSIS: EFFECT OF DIAGONAL HOPPING\n\n\n"


    f"Model Description:\n"
    f"- Base system: Two-leg ladder\n"
    f"- Nearest-neighbour hopping: t = {t1:.2f}\n"
    f"- Inter-chain hopping: t'\n"
    f"- Diagonal hopping: t'' = {tpp:.2f}\n\n"

    f"Numerical Results:\n"
    f"Q2(a): Single diagonal hopping\n"
    f"- Minimum gap: {min_gap_a:.4f}\n"
    f"- Maximum gap: {max_gap_a:.4f}\n"
    f"- Critical point: t' ≈ {q2_crit_a:.4f}\n\n"

    f"Q2(b): Crossed diagonal hopping\n"
    f"- Minimum gap: {min_gap_b:.4f}\n"
    f"- Maximum gap: {max_gap_b:.4f}\n"
    f"- Critical point: t' ≈ {q2_crit_b:.4f}\n\n"

    f"Comparison with Q1:\n"
    f"- In Q1, transition occurs at t' ≈ 2t\n"
    f"- Addition of t'' shifts the transition point\n"
    f"- The shift depends on geometry of diagonal hopping\n\n"

    f"Physical Interpretation:\n"
    f"- Q2(a):\n"
    f"  * Off-diagonal term includes complex phase (e^ik)\n"
    f"  * Breaks inversion symmetry\n"
    f"  * Leads to asymmetric band dispersion\n\n"

    f"- Q2(b):\n"
    f"  * Off-diagonal term ~ cos(k)\n"
    f"  * Preserves symmetry\n"
    f"  * Modifies bandwidth without phase asymmetry\n\n"

    f"Key Insight:\n"
    f"- Diagonal hopping alters electronic coupling pathways\n"
    f"- Geometry of hopping strongly influences band structure\n"
    f"- Both cases still show a single metal–insulator transition\n\n"

    f"Conclusion:\n"
    f"- Presence of t'' shifts and modifies the transition\n"
    f"- Q2(a) and Q2(b) behave differently due to symmetry differences\n"
)

with open(q2_dir / "analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write(q2_text)

# PART Q3: SQUARE LATTICE

q3a_dir = BASE_DIR / "Q3a"
q3b_dir = BASE_DIR / "Q3b"
q3a_plots = q3a_dir / "plots"
q3b_plots = q3b_dir / "plots"
make_dir(q3a_plots)
make_dir(q3b_plots)

t_sq = -2.0
Nk2 = 300
kx2 = np.linspace(-np.pi, np.pi, Nk2)
ky2 = np.linspace(-np.pi, np.pi, Nk2)
KX, KY = np.meshgrid(kx2, ky2)

fillings = np.arange(0.5, 2.0001, 0.25)  # 7 fillings: 0.5 to 2.0 inclusive

def E_nn(kx, ky):
    """
    Q3(a): nearest-neighbour square lattice.
    E(k) = -2t(cos kx + cos ky)
    """
    return -2.0 * t_sq * (np.cos(kx) + np.cos(ky))

def E_nnn(kx, ky, td):
    """
    Q3(b): nearest + next-nearest neighbour square lattice.
    E(k) = -2t(cos kx + cos ky) - 4td cos kx cos ky
    """
    return -2.0 * t_sq * (np.cos(kx) + np.cos(ky)) - 4.0 * td * np.cos(kx) * np.cos(ky)

def fermi_energy_from_filling(E_sorted, filling):

   #filling = electrons per unit cell in [0,2]
   #Because of spin degeneracy, the occupied fraction of states is filling/2.

    N = len(E_sorted)
    idx = int((filling / 2.0) * N)
    idx = max(0, min(idx, N - 1))
    return float(E_sorted[idx])

def save_fermi_plot(E, Ef, title, filename, note_full_band=False):
    
   #Save a contour map with the Fermi contour overlaid.

    plt.figure(figsize=(6.8, 5.8))
    cf = plt.contourf(KX, KY, E, levels=40)
    plt.colorbar(cf, label="Energy (eV)")

    if np.min(E) < Ef < np.max(E):
        cs = plt.contour(KX, KY, E, levels=[Ef], colors="red", linewidths=2)
        plt.clabel(cs, inline=True, fontsize=8, fmt={Ef: "Ef"})
    else:
        plt.text(
            0.04, 0.96,
            "Fermi contour collapses\nat band edge",
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round")
        )
        if note_full_band:
            plt.scatter([0], [0], s=35, c="red")
            plt.text(0.05, 0.05, r"$\Gamma$", transform=plt.gca().transAxes, fontsize=10)

    plt.xlabel(r"$k_x$")
    plt.ylabel(r"$k_y$")
    plt.title(title)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    savefig_close(filename)


# Q3(a): only nearest neighbour
E0 = E_nn(KX, KY)
E0_sorted = np.sort(E0.flatten())

q3a_rows = []
Ef_half_q3a = np.nan

for n in fillings:
    Ef = fermi_energy_from_filling(E0_sorted, n)
    if np.isclose(n, 1.0):
        Ef_half_q3a = Ef
    plot_file = q3a_plots / f"q3a_n_{n:.2f}.png"
    save_fermi_plot(
        E0, Ef,
        title=f"Q3(a): NN only, filling = {n:.2f} e/unit cell",
        filename=plot_file,
        note_full_band=True
    )
    q3a_rows.append({
        "filling": n,
        "fermi_energy_eV": Ef,
        "band_min_eV": float(np.min(E0)),
        "band_max_eV": float(np.max(E0)),
        "filling_fraction_of_states": n / 2.0,
        "plot_file": str(plot_file)
    })

df=save_clean_table(
    q3a_rows,
    q3a_dir / "q3a_fermi_levels.csv",
    q3a_dir / "q3a_fermi_levels.txt",
    extra_cols={
        "state_fraction": lambda row: row["filling"]/2
    }
)

E_min = np.min(E0)
E_max = np.max(E0)

q3a_text = (
    f"Q3(a) ANALYSIS: SQUARE LATTICE (NEAREST-NEIGHBOUR)\n\n\n"


    f"Model Description:\n"
    f"- Square lattice\n"
    f"- Nearest-neighbour hopping only\n"
    f"- Hopping parameter: t = {t_sq:.2f} eV\n\n"

    f"Band Structure:\n"
    f"- Energy range: [{E_min:.4f}, {E_max:.4f}] eV\n"
    f"- Dispersion: E(k) = -2t (cos kx + cos ky)\n\n"

    f"Numerical Results:\n"
    f"- Fermi energy at half filling (n = 1.0): Ef ≈ {Ef_half_q3a:.4f} eV\n\n"

    f"Filling Dependence:\n"
    f"- Filling n represents electrons per unit cell (max = 2 due to spin)\n"
    f"- Occupied fraction of states = n/2\n\n"

    f"Fermi Surface Evolution:\n"
    f"- At low filling (n ≈ 0.5): small electron pockets\n"
    f"- At intermediate filling: contour expands across Brillouin zone\n"
    f"- At half filling (n = 1): symmetric Fermi surface\n"
    f"- Near full filling (n → 2): contour shrinks toward band edge\n\n"

    f"Physical Interpretation:\n"
    f"- The system exhibits particle-hole symmetry\n"
    f"- Fermi energy at half filling lies near band center\n"
    f"- No distortion or asymmetry without next-nearest hopping\n\n"

    f"Conclusion:\n"
    f"- Square lattice with nearest-neighbour hopping shows symmetric band structure\n"
    f"- Fermi surface evolves smoothly with filling\n"
    f"- Serves as reference for comparison with Q3(b)\n"
)

with open(q3a_dir / "analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write(q3a_text)

# Q3(b): nearest + next-nearest neighbour

q3b_rows = []
td_values = [-1.0, 1.0]
Ef_half_q3b = {}

for td in td_values:
    subdir = q3b_plots / f"td_{td:+.1f}"
    make_dir(subdir)

    E1 = E_nnn(KX, KY, td)
    E1_sorted = np.sort(E1.flatten())

    for n in fillings:
        Ef = fermi_energy_from_filling(E1_sorted, n)
        if np.isclose(n, 1.0):
            Ef_half_q3b[td] = Ef

        plot_file = subdir / f"q3b_td_{td:+.1f}_n_{n:.2f}.png"
        save_fermi_plot(
            E1, Ef,
            title=f"Q3(b): t_diag = {td:+.1f} eV, filling = {n:.2f} e/unit cell",
            filename=plot_file,
            note_full_band=True
        )
        q3b_rows.append({
            "t_diag_eV": td,
            "filling": n,
            "fermi_energy_eV": Ef,
            "band_min_eV": float(np.min(E1)),
            "band_max_eV": float(np.max(E1)),
            "filling_fraction_of_states": n / 2.0,
            "plot_file": str(plot_file)
        })

df=save_clean_table(
    q3b_rows,
    q3b_dir / "q3b_fermi_levels.csv",
    q3b_dir / "q3b_fermi_levels.txt",
    extra_cols={
        "state_fraction": lambda row: row["filling"]/2
    }
)


E_min_b = np.min(E1)
E_max_b = np.max(E1)

Ef_neg = Ef_half_q3b.get(-1.0, np.nan)
Ef_pos = Ef_half_q3b.get(1.0, np.nan)

q3b_text = (
    f"Q3(b) ANALYSIS: EFFECT OF NEXT-NEAREST-NEIGHBOUR HOPPING\n\n\n"

    f"Model Description:\n"
    f"- Square lattice with nearest and next-nearest neighbour hopping\n"
    f"- Nearest-neighbour hopping: t = {t_sq:.2f} eV\n"
    f"- Next-nearest neighbour hopping: t'' = ±1.0 eV\n\n"

    f"Band Structure:\n"
    f"- Energy range: [{E_min_b:.4f}, {E_max_b:.4f}] eV\n"
    f"- Dispersion: E(k) = -2t (cos kx + cos ky) - 4t'' cos kx cos ky\n\n"

    f"Numerical Results:\n"
    f"- Half-filling Fermi energy (t'' = -1.0 eV): Ef ≈ {Ef_neg:.4f} eV\n"
    f"- Half-filling Fermi energy (t'' = +1.0 eV): Ef ≈ {Ef_pos:.4f} eV\n\n"

    f"Comparison with Q3(a):\n"
    f"- Q3(a) had particle-hole symmetry\n"
    f"- Inclusion of t'' breaks this symmetry\n"
    f"- Fermi level shifts away from band center\n\n"

    f"Fermi Surface Evolution:\n"
    f"- t'' > 0:\n"
    f"  * Fermi surface becomes more rounded\n"
    f"  * Band curvature increases\n\n"

    f"- t'' < 0:\n"
    f"  * Fermi surface becomes more square-like\n"
    f"  * Strong anisotropy appears\n\n"

    f"Physical Interpretation:\n"
    f"- Next-nearest hopping introduces diagonal electronic pathways\n"
    f"- This modifies band curvature and density of states\n"
    f"- Breaks particle-hole symmetry of the lattice\n\n"

    f"Topological Insight:\n"
    f"- Change in Fermi surface shape indicates possible Lifshitz transition\n"
    f"- Electronic properties strongly depend on sign of t''\n\n"

    f"Conclusion:\n"
    f"- Inclusion of t'' significantly alters electronic structure\n"
    f"- Sign of t'' controls Fermi surface geometry\n"
    f"- System shows richer physics compared to nearest-neighbour model\n"
)

with open(q3b_dir / "analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write(q3b_text)


print(f"All outputs saved under: {BASE_DIR.resolve()}")
