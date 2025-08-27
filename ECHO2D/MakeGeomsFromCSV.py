#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Helpers & simple data structs ----------
def cleave(x, sig=5):
    """Round a number to a specific precision (keeps geom.txt tidy)."""
    return round(x, -int(np.log(np.abs(x))) + sig) if float(x) != 0.0 else x

class SimpleBoundary:
    def __init__(self, zs=None, rs=None):
        self.zs = np.asarray(zs if zs is not None else [], dtype=float)
        self.rs = np.asarray(rs if rs is not None else [], dtype=float)

class SimpleLayer:
    def __init__(self, name="Layer", ep=1.0, mu=1.0, sg=0.0):
        self.name, self.ep, self.mu, self.sg = name, ep, mu, sg
        self.top: SimpleBoundary = None
        self.bot: SimpleBoundary = None
def offset_boundary(boundary: SimpleBoundary, t: float) -> SimpleBoundary:
    """
    Offset the curve *t* meters normal to itself, using an arc‑length parameterization.
    """
    zs = boundary.zs
    rs = boundary.rs

    # 1) compute segment lengths and cumulative s
    dz = np.diff(zs)
    dr = np.diff(rs)
    ds = np.hypot(dz, dr)
    s  = np.concatenate(([0], np.cumsum(ds)))  # s[i] = distance from point 0 to i

    # 2) compute derivatives wrt s
    dz_ds = np.gradient(zs, s)
    dr_ds = np.gradient(rs, s)

    # 3) normalized tangent is already unit length
    tz, tr = dz_ds, dr_ds

    # 4) rotate tangent by +90° → outward normal
    nz = -tr
    nr =  tz

    # 5) offset points
    new = SimpleBoundary()
    new.zs = zs + t * nz
    new.rs = rs + t * nr
    return new


def add_wall_legs(wall: SimpleLayer, leg_factor=1.2, leg_length=0.005):
    """
    Append two 'legs' to wall.bot to close the chamber for the ECHO input.
    """
    z_end, r_end = wall.bot.zs[-1], wall.bot.rs[-1]
    r_ext = leg_factor * r_end
    wall.bot.zs = np.concatenate([wall.bot.zs, [z_end, z_end + leg_length]])
    wall.bot.rs = np.concatenate([wall.bot.rs, [r_ext, r_ext]])

def write_geom_txt(bounds, outpath):
    """
    Write geom.txt from bounds[0]=wall and bounds[1:]=dielectrics.
    Assumes wall.bot.zs/rs already include the potential extension legs.
    """
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    with open(outpath, 'w') as f:
        f.write(f"% Number of materials\n{len(bounds)}\n")

        # --- WALL (open chain) ---
        wall = bounds[0]
        zc = [cleave(z) for z in wall.bot.zs]
        rc = [cleave(r) for r in wall.bot.rs]

        f.write("% Number of elements in metal with conductive walls, permeability, permitivity, cond.\n")
        f.write(f"{len(zc)-1} {cleave(wall.ep)} {cleave(wall.mu)} {cleave(wall.sg)}\n")
        f.write("% Segments of lines and elipses with wall conductivity\n")
        for i in range(len(zc)-1):
            f.write(
                f"{zc[i]:.6e} {rc[i]:.6e} "
                f"{zc[i+1]:.6e} {rc[i+1]:.6e} "
                "0 0 0 0 0 "
                f"{wall.sg:.6g}\n"
            )

        # --- DIELECTRICS (closed loops) ---
        for layer in bounds[1:]:
            z_loop = np.concatenate((layer.top.zs, layer.bot.zs[::-1]))
            r_loop = np.concatenate((layer.top.rs, layer.bot.rs[::-1]))
            zc = [cleave(z) for z in z_loop]
            rc = [cleave(r) for r in r_loop]

            f.write("% Number of elements in metal with conductive walls, permeability, permitivity, cond.\n")
            f.write(f"{len(zc)} {cleave(layer.ep)} {cleave(layer.mu)} {cleave(layer.sg)}\n")
            f.write("% Segments of lines and elipses with wall conductivity\n")
            for i in range(len(zc)):
                j = (i + 1) % len(zc)
                f.write(
                    f"{zc[i]:.6e} {rc[i]:.6e} "
                    f"{zc[j]:.6e} {rc[j]:.6e} "
                    "0 0 0 0 0 "
                    f"{layer.sg:.6g}\n"
                )
    print(f"Wrote geometry → {outpath}")

def plot_geometry(bounds, out_txt):
    """
    Plot wall (with its legs) and dielectric loops, save as PNG.
    """
    plt.figure(figsize=(8,4))

    # Wall
    w = bounds[0]
    plt.plot(w.bot.zs, w.bot.rs, 'b-', lw=2, label='Chamber Wall (+legs)')

    # Only one dielectric assumed here; extend if you add more
    d = bounds[1]
    z_d = np.concatenate((d.top.zs, d.bot.zs[::-1]))
    r_d = np.concatenate((d.top.rs, d.bot.rs[::-1]))
    plt.plot(z_d, r_d, 'orange', lw=2, label='Dielectric Loop',alpha=0.7,linestyle='--')

    plt.xlabel("Z (m)")
    plt.ylabel("R (m)")
    plt.title("Extended Geometry")
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

    png = os.path.splitext(out_txt)[0] + ".png"
    plt.tight_layout()
    plt.savefig(png, dpi=300)
    plt.close()
    print(f"Saved geometry plot → {png}")

def sanitize_k_for_folder(k_val):
    """Create a filesystem-friendly folder name from a float."""
    return f"k_{k_val:.6g}"

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser(
        description="Build a geom.txt for every k column in a radius CSV (wide format)."
    )
    p.add_argument("--csv", help="CSV with 'z (m)' and columns named like 'k=...' (radius profiles).", default='figsLinear/linear_taper_profiles.csv')
    p.add_argument("--thickness", type=float, default=60e-6, help="Dielectric thickness (m)")
    p.add_argument("--er", type=float, default=3.81, help="Dielectric relative permittivity")
    p.add_argument("--outdir", default=".", help="Base output directory")
    p.add_argument("--wall_sigma", type=float, default=0.0, help="Wall conductivity")
    p.add_argument("--wall_ep", type=float, default=1.0, help="Wall permittivity")
    p.add_argument("--wall_mu", type=float, default=1.0, help="Wall permeability")
    p.add_argument("--leg_factor", type=float, default=1.2, help="Wall extension radius factor")
    p.add_argument("--leg_length", type=float, default=0.035, help="Wall extension length (m)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if 'z (m)' not in df.columns:
        raise ValueError("CSV must contain a 'z (m)' column.")
    z = df['z (m)'].values

    # instead of looking for “k=” columns, just take every column except z
    slope_cols = [c for c in df.columns if c != 'z (m)']

    # regex to parse “in” or “out” and the slope number
    pattern = re.compile(r'^slope=([0-9eE+\-\.]+)$')

    for col in slope_cols:
        m = pattern.match(col)
        if not m:
            print(f"Skipping column '{col}' (doesn’t match in/out_β=…)")
            continue

        slope_str = m.group(1)
        slope = float(slope_str)
        mode = 'linear'

        # read the radius profile
        r_bot = df[col].values.astype(float)
        r_top = r_bot + args.thickness

        # build the metal wall boundary first
        wall = SimpleLayer("Chamber Wall", ep=args.wall_ep,
                        mu=args.wall_mu, sg=args.wall_sigma)
        wall.bot = SimpleBoundary(zs=z, rs=r_top) # wall only needs the r coordinate of the top of the waveguide

        # now make the dielectric layer by offsetting the wall outward by thickness
        diel = SimpleLayer("Dielectric Layer", ep=args.er, mu=1.0, sg=0.0)
        diel.top = SimpleBoundary(zs=wall.bot.zs.copy(), rs=wall.bot.rs.copy())
        diel.bot = offset_boundary(wall.bot, -args.thickness) # a bit overkill but this makes sure the thickness is perpendicular to the metal


        add_wall_legs(wall, leg_factor=args.leg_factor,
                    leg_length=args.leg_length) # add legs after the simulation area to simulate transition to vacuum
        bounds = [wall, diel]

        # sanitize folder name: e.g. "in_beta_0.316227"
        # slope = slope_str.lstrip('+').replace('.', 'p')  # e.g. "0p316227"
        slope = slope_str
        subdir = os.path.join(args.outdir, f"{mode}_beta_{slope}")
        os.makedirs(subdir, exist_ok=True)

        out_txt = os.path.join(subdir, "geom.txt")
        write_geom_txt(bounds, out_txt)
        plot_geometry(bounds, out_txt)

        # dump data as before
        np.savez(out_txt.replace(".txt","_data.npz"),
                wall_z=wall.bot.zs, wall_r=wall.bot.rs,
                top_z=diel.top.zs, top_r=diel.top.rs,
                bot_z=diel.bot.zs, bot_r=diel.bot.rs,
                mode=mode, slope=slope)
        print(f"Done {mode} slope={slope}")

if __name__ == "__main__":
    main()
