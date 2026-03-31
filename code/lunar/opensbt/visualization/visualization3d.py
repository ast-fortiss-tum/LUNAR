from pathlib import Path
import numpy as np
from pymoo.visualization.scatter import Scatter
import os
import pandas as pd
from opensbt.model_ga.individual import Individual
from opensbt.model_ga.population import Population, PopulationExtended
from opensbt.utils.sorting import get_nondominated_population
from opensbt.utils.duplicates import duplicate_free

from opensbt.visualization.visualizer import color_critical, color_optimal, color_not_critical, color_not_optimal
import matplotlib.colors as mcolors


def _ensure_color(c):
    """Convert any color spec to a proper RGBA tuple.
    handles bare numpy scalars (greyscale floats) that newer matplotlib rejects."""
    if isinstance(c, (float, int, np.floating, np.integer)):
        # treat scalar as a greyscale level in [0, 1]
        c = (float(c), float(c), float(c))
    return mcolors.to_rgba(c)



def read_testcases(filename):
    individuals = []
    table = pd.read_csv(filename)
    var_names = []
    n_var = -1
    k = 0
    # identify number of objectives
    for col in table.columns[1:]:
        if col.startswith("Fitness_"):
            n_var = k
            break
        var_names.append(col)
        k = k + 1
    for i in range(len(table)):
        X = table.iloc[i, 1:n_var + 1].to_numpy()
        F = table.iloc[i, n_var + 1:-2].to_numpy()
        CB = table.iloc[i, -1]
        ind = Individual()
        ind.set("X", X)
        ind.set("F", F)
        ind.set("CB", CB)
        individuals.append(ind)
    
    print("Csv file successfully read")
    return Population(individuals=individuals), var_names

def visualize_3d(population, 
                save_folder, 
                labels, 
                mode="critical", 
                markersize=20, 
                do_save=False,
                dimension="X",
                angles=[(45,-45),
                        (45,45),
                        (45,135),
                        (45,225),
                        (0,0),
                        (0,90),
                        (0,180),
                        (0,270)],
                show=False):
    """This function generated 3D plots of executed test inputs."""

    save_folder_design = save_folder + "3d" + os.sep

    n_var = 3

    pop = duplicate_free(population)

    if not isinstance(pop, PopulationExtended):
        pop = PopulationExtended(pop)

    print(f"[visualization3d] Number all: {len(pop)}")

    opt = get_nondominated_population(pop)
    print(f"[visualization3d] Number optimal: {len(opt)}")

    crit, _ = pop.divide_critical_non_critical()
    print(f"[visualization3d] Number critical: {len(crit)}")

    X = pop.get(dimension)
    CB = np.array(pop.get("CB"),dtype=bool)
    X_opt = opt.get(dimension)
    CB_opt = np.array(opt.get("CB"),dtype=bool)

    X_plus_opt = np.ma.masked_array(X_opt, mask=np.dstack([np.invert(CB_opt)] * n_var))
    X_minus_opt = np.ma.masked_array(X_opt, mask=np.dstack([CB_opt] * n_var))

    mask_plus = np.invert(CB)
    # print(f"mask_plus_len:{len(mask_plus)}")

    mask_minus = CB
    # print(f"mask_minus_len:{len(mask_minus)}")

    X_plus = np.ma.masked_array(X, mask=np.dstack([mask_plus] * n_var))
    # print(X_plus)

    X_minus = np.ma.masked_array(X, mask=np.dstack([mask_minus] * n_var))
    # print(X_minus)
    if dimension == "F":
        title = 'Objective_Space'
    elif dimension == "X":
        title = 'Design_Space'
    else:
        title = '3D_Space'
    
    # Normalize colors to proper RGBA tuples
    fc_not_optimal = _ensure_color(color_not_optimal)
    fc_optimal = _ensure_color(color_optimal)
    ec_critical = _ensure_color(color_critical)
    ec_not_critical = _ensure_color(color_not_critical)

    if do_save:
        Path(save_folder_design).mkdir(parents=True, exist_ok=True)   
        for angle in angles:
            plot_des = Scatter(title=title, labels = list(labels), angle=angle)
            points_added = False
            if np.ma.count(X_plus, axis=0)[0] != 0:
                plot_des.add(X_plus, facecolor=fc_not_optimal, edgecolor=ec_critical, s=markersize)
                if mode == 'all' and np.ma.count(X_minus, axis=0)[0] != 0:
                        plot_des.add(X_minus, facecolor=fc_not_optimal, edgecolor=ec_not_critical, s=markersize)
                points_added = True

            if mode != "critical" and np.ma.count(X_minus_opt, axis=0)[0] != 0:
                plot_des.add(X_minus_opt, facecolor=fc_optimal, edgecolor=ec_not_critical, s=markersize)
                points_added = True
                    
            if (mode=="opt" or mode== "all") and np.ma.count(X_plus_opt, axis=0)[0] != 0:
                print("added optimal and critical")
                plot_des.add(X_plus_opt, facecolor=fc_optimal, edgecolor=ec_critical, s=markersize)
                points_added = True
            
            # Need this check, otherwise an exception is thrown, when there is nothing to be plotted
            if points_added:
                if show:
                    plot_des.show()
                plot_des.save(save_folder_design + f"{title}_3d_angle{angle}.png")
                plot_des.save(save_folder_design + f"{title}_3d_angle{angle}.pdf", format="pdf")

    else:
        plot_des = Scatter(title=title, labels = list(labels))
        if np.ma.count(X_plus, axis=0)[0] != 0:
            plot_des.add(X_plus, facecolor=fc_not_optimal, edgecolor=ec_critical, s=markersize)
            if mode == 'all':
                if np.ma.count(X_minus, axis=0)[0] != 0:
                    plot_des.add(X_minus, facecolor=fc_not_optimal, edgecolor=ec_not_critical, s=markersize)

        if mode != "critical":    
            if np.ma.count(X_minus_opt, axis=0)[0] != 0:
                    plot_des.add(X_minus_opt, facecolor=fc_optimal, edgecolor=ec_not_critical, s=markersize)
    
        if mode=="opt" or mode== "all":
            if np.ma.count(X_plus_opt, axis=0)[0] != 0:
                print("added optimal and critical")
                plot_des.add(X_plus_opt, facecolor=fc_optimal, edgecolor=ec_critical, s=markersize)

        plot_des.show()
