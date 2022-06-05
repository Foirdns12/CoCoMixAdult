import os
from demonstration.proxy_measures.analyze import analyze_foils, instantiate_all_measures
from demonstration.competing_approaches.wachter.util import calculate_mad
from demonstration.demonstration_data import load_df, FEATURES, VAR_TYPES
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as tick

def load_foils():
    PATH = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(PATH, 'daten', '20210325-18181745_foils.json'), "rt") as f:
        record = json.load(f)
    for filename in os.listdir(os.path.join(PATH, 'daten')):
        print(filename)
        if filename != '20210325-18181745_foils.json':
            with open(os.path.join(PATH, 'daten', filename), "rt") as f:
                record += json.load(f)
    return record

if __name__ == "__main__":
    df_train = load_df(train=True)
    PATH = os.path.dirname(os.path.abspath(__file__))
    correctness=True
    everythingelse=True
    factdist=False
    catspars=True
    mad = calculate_mad(df_train[[feature for feature, var_type in zip(FEATURES, VAR_TYPES)
                                  if var_type == "c"]].to_numpy())
    mad[mad == 0.0] = 0.5
    measures = instantiate_all_measures(mad)
    record=load_foils()
    x1 = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25]
    x2 = [0, 1, 2, 2.5, 3, 3.5, 4, 5]
    #x1=[0.5,1]
    #x2=[0,1.5]
    #factset=["0bbc5dd87e39ec3cc43ba862cc7a8b43"]
    sparsitymeanfin = []
    closenessmeanfin = []
    feasibility1meanfin = []
    feasibility1bmeanfin = []
    feasibility2meanfin = []
    typicalitymeanfin = []
    catdistancemeanfin = []
    catdistancebmeanfin = []
    correctnessmeanfin = []
    catsparsmeanfin = []
    for mu in x1:
        sparsitymean = []
        closenessmean = []
        feasibility1mean = []
        feasibility1bmean = []
        feasibility2mean = []
        typicalitymean = []
        catdistancemean = []
        catdistancebmean = []
        correctnessmean = []
        catsparsmean = []
        for alpha in x2:
            configuration = {
                "lambda_": 120.0,
                "mu": mu,
                "alpha": alpha,
                "beta": 0.5,
                "budget": 1000,
                "densitycut": 8,
                "densityaddloss": 1.5,
                "densityscaler": 0.05
            }
            conf2 = json.dumps(configuration)
            full_set = []
            for data in record:
                if data['conf'] == conf2:
                    # if data['factid'] in factset:
                    full_set.append((data['fact'], data['foil'], np.array(data['history']["pdf"]), 4, 6))
            print('start analysis')
            analysis = analyze_foils(full_set, measures)
            print('passed')
            if everythingelse:
                sparsitymean.append(analysis['sparsity']['mean'])
                closenessmean.append(analysis['cocomix_distance']['mean'])
                feasibility1mean.append(analysis['density_opt_step_min']['mean'])
                feasibility1bmean.append(analysis['density_opt_step_max']['mean'])
                feasibility2mean.append(analysis['density_min_start']['mean'])
                typicalitymean.append(analysis['density_foil']['mean'])
                catdistancemean.append(analysis['cat_distance']['mean'])
                catdistancebmean.append(analysis['cat_distance_naive']['mean'])
            if correctness:
                correctnessmean.append(analysis['correctness']['mean'])
            if catspars:
                catsparsmean.append(analysis['cat_sparsity']['mean'])
        if everythingelse:
            sparsitymeanfin.append(sparsitymean)
            closenessmeanfin.append(closenessmean)
            feasibility1meanfin.append(feasibility1mean)
            feasibility1bmeanfin.append(feasibility1bmean)
            feasibility2meanfin.append(feasibility2mean)
            typicalitymeanfin.append(typicalitymean)
            catdistancemeanfin.append(catdistancemean)
            catdistancebmeanfin.append(catdistancebmean)
        if correctness:
            correctnessmeanfin.append(correctnessmean)
        if catspars:
            catsparsmeanfin.append(catsparsmean)
    if everythingelse:
        print('sparsity')
        print(np.array(sparsitymeanfin))
        print('closeness')
        print(np.array(closenessmeanfin))
        print(np.array(feasibility2meanfin))
        print(np.array(feasibility1meanfin))
        print(np.array(feasibility1bmeanfin))
        print('typicality')
        print(np.array(typicalitymeanfin))
        print('fesibility 3 cat dist')
        print(np.array(catdistancemeanfin))
        print(np.array(catdistancebmeanfin))
    if correctness:
        print('correctness')
        print(np.array(correctnessmeanfin))
    if catspars:
        print('cat_sparsity')
        print(np.array(catsparsmeanfin))
    X, Y = np.meshgrid(x1, x2)

    if everythingelse:
        #plot sparsity

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, np.array(sparsitymeanfin).transpose(), cmap=cm.Greys,
                              linewidth=0, antialiased=False)
        #Customize the z axis.
        ax.set_zlim(0, 9)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        #Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        #plt.savefig(os.path.join(PATH, 'plotsparsity.png'))
        plt.show()

        #plot closeness

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, np.array(closenessmeanfin).transpose(), cmap=cm.Greys,
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        #plt.savefig(os.path.join(PATH, 'plotcloseness.png'))
        plt.show()


        #Plot Feasibility 2

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        surf = ax.plot_surface(X, Y, np.array(feasibility2meanfin).transpose(), cmap=cm.Greys,
                               linewidth=0, antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        #plt.savefig(os.path.join('plotfeasibility2.png'))

        # Plot Feasibility 1

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, np.array(feasibility1meanfin).transpose(), cmap=cm.Greys,
                               linewidth=0, antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        ax.view_init(elev=10., azim=120)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        #plt.savefig(os.path.join('plotfeasibility1.png'))


        # Plot Feasibility 1

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, np.array(feasibility1meanfin).transpose(), cmap=cm.Greys,
                               linewidth=0, antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Plot Feasibility 1b

        surf = ax.plot_surface(X, Y, np.array(feasibility1bmeanfin).transpose(), cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        # Add a color bar which maps values to colors.
        ax.set_zlim(0, 80)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        #plt.savefig(os.path.join('plotfeasibility1b.png'))
        plt.show()


        #Plot Typicality
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, np.array(typicalitymeanfin).transpose(), cmap=cm.Greys,
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        #plt.savefig(os.path.join('plottypicality.png'))
        plt.show()

        # Plot Feasibility3 cat dist
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, np.array(catdistancemeanfin).transpose(), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.savefig(os.path.join('plottypicality.png'))
        plt.show()


        # Plot Feasibility3b cat dist
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, np.array(catdistancebmeanfin).transpose(), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.savefig(os.path.join('plottypicality.png'))
        plt.show()


    #Plot Correctness
    if correctness:
        # Plot correctness
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, np.array(correctnessmeanfin).transpose(), cmap=cm.Greys,
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.savefig(os.path.join('plottypicality.png'))
        plt.show()


    # if factdist:
    #     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #     surf = ax.plot_surface(X, Y, np.array(factdistmeanfin).transpose(), cmap=cm.coolwarm,
    #                            linewidth=0, antialiased=False)
    #     ax.zaxis.set_major_locator(LinearLocator(10))
    #     # Add a color bar which maps values to colors.
    #     fig.colorbar(surf, shrink=0.5, aspect=5)
    #     plt.savefig(os.path.join('plotfactdist.png'))
    #     #plt.show()
    if catspars:
        # Plot correctness

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, np.array(catsparsmeanfin).transpose(), cmap=cm.Greys,
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.set_zlim(0.15, 0.35)
        ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.savefig(os.path.join('plottypicality.png'))
        plt.show()
