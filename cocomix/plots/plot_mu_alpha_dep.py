#from demonstration.cocomix.compute_foil_immo import calculate_foils
from demonstration.cocomix.compute_foil_adult import calculate_foils
from demonstration.cocomix.compute_adult_cocomix import mad,df_test,df_train,model,transition_matrices,distance_matrices
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import os
import datetime


if __name__ == "__main__":
    PATH = os.path.dirname(os.path.abspath(__file__))
    X = np.arange(0, 3, 0.5)
    Y = np.arange(0, 2, 0.5)
    X, Y = np.meshgrid(X, Y)
    print(X)
    print(Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    print(Z)
    x1=[0.3,0.6,0.9,1]
    x2=[0,1]
    X, Y = np.meshgrid(x1, x2)
    print(X)
    print(Y)
    Z=X*Y
    print(Z)
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
     #                      linewidth=0, antialiased=False)


    sparsitymean=[]
    sparsitymin=[]
    sparsitymax=[]
    closenessmean=[]
    closenessmin=[]
    closenessmax=[]
    feasibility2mean=[]
    feasibility2min = []
    feasibility2max = []
    typicalitymean=[]
    typicalitymin=[]
    typicalitymax=[]
    #x=[0,0.5,1,1.5,2,2.5]
    m=False
    if m:
        x = [0,0.5,1,1.5,2,2.5]
        for mu in x:
            configuration = {
                "lambda_": 120.0,
                "mu": mu,
                "alpha": 3.0,
                "beta": 0.5,
                "budget": 1000,
                "densitycut": 8,
                "densityaddloss": 1.5,
                "densityscaler": 0.05
            }
            result,_ = calculate_foils(configuration,mad,df_test,model,transition_matrices,distance_matrices,df_train,n=25,factset=None,randomstate=60,metrics=True,boundaries=True)
            analysis=result['joint_analysis']
            sparsitymean.append(analysis['sparsity']['mean'])
            sparsitymin.append(analysis['sparsity']['min'])
            sparsitymax.append(analysis['sparsity']['max'])
            closenessmean.append(analysis['cocomix_distance']['mean'])
            closenessmin.append(analysis['cocomix_distance']['min'])
            closenessmax.append(analysis['cocomix_distance']['max'])
            feasibility2mean.append(analysis['density_min_start']['mean'])
            feasibility2min.append(analysis['density_min_start']['min'])
            feasibility2max.append(analysis['density_min_start']['max'])
            typicalitymean.append(analysis['density_foil']['mean'])
            typicalitymin.append(analysis['density_foil']['min'])
            typicalitymax.append(analysis['density_foil']['max'])
    alph=False
    if alph:
        x = [1, 2, 3, 4, 5, 6]
        for alpha in x:
            configuration = {
                "lambda_": 120.0,
                "mu": 1.2,
                "alpha": alpha,
                "beta": 0.5,
                "budget": 1000,
                "densitycut": 8,
                "densityaddloss": 1.5,
                "densityscaler": 0.05
            }
            result, _ = calculate_foils(configuration,mad,df_test,model,transition_matrices,distance_matrices,df_train,n=25,factset=None,randomstate=60,metrics=True,boundaries=True)
            analysis = result['joint_analysis']
            sparsitymean.append(analysis['sparsity']['mean'])
            sparsitymin.append(analysis['sparsity']['min'])
            sparsitymax.append(analysis['sparsity']['max'])
            closenessmean.append(analysis['cocomix_distance']['mean'])
            closenessmin.append(analysis['cocomix_distance']['min'])
            closenessmax.append(analysis['cocomix_distance']['max'])
            feasibility2mean.append(analysis['density_min_start']['mean'])
            feasibility2min.append(analysis['density_min_start']['min'])
            feasibility2max.append(analysis['density_min_start']['max'])
            typicalitymean.append(analysis['density_foil']['mean'])
            typicalitymin.append(analysis['density_foil']['min'])
            typicalitymax.append(analysis['density_foil']['max'])
    if m or alph:
        print("sparsity mean: ",sparsitymean)
        print(sparsitymin)
        print(sparsitymax)
        print(closenessmean)
        print(closenessmin)
        print(closenessmax)
        print(feasibility2mean)
        print(feasibility2min)
        print(feasibility2max)
        print(typicalitymean)
        print(typicalitymin)
        print(typicalitymax)
        plt.plot(x, sparsitymean)
        plt.plot(x, sparsitymin)
        plt.plot(x, sparsitymax)
        plt.xlabel("$mu$")
        plt.ylabel("Sparsity bei alpha=3")
        plt.show()


        plt.plot(x, closenessmean)
        plt.plot(x, closenessmin)
        plt.plot(x, closenessmax)
        plt.xlabel("$mu$")
        plt.ylabel("closeness bei alpha=3")
        plt.show()

        plt.plot(x, feasibility2mean)
        plt.plot(x, feasibility2min)
        plt.plot(x, feasibility2max)
        plt.xlabel("$mu$")
        plt.ylabel("feasibility2 bei alpha=3")
        plt.show()




        plt.plot(x, typicalitymean)
        plt.plot(x, typicalitymin)
        plt.plot(x, typicalitymax)
        plt.xlabel("$mu$")
        plt.ylabel("typicality bei alpha=3")
        plt.show()





    p3 = True
    if p3:
        import json
        import numpy as np


        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return super(NpEncoder, self).default(obj)
        foilsettot=[]
        sparsitymeanfin = []
        closenessmeanfin = []
        correctnessmeanfin = []
        feasibility1meanfin = []
        feasibility1bmeanfin = []
        feasibility2meanfin = []
        typicalitymeanfin = []
        x1=[1,1.5,2,2.5,3]
        x2=[0,1,2,2.5,3,3.5,4,5]
        #x1 = [1, 1.5]
        #x2 = [1,2]
        for mu in x1:
            print('##################################################')
            print("mu: ", mu)
            print('##################################################')
            sparsitymean = []
            closenessmean = []
            correctnessmean= []
            feasibility1mean = []
            feasibility1bmean = []
            feasibility2mean = []
            typicalitymean = []
            for alpha in x2:
                print('##################################################')
                print("alpha: ", alpha)
                print('##################################################')
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
                result, foilset = calculate_foils(configuration,mad,df_test,model,transition_matrices,distance_matrices,df_train,n=25,factset=None,randomstate=60,metrics=True,boundaries=True)
                foilsettot=foilsettot+foilset
                analysis = result['joint_analysis']
                sparsitymean.append(analysis['sparsity']['mean'])
                #sparsitymean.append(alpha*mu)
                closenessmean.append(analysis['cocomix_distance']['mean'])
                #feasibility1mean.append(analysis['density_opt_step_min']['mean'])
                #feasibility1bmean.append(analysis['density_opt_step_max']['mean'])
                #feasibility2mean.append(analysis['density_min_start']['mean'])
                typicalitymean.append(analysis['density_foil']['mean'])
                correctnessmean.append(analysis['correctness']['mean'])
            sparsitymeanfin.append(sparsitymean)
            closenessmeanfin.append(closenessmean)
            #feasibility1meanfin.append(feasibility1mean)
            #feasibility1bmeanfin.append(feasibility1bmean)
            #feasibility2meanfin.append(feasibility2mean)
            typicalitymeanfin.append(typicalitymean)
            correctnessmeanfin.append(correctnessmean)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        X, Y = np.meshgrid(x1, x2)
        print(np.array(sparsitymeanfin))
        print(np.array(closenessmeanfin))
        #print(np.array(feasibility1meanfin))
        #print(np.array(feasibility1bmeanfin))
        #print(np.array(feasibility2meanfin))
        print(np.array(typicalitymeanfin))
        print(np.array(correctnessmeanfin))

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        fnamejsn = f"{current_time}_foils.json"
        with open(fnamejsn, "wt") as f:
            json.dump(foilsettot, f, indent=4, cls=NpEncoder)
        # Plot the surface.
        surf = ax.plot_surface(X, Y, np.array(sparsitymeanfin), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(1, 10)
        #plt.show()
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        #ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(os.path.join(PATH,'plotsparsity.png'))
        plt.show()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        surf = ax.plot_surface(X, Y, np.array(closenessmeanfin), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # plt.show()
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        # ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(os.path.join(PATH,'plotcloseness.png'))
        plt.show()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        #surf = ax.plot_surface(X, Y, np.array(feasibility2meanfin), cmap=cm.coolwarm,
        #                       linewidth=0, antialiased=False)

        # plt.show()
        #ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        # ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
       # fig.colorbar(surf, shrink=0.5, aspect=5)
       # plt.savefig(os.path.join('plotfeasibility2.png'))

        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        #surf = ax.plot_surface(X, Y, np.array(feasibility1meanfin), cmap=cm.coolwarm,
        #                       linewidth=0, antialiased=False)

        # plt.show()
        #ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        # ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        #plt.savefig(os.path.join('plotfeasibility1.png'))

        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        #surf = ax.plot_surface(X, Y, np.array(feasibility1bmeanfin), cmap=cm.coolwarm,
        #                       linewidth=0, antialiased=False)

        # plt.show()
        #ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        # ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        #plt.savefig(os.path.join('plotfeasibility1b.png'))



        #plt.show()
        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        #surf = ax.plot_surface(X, Y, np.array(typicalitymeanfin), cmap=cm.coolwarm,
        #                       linewidth=0, antialiased=False)
        # plt.show()
        #ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        # ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)


        #plt.savefig(os.path.join('plottypicality.png'))
        #plt.show()