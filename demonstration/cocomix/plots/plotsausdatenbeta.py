import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import os
import matplotlib.ticker as tick

if __name__ == "__main__":
    x1 = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25]
    x2 = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
    X, Y = np.meshgrid(x1, x2)
    print(X)
    print(Y)

    sparsitymeanfin=[[8.,   7.44, 7.12 ,6.84 ,6.56, 6.12, 6.24, 6.16],
 [7.2,  6.96, 6.12, 5.8,  5.84 ,6.  , 5.4 , 5.4 ],
 [6.68 ,6.12, 5.76 ,5.6,  5.12, 5.12, 5.36, 5.  ],
 [6.52, 5.84 ,5.84 ,5.28, 5.08, 5.16 ,4.6 , 4.56],
 [5.96 ,5.6,  5.16 ,4.72 ,4.72 ,4.4 , 4.48 ,4.24],
 [6. ,  5.28, 4.44, 4.72 ,4.6 , 4.32 ,4.08, 4.16],
 [5.48, 5.32, 4.28, 3.84, 4.04, 4.16, 3.84, 3.56],
 [5.4 , 4.76 ,4.12 ,3.84, 4.16, 3.76, 3.6,  3.44]]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.array(sparsitymeanfin).transpose(), cmap=cm.Greys,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(0, 8)
    ax.zaxis.set_major_locator(LinearLocator(9))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.0f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(os.path.join(PATH, 'plotsparsity.png'))
    plt.show()


    closenessmeanfin=[[38.66090751, 30.64094069, 25.39845713, 24.38376971, 20.74018585, 17.95499687,
  18.40055281, 15.78882317],
 [32.77193578, 27.32007021, 19.01524195 ,16.64140621, 15.35077229, 16.32290264,
  13.85302482, 13.59541558],
 [29.70169463, 20.54428392, 18.12591921, 16.27563131 ,13.63168294 ,13.30990557,
  13.2126663,  11.60920487],
 [25.65150195, 20.89949627, 17.70917334, 14.91841327, 12.21538101, 13.79147411,
  10.00722847 , 9.45666722],
 [24.02933601, 18.35783225, 15.94099272, 11.16112007, 11.00897641,  8.74695653,
   8.45749641 , 7.57688393],
 [22.71532154 ,16.53080752 ,10.69808267 ,10.73552287, 10.69052461 , 9.01002964,
   7.83496494 , 6.74856807],
 [21.5565146 , 17.11854269 ,11.56701521,  9.00854987 , 8.1994449 ,  6.68265061,
   7.28032995, 5.55273673],
 [20.19942831, 15.37257835, 10.56940456 , 9.38603172 , 8.34504286,  6.67045577,
   4.94041312 , 4.24131715]]


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.array(closenessmeanfin).transpose(), cmap=cm.Greys,
                           linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(9))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.0f'))
    ax.set_zlim(4, 40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(os.path.join(PATH, 'plotcloseness.png'))
    plt.show()


    typicalitymeanfin=[[2.71669504e-15, 1.40512776e-15 ,1.36150781e-15, 1.98469195e-15,
  1.96612102e-15, 2.40735998e-15 ,8.98618864e-16, 1.15283906e-15],
 [1.66304446e-15 ,2.16672122e-15, 1.40812922e-15, 9.76096009e-16,
  9.95901028e-16, 1.41500948e-15, 1.08596500e-15, 1.48654323e-15],
 [1.50838489e-15, 1.29798451e-15 ,1.20831284e-15 ,1.57884024e-15,
  9.17350604e-16 ,1.15274628e-15 ,9.92485464e-16 ,7.91012649e-16],
 [1.19655455e-15, 1.32978095e-15, 7.07954282e-16, 1.41656503e-15,
  7.98567010e-16, 9.13855796e-16 ,6.08450608e-16, 1.16764553e-15],
 [1.06823296e-15, 1.36846001e-15 ,1.17518664e-15 ,8.97770790e-16,
  1.12764338e-15 ,9.34162356e-16 ,6.31230663e-16 ,7.74651992e-16],
 [1.10910282e-15 ,1.03581314e-15 ,7.72288534e-16 ,7.82897528e-16,
  7.14786719e-16, 8.98154770e-16, 7.62934911e-16 ,7.63928230e-16],
 [1.22900346e-15, 7.38163028e-16, 7.97050113e-16 ,8.27927060e-16,
  7.91241811e-16, 6.71623197e-16, 8.90184479e-16, 1.15745033e-15],
 [1.27616317e-15, 8.73758895e-16 ,7.62941749e-16, 8.52195316e-16,
  8.33374563e-16, 8.81342200e-16 ,1.31598102e-15, 7.82439007e-16]]



    # Plot Typicality
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.array(typicalitymeanfin).transpose()*1000000000000000, cmap=cm.Greys,
                       linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(8))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
    ax.set_zlim(0, 3.5)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(os.path.join('plottypicality.png'))
    plt.show()



    feasibility2meanfin=[[0.36246557, 0.32449985, 0.39590586, 0.45834464 ,0.5218267 , 0.54781634,
  0.47806868, 0.45423024],
 [0.40726278, 0.47989884, 0.64939865, 0.7065398,  0.54087893,0.54891464,
  0.60400808, 0.61430942],
 [0.4475158 , 0.59981148, 0.57382657, 0.5456217,  0.54295057 ,0.53407964,
  0.630112 ,  0.5971262 ],
 [0.54686005, 0.4572245,  0.5162846,  0.59546496, 0.61174017, 0.65850003,
  0.45811832, 0.56928443],
 [0.5500713,  0.66898008, 0.61187425 ,0.70292284 ,0.59034182,0.62860642,
  0.58085279, 0.51663173],
 [0.43889138 ,0.58874665 ,0.62756257, 0.50699967 ,0.6945446 , 0.6103503,
  0.60378303, 0.66464729],
 [0.49482024, 0.58186234, 0.60715303, 0.62687244, 0.68761348 ,0.5850479,
  0.64594179, 0.65235221],
 [0.62750206, 0.61687617 ,0.68625357 ,0.61031356 ,0.6828693 , 0.64733366,
  0.72475039, 0.64706457]]


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, np.array(feasibility2meanfin).transpose(), cmap=cm.Greys,
                       linewidth=0, antialiased=False)
    ax.set_zlim(0, 0.8)
    ax.zaxis.set_major_locator(LinearLocator(9))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


    feasibility1meanfin= [[0.80507115, 0.83215749, 0.7888062 , 0.76841466, 0.7060514  ,0.7117623,
  0.77446515 ,0.745085  ],
 [0.80382957, 0.76307313 ,0.64579284, 0.66885125 ,0.68094436, 0.71885455,
  0.67931497, 0.67971629],
 [0.7730857,  0.65615305 ,0.69839711, 0.65461581,0.64129561, 0.68961937,
  0.67346944, 0.66165469],
 [0.73232281, 0.7201314 , 0.70481298, 0.696943,   0.67937365 ,0.66971466,
  0.64831047, 0.6076129 ],
 [0.73822367, 0.62956212 ,0.66410224, 0.62435121, 0.66111139, 0.56224907,
  0.69386239, 0.70881448],
 [0.75450732, 0.70708772, 0.62141376, 0.68211256, 0.58876936, 0.56344528,
  0.62525783, 0.5425446 ],
 [0.64592723, 0.75834687, 0.62690189 ,0.62370534 ,0.51357816 ,0.67384404,
  0.59007175, 0.59552328],
 [0.64326252, 0.62279823 ,0.58566433, 0.61225206 ,0.56222349 ,0.59658046,
  0.42882118,0.53770703]]



    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.array(feasibility1meanfin).transpose(), cmap=cm.Greys,
                       linewidth=0, antialiased=False)
    ax.set_zlim(0.44, 1)
    ax.zaxis.set_major_locator(LinearLocator(9))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    correctnessmeanfin=[[1.,   0.96, 0.88, 0.92, 0.88 ,0.88, 0.88 ,0.88],
 [0.92, 0.88, 0.88, 0.84, 0.84, 0.88 ,0.8 , 0.84],
 [1.  , 0.84, 0.88, 0.84 ,0.72 ,0.8  ,0.8 , 0.8 ],
 [0.8 , 0.88, 0.88 ,0.8  ,0.72 ,0.84 ,0.68 ,0.68],
 [0.88, 0.88, 0.84, 0.72, 0.76, 0.64 ,0.6,  0.56],
 [0.84, 0.88, 0.72, 0.76, 0.72, 0.68 ,0.6 , 0.52],
 [0.84 ,0.88, 0.72, 0.68, 0.6, 0.56, 0.56 ,0.44],
 [0.88 ,0.84, 0.72, 0.68, 0.56 ,0.52 ,0.4,  0.36]]




    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.array(correctnessmeanfin).transpose(), cmap=cm.Greys,
                       linewidth=0, antialiased=False)

    ax.set_zlim(0.3, 1)
    ax.zaxis.set_major_locator(LinearLocator(8))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(os.path.join('plottypicality.png'))
    plt.show()


    catsparsmeanfin=[[0.44818903, 0.40571429, 0.31590476, 0.30850794, 0.28285714, 0.25739683,
  0.24566667, 0.23047619],
 [0.4011746 , 0.36296825, 0.29093651, 0.27666667, 0.22480952, 0.22906349,
  0.19733333, 0.18666667],
 [0.41131746, 0.31411111 ,0.28536508, 0.26114286, 0.20738095, 0.18290476,
  0.19209524 ,0.16666667],
 [0.38857143,0.32507937 ,0.26373016, 0.23880952, 0.1872381 , 0.20209524,
  0.15380952, 0.12238095],
 [0.32828571, 0.31449206 ,0.27907937 ,0.18904762, 0.18590476 ,0.16,
  0.11542857 ,0.11704762],
 [0.32947619, 0.27431746, 0.1887619 , 0.1827619 , 0.17447619, 0.16380952,
  0.14866667 ,0.10933333],
 [0.33442857, 0.29839683, 0.22 ,      0.17142857 ,0.13209524, 0.088,
  0.12533333, 0.09      ],
 [0.34552381 ,0.29480952 ,0.21714286 ,0.188 ,     0.15266667, 0.122,
  0.07866667, 0.054     ]]




    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.array(catsparsmeanfin).transpose(), cmap=cm.Greys,
                       linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_zlim(0, 0.45)
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(os.path.join('plottypicality.png'))
    plt.show()