import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import os
import matplotlib.ticker as tick

if __name__ == "__main__":
    x1 = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
    x2 = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
    X, Y = np.meshgrid(x1, x2)
    print(X)
    print(Y)

    sparsitymeanfin=[[6.88, 5.88, 5.6,  4.88, 4.88 ,4.84 ,4.56, 4.32],
 [5.44 ,4.76, 4.72 ,4.6 , 3.92, 4.08 ,3.92 ,3.68],
 [5.24, 4.28, 4. ,  3.6  ,3.48 ,3.56, 3.48 ,3.44],
 [4.88 ,3.92, 3.64, 3.2 , 3.32, 3.4 , 3.16 ,3.08],
 [4.4 , 3.52 ,3.28, 3.12 ,3.36, 2.72, 2.88, 2.96],
 [4.04 ,3.52 ,2.96 ,3.12, 2.8 , 2.52, 2.68, 2.32],
 [4.24, 3.36, 3.  , 2.64, 2.12 ,2.44, 2.24, 2.12],
 [3.92, 3.52, 2.92, 2.36 ,2.2 , 2.16, 2.  , 2.  ],
 [3.92 ,3.04, 2.28 ,2.16 ,1.88, 1.8 , 1.76 ,1.8 ]]


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

    x1 = [0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
    x2 = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
    X, Y = np.meshgrid(x1, x2)

    closenessmeanfin = [[8.51985590e+03, 4.98229292e+03, 2.61343026e+03, 2.77976900e+03,
  1.39151516e+03, 2.22140161e+03 ,3.92896552e+02, 1.07257671e+03],
 [5.01734374e+03, 1.24061999e+03, 3.69540307e+03 ,1.38508603e+03,
  2.44541875e+03, 8.99838007e+01, 5.30605028e+01, 3.39069153e+01],
 [3.64291400e+03, 3.85033982e+03, 3.60028218e+03 ,2.20005222e+03,
  1.07487117e+02, 3.44096081e+01, 3.44486436e+01, 1.59675988e+01],
 [4.45533569e+03, 2.50452449e+03, 3.51690941e+03, 4.02505898e+02,
  1.36844255e+03, 1.21863434e+01, 1.11243206e+01, 2.74036404e+02],
 [5.77323760e+03, 3.66455936e+03, 1.75673094e+03, 7.20836699e+01,
  3.36003281e+01, 1.11514766e+01, 2.72068099e+02, 8.46581195e+00],
 [3.66465730e+03, 1.20728482e+03, 7.20351746e+01, 3.09517540e+02,
  1.10541013e+01, 9.36400612e+00, 7.75305329e+00, 7.61875660e+00],
 [5.93212862e+03, 6.50411189e+02, 1.18062002e+03, 2.96522610e+01,
  8.55562402e+00, 6.54096705e+00 ,6.46422630e+00, 6.18980916e+00],
 [3.74327774e+03, 4.14938044e+02, 1.15760201e+03, 1.26619701e+01,
  5.91405290e+00 ,5.42220659e+00 ,5.28089958e+00 ,4.91545808e+00]]


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.array(closenessmeanfin).transpose(), cmap=cm.Greys,
                           linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(8))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.0f'))
    ax.set_zlim(0, 5000)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(os.path.join(PATH, 'plotcloseness.png'))
    plt.show()







    x1 = [2, 2.25, 2.5]
    x2 = [0.625, 0.75, 0.875, 1]
    X, Y = np.meshgrid(x1, x2)
    closenessmeanfin=[ [ 1.10541013e+01 ,9.36400612e+00 ,7.75305329e+00, 7.61875660e+00],
 [  8.55562402e+00 ,6.54096705e+00 ,6.46422630e+00, 6.18980916e+00],
 [ 5.91405290e+00 ,5.42220659e+00, 5.28089958e+00, 4.91545808e+00]]



    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.array(closenessmeanfin).transpose(), cmap=cm.Greys,
                           linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(8))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.0f'))
    ax.set_zlim(0, 14)
    ax.set_ylim(0.625,1)
    ax.yaxis.set_major_locator(LinearLocator(4))
    ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(os.path.join(PATH, 'plotcloseness.png'))
    plt.show()

    x1 = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
    x2 = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
    X, Y = np.meshgrid(x1, x2)

    typicalitymeanfin=[[3.54249606e-17, 1.00029581e-18 ,2.27749600e-18, 1.20596092e-18,
  1.28605753e-17, 1.37708287e-18 ,5.78220482e-19, 1.10565076e-17],
 [8.07526783e-17, 3.30553785e-17, 1.13039841e-17, 8.67109535e-17,
  1.27319269e-17, 2.18115134e-18 ,9.33554045e-18, 1.44427198e-16],
 [4.15608786e-17, 1.38375777e-17, 7.71069064e-17 ,1.53405811e-17,
  1.69508957e-17, 3.78857028e-17, 1.91970992e-17, 8.52821811e-17],
 [4.93081954e-17, 8.65204032e-18 ,1.27936694e-17, 5.00886121e-17,
  2.49304340e-17, 9.86846421e-17, 1.15995517e-17, 5.06501725e-17],
 [1.20805513e-16, 8.74035343e-17 ,7.03728278e-17 ,2.11286007e-16,
  4.54322783e-17, 8.78220577e-18, 4.98323547e-17 ,1.80939733e-17],
 [2.47554314e-16, 5.11811412e-17, 1.51107461e-17, 5.46331455e-17,
  4.31349587e-17, 3.33664353e-17, 1.05271990e-16, 2.88121032e-17],
 [6.52555787e-17, 2.51343284e-17, 6.23305822e-17, 2.22548124e-16,
  2.46571309e-17, 1.45699704e-17, 3.71763084e-17, 1.77775584e-17],
 [6.96499014e-17, 4.84387576e-17, 7.55613931e-17 ,4.27499841e-17,
  3.25468879e-17, 2.18267923e-17, 2.04269041e-17, 3.49882323e-17],
 [4.45312934e-17, 1.89106800e-17, 3.43273184e-17 ,1.65651780e-17,
  3.07999045e-17, 3.36064061e-17, 3.58853885e-17 ,2.30874644e-17]]




    # Plot Typicality
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.array(typicalitymeanfin).transpose()*1000000000000000, cmap=cm.Greys,
                       linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(8))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    ax.set_zlim(0, 0.35)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(os.path.join('plottypicality.png'))
    plt.show()



    feasibility2meanfin=[[0.00091328, 0.00359995 ,0.01295773 ,0.01172615 ,0.02511137, 0.00134768,
  0.00153526, 0.00290986],
 [0.06690695 ,0.02639783 ,0.09494688 ,0.00224735, 0.05292895, 0.04662074,
  0.0348164  ,0.06101418],
 [0.05673537 ,0.07136709 ,0.1145639 , 0.079289 ,  0.06267486 ,0.11381869,
  0.03225934 ,0.01165255],
 [0.07760582, 0.03252054, 0.10433465, 0.19262289 ,0.1169311 , 0.10622168,
  0.07526529, 0.16029281],
 [0.10206615, 0.13922103, 0.14317704, 0.0928614 , 0.16291552 ,0.14182585,
  0.18930475 ,0.12472206],
 [0.22527746, 0.14818416, 0.13755057, 0.11131887, 0.14907224 ,0.20245125,
  0.14377953, 0.19443619],
 [0.06691418 ,0.12938701 ,0.1476722 , 0.0177371,  0.0973984 , 0.12601878,
  0.07604776, 0.0935527 ],
 [0.27275978, 0.15432533, 0.13524874, 0.04337014, 0.1303882,  0.12314026,
  0.10778999, 0.09738565],
 [0.22013944 ,0.16610288 ,0.05214334, 0.10597611 ,0.10029146 ,0.09202533,
  0.02120139, 0.06559381]]



    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, np.array(feasibility2meanfin).transpose(), cmap=cm.Greys,
                       linewidth=0, antialiased=False)
    ax.set_zlim(0, 0.4)
    ax.zaxis.set_major_locator(LinearLocator(9))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


    feasibility1meanfin= [[0.98020751 ,0.98793145, 0.97918724 ,0.98163519 ,0.96417558, 0.9856826,
  0.99373273, 0.97370203],
 [0.91616357 ,0.9450265  ,0.95652027 ,0.97092855, 0.93157531, 0.96445351,
  0.95647383, 0.92897624],
 [0.89438304, 0.90690958 ,0.88012165, 0.88379208 ,0.97155848 ,0.89563656,
  0.96333857, 0.96110954],
 [0.8916183  ,0.91356264, 0.90202524 ,0.84955984 ,0.89746629 ,0.91308729,
  0.94882639 ,0.88620079],
 [0.85775585 ,0.8742168 , 0.85393975 ,0.95963547 ,0.91096856, 0.89486209,
  0.88884097 ,0.93407906],
 [0.79000283 ,0.80810147, 0.865005 ,  0.89308207 ,0.85705049 ,0.83234417,
  0.93286568 ,0.84305267],
 [0.89701921 ,0.87130817, 0.83598458 ,0.88779985 ,0.75763548 ,0.86493802,
  0.83764875 ,0.87424716],
 [0.76910135, 0.86927688, 0.77382777, 0.80656279 ,0.74284263 ,0.79612134,
  0.75801865, 0.78682784],
 [0.84330446 ,0.79353021, 0.81601511, 0.77848262, 0.74776971, 0.76225854,
  0.80190411 ,0.74859737]]




    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.array(feasibility1meanfin).transpose(), cmap=cm.Greys,
                       linewidth=0, antialiased=False)
    ax.set_zlim(0.6, 1)
    ax.zaxis.set_major_locator(LinearLocator(9))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    correctnessmeanfin=[[0.96, 1. ,  1.,   1.  , 1. ,  1.  , 1. ,  0.96],
 [0.96, 1.,   0.96, 1.  , 0.92, 0.96, 1.  , 0.96],
 [1.  , 0.92, 0.92 ,0.92 ,1. ,  0.96 ,0.96, 0.96],
 [0.96, 0.92, 0.92, 0.92, 0.88, 1.  , 1.,   0.96],
 [0.92, 0.92, 0.88, 0.96, 0.96 ,0.92 ,0.92, 0.96],
 [0.92, 0.92, 0.88 ,0.92, 0.92 ,0.92 ,0.92, 0.84],
 [0.92, 0.92, 0.92, 0.88 ,0.84 ,0.88, 0.8  ,0.8 ],
 [0.92, 0.92, 0.88 ,0.84, 0.84 ,0.8 , 0.76, 0.76],
 [0.92 ,0.92 ,0.84 ,0.8 , 0.72, 0.72 ,0.68 ,0.68]]





    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.array(correctnessmeanfin).transpose(), cmap=cm.Greys,
                       linewidth=0, antialiased=False)

    ax.set_zlim(0.6, 1)
    ax.zaxis.set_major_locator(LinearLocator(9))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(os.path.join('plottypicality.png'))
    plt.show()


    catsparsmeanfin=[[0.52865079, 0.43701587, 0.34257143 ,0.3107619,  0.26609524 ,0.23247619,
  0.20647619, 0.19466667],
 [0.53871429, 0.414  ,    0.34095238, 0.322  ,    0.254   ,   0.222,
  0.21666667 ,0.17333333],
 [0.52257143 ,0.43380952 ,0.37666667, 0.29533333 ,0.21866667, 0.202,
  0.18466667, 0.14933333],
 [0.52390476, 0.36580952 ,0.334   ,   0.27809524, 0.23066667, 0.19533333,
  0.14933333, 0.17266667],
 [0.47952381 ,0.38666667, 0.30133333 ,0.20933333 ,0.20847619 ,0.16466667,
  0.12666667 ,0.13704762],
 [0.5   ,     0.43447619 ,0.31     ,  0.252   ,   0.23333333 ,0.19666667,
  0.12466667 ,0.08      ],
 [0.4487619 , 0.43047619, 0.31466667, 0.22466667 ,0.27333333, 0.13142857,
  0.12     ,  0.06      ],
 [0.51333333 ,0.41866667 ,0.34133333 ,0.23666667, 0.208    ,  0.14,
  0.13     ,  0.11333333],
 [0.487    ,  0.41333333, 0.28333333 ,0.21333333 ,0.14666667, 0.12,
  0.1      ,  0.08666667]]




    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.array(catsparsmeanfin).transpose(), cmap=cm.Greys,
                       linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(7))
    ax.set_zlim(0, 0.60)
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(os.path.join('plottypicality.png'))
    plt.show()