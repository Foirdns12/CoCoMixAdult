import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import os
import matplotlib.ticker as tick

if __name__ == "__main__":

    x1 =[0.5 ,0.75 ,1 ,1.25 ,1.5 ,1.75 ,2 ,2.25]
    x2 = [0, 1, 2, 2.5, 3, 3.5, 4, 5]
    X, Y = np.meshgrid(x1, x2)
    print(X)
    print(Y)

    sparsitymeanfin=np.array([[5.08, 6.24, 6.28, 6.76, 6.64, 6.4,  6.44, 6.32],
 [4.32, 5.44, 6.4,  5.84, 6.,   5.48, 6.04, 5.8 ],
 [3.92 ,5.04, 5.08, 5.56 ,5.44, 5.48, 5.48, 5.88],
 [3.8 , 4.64, 4.72, 5.04 ,5.08, 5.24, 5.44, 5.36],
 [3.52 ,4.32, 5.04 ,4.6,  4.44, 5.2,  5.16,4.92],
 [3.08, 3.92, 4.32, 4.24, 4.12 ,4.76 ,4.84 ,5.12],
 [2.88, 3.92, 4.44, 4.2,  4.08, 4.28, 4.36, 4.36],
 [2.44, 3.68, 4.04, 3.68, 4.16, 3.88, 3.92, 4.28]])
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


    closenessmeanfin=[[24.79300104, 21.70288031, 21.97294771, 25.11961516, 22.91971759, 20.54739834,
  22.55092772, 21.18069369],
 [20.73662539, 18.17870652, 20.13565072, 18.47778243, 19.29028885, 17.83008627,
  18.89546209, 18.62074052],
 [18.6892262,  16.60677446 ,14.96410818, 17.09196289 ,15.56648141 ,15.42320398,
  15.10801719, 16.47130944],
 [17.09439411, 14.38805233, 14.19633561, 14.5365986 , 14.25445763, 15.66935596,
  14.64905019, 14.31364938],
 [15.6217884,  12.93294892, 14.79078459 ,11.64483516, 12.09078832, 14.48461636,
  13.29509828, 11.81805141],
 [13.97645575, 12.76996735 ,11.60934317, 10.59920857, 10.87846217, 11.32585067,
  13.70213751, 13.04528512],
 [11.43004439, 11.050066,   11.31794188, 11.1280156 ,  9.46497751,  9.11890081,
   9.00079798 ,10.95951992],
 [ 9.21815398, 10.10879989 ,10.56447808,  8.76203276, 10.86366618 , 8.57389715,
  10.2191229,   9.47282256]]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.array(closenessmeanfin).transpose(), cmap=cm.Greys,
                           linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(9))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.0f'))
    ax.set_zlim(8, 24)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(os.path.join(PATH, 'plotcloseness.png'))
    plt.show()


typicalitymeanfin=[[1.70360179e-17, 4.05271138e-16, 1.43371106e-15, 1.06250954e-15,
  7.71025091e-16, 1.80767267e-15, 2.03956896e-15, 3.33845221e-15],
 [9.86622981e-18, 3.94202224e-16, 1.03005898e-15, 9.44983458e-16,
  1.76112210e-15, 2.28605940e-15, 3.06560639e-15, 2.63412483e-15],
 [5.33728159e-17, 2.29678559e-16, 7.71280253e-16, 1.11664242e-15,
  1.50233095e-15, 1.54030241e-15, 2.72274179e-15, 2.22435273e-15],
 [4.04401308e-17, 2.91090116e-16, 1.01647979e-15, 1.76602074e-15,
  1.14907568e-15, 1.24539122e-15, 1.62873680e-15, 2.95051731e-15],
 [1.57599198e-17, 3.89927095e-16, 2.38484487e-16, 1.18832018e-15,
  1.14875831e-15, 1.58706573e-15, 1.39796167e-15, 1.93464175e-15],
 [1.29977787e-17, 1.87421697e-16, 7.56410365e-16, 6.57780415e-16,
  1.25969619e-15, 7.92429260e-16, 2.13284422e-15, 2.05739765e-15],
 [7.87495104e-17, 4.89832269e-16, 3.17829517e-16, 7.38731405e-16,
  9.37916866e-16, 1.62772658e-15, 1.21852511e-15, 1.81171369e-15],
 [4.13816491e-17, 1.86471596e-16, 2.32535917e-16, 9.44763792e-16,
  8.39882514e-16, 9.09705005e-16, 9.84398594e-16, 1.79780148e-15]]

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



feasibility2meanfin=[[0.01530181, 0.23571009, 0.3541424 , 0.32102792 ,0.29721092 ,0.41131624,
  0.54379359, 0.48562011],
 [0.06326146 ,0.3118469 , 0.4285558,  0.47987029, 0.46823502, 0.64329919,
  0.57185965 ,0.64648247],
 [0.10442883, 0.32652486, 0.49093227, 0.45443291, 0.62772061, 0.66337763,
  0.64712896, 0.73252111],
 [0.07376791, 0.32004578, 0.47609128 ,0.54869916, 0.5168469,  0.62972805,
  0.55706587, 0.70916765],
 [0.19332785, 0.3709123,  0.47157971 ,0.57130311, 0.57080791 ,0.54302919,
  0.74843401, 0.75593891],
 [0.14336385, 0.35577091, 0.52325049, 0.49574423, 0.66355329, 0.58996312,
  0.63027495, 0.66307422],
 [0.06941532, 0.33633824, 0.52548284, 0.58679285, 0.63698778, 0.7087259,
  0.74703579, 0.68339061],
 [0.06149472, 0.42540247, 0.52612226, 0.68003955, 0.69034023, 0.63588282,
  0.71599856, 0.71071519]]


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(X, Y, np.array(feasibility2meanfin).transpose(), cmap=cm.Greys,
                       linewidth=0, antialiased=False)
ax.set_zlim(0, 0.8)
ax.zaxis.set_major_locator(LinearLocator(9))
ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


feasibility1meanfin= [[0.97760487, 0.8503512,  0.81174657, 0.83050266, 0.81522477, 0.73662532,
  0.75261642, 0.70719447],
 [0.97106625, 0.85087998 ,0.79032714,0.68834942, 0.68863451, 0.67774631,
  0.59149767, 0.69745802],
 [0.93717086, 0.8878133 , 0.71526196, 0.81393604, 0.67687108, 0.64066644,
  0.69537372, 0.63478852],
 [0.94735342 ,0.76066958, 0.76382663, 0.69821699, 0.68634155, 0.6797265,
  0.67392052, 0.6179834 ],
 [0.85436065, 0.80586504, 0.79631728, 0.61066298 ,0.66354903, 0.70137812,
  0.57001069, 0.48447869],
 [0.8578529,  0.7290903 , 0.74747736, 0.73647577, 0.61226683, 0.62756562,
  0.641922,   0.72339905],
 [0.84489888, 0.77321331 ,0.70286257, 0.6674981,  0.56354141, 0.57230941,
  0.50367635, 0.50586519],
 [0.80207153, 0.73795024, 0.75124566, 0.54501297 ,0.53298247, 0.56108608,
  0.5705848,  0.56825584]]


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, np.array(feasibility1meanfin).transpose(), cmap=cm.Greys,
                       linewidth=0, antialiased=False)
ax.set_zlim(0.44, 1)
ax.zaxis.set_major_locator(LinearLocator(9))
ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

correctnessmeanfin=[[1. ,  0.96, 0.92, 0.96 ,0.96, 0.88, 0.96, 0.92],
 [1.  , 0.96 ,1.  , 0.92 ,0.92 ,0.92, 0.92 ,0.92],
 [0.96 ,0.92, 0.88,0.96 ,0.88 ,0.88 ,0.8,  0.84],
 [0.96, 0.88, 0.92 ,0.88, 0.84, 0.84 ,0.84, 0.8 ],
 [0.96, 0.88, 0.88, 0.76 ,0.76, 0.84, 0.8,  0.72],
 [0.92, 0.84, 0.84 ,0.76, 0.8,  0.72, 0.84, 0.8 ],
 [0.88, 0.84, 0.84, 0.76 ,0.76 ,0.64, 0.64, 0.68],
 [0.84, 0.76, 0.76, 0.64, 0.8,  0.64, 0.68, 0.64]]



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


catsparsmeanfin=[[0.3332381,  0.309,      0.29477778, 0.34838095, 0.32011111, 0.30815873,
  0.32001587, 0.28806349],
 [0.288,      0.32238095, 0.2905873,  0.28033333, 0.31933333, 0.31806349,
  0.29068254, 0.27520635],
 [0.324,      0.27433333, 0.24990476, 0.26819048, 0.25890476, 0.24815873,
  0.24273016, 0.24919048],
 [0.30266667, 0.24952381, 0.23942857, 0.24004762, 0.22514286, 0.27173016,
  0.23961905, 0.24890476],
 [0.276 ,     0.22  ,     0.24261905, 0.21009524 ,0.23066667, 0.25185714,
  0.22580952, 0.202     ],
 [0.29066667, 0.26809524, 0.21466667, 0.2107619,  0.20133333, 0.17314286,
  0.2317619,  0.22033333],
 [0.28133333, 0.20666667, 0.19419048, 0.216   ,   0.19966667 ,0.1947619,
  0.17038095, 0.20609524],
 [0.27   ,    0.22533333 ,0.21466667, 0.18133333, 0.194 ,     0.20342857,
  0.20066667, 0.21666667]]



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, np.array(catsparsmeanfin).transpose(), cmap=cm.Greys,
                       linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(9))
ax.set_zlim(0.15, 0.35)
ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.savefig(os.path.join('plottypicality.png'))
plt.show()