import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin, pi

X  = np.linspace(0,10,500)
Y1 = 10*np.exp(-(X-8)**2)
Y2 = np.cos(3*pi*X)*np.exp(-X/2)

plt.figure("Tracé de courbe avec matplotlib.pyplot", figsize = (15,5))
# ajustement des espaces :
plt.subplots_adjust(left=0.05, right=0.95, wspace=0.25)   

plt.subplot(1,2,1)  # 1 ligne, 2 subplots par ligne : subplot n° 1
deltaX = abs(X.max()-X.min())
plt.xlim(X.min()-0.01*deltaX, X.max()+0.01*deltaX)
ymax = max(abs(Y2.min()), abs(Y2.max()))
plt.ylim(-1.1*ymax, 1.1*ymax)
plt.ylabel("Axe Y1", color="g")
p1 = plt.plot(X, Y2, "g-", linewidth=2, label=r"$\cos(2\pi x)e^{-x/2}$")
plt.grid()

# activation du deuxième système d'axes (X,Y2) :
plt.twinx()
plt.title("SubPlot 1 : deux tracés dans les systèmes d'axes (X,Y1) et (X,Y2)")
deltaY1 = abs(Y1.max()-Y1.min())
plt.ylim(Y1.min()-0.05*deltaY1, Y1.max()+0.05*deltaY1)
p2 = plt.plot(X,Y1,"b-",linewidth=2,label="$10\,e^{-(x - 8)^2}$")
plt.xlabel("x (abcisse commune)")
plt.ylabel("Axe Y2", color="b")


# une seule  boite de légende pour les deux courbes
p12 = p1+p2
labels = [L.get_label() for L in p12]
plt.legend(p12, labels, loc="upper center", fontsize=20)

plt.subplot(1,2,2)  # 1 ligne, 2 subplots par ligne : subplot n° 2
plt.title("SubPlot 2 : courbe paramétrée")
A = np.linspace(0, pi, 300)
X = (np.sin(5*A)-3*np.cos(A))*np.cos(A)
Y = (np.sin(5*A)-3*np.cos(A))*np.sin(A)
plt.plot( X, Y, "m-", linewidth=4);
plt.plot(-X, Y, "m-", linewidth=4, label="Papillon")
plt.axis("equal");
plt.legend()
plt.grid();

plt.savefig("traceCourbes.png")
plt.show()
