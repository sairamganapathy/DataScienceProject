import matplotlib.pyplot as plt
import numpy as np

x = [1,3,5,7,9]
y = [2,4,6,8,10]
# plt.plot(x,y,marker = 'x', color = 'green', label = 'my visualization')#we can use marker as o
# plt.xlabel('x data')
# plt.ylabel('y data')
# plt.legend()
# plt.show()

#Scatter chart
# plt.scatter(x,y,marker = 'x', c = x, label = 'my visualization')#we can use marker as o
# plt.scatter(x,y,marker = 'x', c = [2,5,6,7,7], label = 'my visualization')#we can use marker as o
# plt.scatter(x,y,marker = 'x', c = x, cmap = 'rainbow', label = 'my visualization')#we can use marker as o
# plt.xlabel('x data')
# plt.ylabel('y data')
# plt.legend()
# plt.colorbar()
# plt.show()

#bar chart
# team = ['csk','mi','rcb', 'rr']
# trophies = [4,4,1, 2]
# plt.bar(team, trophies, label = 'my team trophies')
# plt.xlabel('team')
# plt.ylabel('trophies')
# plt.legend()
# plt.show()

##sine wave(mean data)(normally distributed data)
## x = np.linspace(0,10,100)
## y = np.sin(x)
# x = np.random.rand(100)
# y = 2*x + np.random.randn(100)
# ## plt.plot(x,y,label = 'sine wave', color = 'blue', linestyle = '-', linewidth = 10)
# plt.scatter(x,y, color = 'blue', marker = 'o', label = 'Random Data')
# plt.xlabel('x data')
# plt.ylabel('y data')
# plt.title('Random Data chart')
# plt.legend()
# plt.show()


#bar chart
# categories = ['a','b','c']
# values = [25,40,35]
# plt.bar(categories, values, color = 'red', label = 'Bar Chart', edgecolor = 'black')
# plt.xlabel('categories')
# plt.ylabel('values')
# plt.title('Bar Chart')
# plt.legend()
# plt.show()


##Box plot
# value = [10, 20, 30, 40, 50, 2000, -9000, 75]
# plt.boxplot(value)
# plt.show()

#using numpy higher and lower bound
value = [10, 20, 30, 40, 50, 2000, -900, 75]
q1 = np.percentile(value, 25)
q3 = np.percentile(value, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
higher_bound = q3 + 1.5 * iqr

outlier = []
for v in value:
    if v < lower_bound:
        outlier.append(v)
    elif v > higher_bound:
        outlier.append(v)
print(outlier)