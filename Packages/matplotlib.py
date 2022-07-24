from typing import Literal
from PIL.Image import EXTENT
import matplotlib as mpl
from matplotlib import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.indexes import category
plt.style.use(['seaborn-notebook'])
#---------------------------서브플롯

#  -----------------------원하는 axes에 그래프 그리기
# fig, ax = plt.subplots(2,2)
# x= [1,2,3,4,5]
# y=[3,4,5,6,7]

# ax[0][0].plot(np.random.randn(100))   #인덱스지정해야함
# plt.plot(x)

# for i in range(1):
#     fig, ax = plt.subplots(2,2, figsize=(10,10))
#     for j in range(2):
#         ax[i][j].plot()



# fig = plt.figure()
# ax = plt.axes()
# plt.show()
# 직선그래프
# plt.plot([0,0.2,0.4,0.6,0.8,1]*5)
# plt.show()
# 
# 싸인, 코싸인 그리기
# x = np.arange(0,10, 0.01)
# plt.plot(x, np.sin(x))
# plt.plot(x, np.cos(x))
# plt.show()

# 누적함수
# plt.plot(np.arange(0,10,2))
# plt.show()

# #라인스타일
# plt.plot(np.random.randn(50).cumsum(),linestyle='-')
# plt.plot(np.random.randn(50).cumsum(),linestyle='--')
# plt.plot(np.random.randn(50).cumsum(),linestyle='-.')
# plt.plot(np.random.randn(50).cumsum(),linestyle=':')
# plt.show()

#__________________________________________________________
# plt.plot(np.random.randn(50))           
# plt.show()

# plt.plot(np.arange(0,50, 2))
# plt.show()

# plt.plot([0,10,20,30,40,50])
# plt.show()

# ___________________________________________________________

# Plot Axis
# plt.plot(np.random.randn(50))
# plt.axis([-1, 50, -5, 5])
#    # plt.xlim(-1 50), ylime(-5, 5) 랑 같음
# # plt.axis('tight')
# # plt.axis('equal')
# plt.show()

#플롯 레이블
# plt.plot(np.random.randn(50))
# plt.title('title')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# plt.plot(np.random.randn(50), label ='A')
# plt.plot(np.random.randn(50), label ='B')
# plt.plot(np.random.randn(50), label ='C')
# plt.title('TITLE')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.show()

#폰트 관리자
# x = set([f.name for f in mpl.font_manager.fontManager.ttflist])

# font1 = {'family': 'DejaVu Sans', 'size':24, 'color':'black'}
# font2 = {'family': 'Liberation Mono', 'size':18, 'weight': 'bold', 'color':'darkred'}
# font3 = {'family': 'Humor Sans', 'size':43, 'weight':'light','color':'blue'}

# plt.plot([1,2,3,4,5], [1,2,3,4,5])
# plt.title('title', fontdict=font1)
# plt.xlabel('xlabel', fontdict=font2)
# plt.ylabel('ylabel',fontdict=font3)
# plt.legend()
# plt.show()

#Plot legend(범례)
# fig, ax = plt.subplots()
# ax.plot(np.random.randn(10), '-r', label='A')
# ax.plot(np.random.randn(10), ':b', label='B')
# ax.plot(np.random.randn(10), '--g', label='C')
# # ax.axis('equal')
# ax.legend(framealpha=1, shadow=True, borderpad=1)
# plt.show()

# plt.figure(figsize=(8,4))
# x = np.linspace(0,10,1000)
# y = np.cos(x[:, np.newaxis]* np.arange(0,2, 0.2))
# lines = plt.plot(x, y)

# plt.legend(lines[:4], ['c1,', 'c2', 'c3'])
# plt.show()

# #컬러바
# x = np.linspace(0,20,100)
# I = np.cos(x) - np.cos(x[:, np.newaxis])
# plt.imshow(I, cmap='Purples')
# plt.colorbar()
# plt.show()

# 다중플롯
# ax1 = plt.axes()
# ax2 = plt.axes([0.6, 0.5, 0.2, 0.3]) #위치좌표
# plt.show()

# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.4) # 그래프간의 간격

# for i in range(1, 10):
#        plt.subplot(3,3,i)
#        plt.text(0.5, 0.5, str((3,3,i)), ha='center')
# plt.show()

# 한번에 만들기
# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
# fig, ax = plt.subplots(5, 5, x = 'col', y ='row')

# for i in range(5):
#        for j in range(5):
#               ax[i,j].text(0.5,0.5, str((i,j)), ha = 'center')
# plt.show()

# 그리드 합치기
# grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.4)
# plt.subplot(grid[0,0])
# plt.subplot(grid[0,1:])
# plt.subplot(grid[1,:1])
# plt.subplot(grid[1,1:])

# plt.show()

# plt.figure(figsize=(5, 6))

# x = range(1,21)
# columns = [np.random.randn(20) * i for i in range(1,7)]
# print(len(columns))
# i = 0

# for c in columns:
#        i+=1

#        plt.subplot(3,2,i)
#        plt.plot(x, c, marker='o', linewidth=1, label=c)
#        plt.xlim(-1, 21)
#        plt.ylim(c.min()-1, c.max()+1)
# plt.show()#

#텍스트와 주석
# fig, ax = plt.subplots()
# ax.axis([0,10,0,10])
# ax.text(3, 6, ". transData(3,6)", transform=ax.transData)
# ax.text(0.2,0.4, ". transAxes(0.2,0.4)", transform=ax.transAxes)
# ax.text(0.2, 0.2, ". transFigure(0.2,0.2)",transform=fig.transFigure)


# ax.set_xlim(-6,10) #transdata만 바뀜
# ax.set_ylim(-6,10)

# plt.show()

# 단순주석
# x = np.arange(1,40)
# y = x*1.1
# plt.scatter(x,y, marker='.')
# plt.axis('equal')
# plt.annotate('interseting point', xy=(4,5),xytext=(20,10), arrowprops=dict(shrink=0.05))
# plt.show()

#복잡주석
# x1= np.random.normal(30,3,100)
# x2= np.random.normal(20,3,100)
# x3= np.random.normal(10,3,100)

# plt.plot(x1, label='p1')
# plt.plot(x2, label='p2')
# plt.plot(x3, label='p3')

# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=3,
#        mode='extend', borderaxespad=0.)
# plt.annotate('important value', (50, 20), xytext=(5,40), arrowprops=dict(arrowstyle='->'))
# plt.annotate('incorrect value', (40, 30), xytext=(50,40), arrowprops=dict(arrowstyle='->'))
# plt.show()

#눈금맞춤(customizing ticks)

# plt.axes(xscale='log', yscale='log')

# ax = plt.axes()
# ax.plot(np.random.randn(100).cumsum())
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

# plt.show()

# fig,ax = plt.subplots(5,5,x=True,y=True)

# for axi in ax.flat:
#     axi.xaxis.set_major_locator(plt.MaxNLocator(4))
#     axi.yaxis.set_major_locator(plt.MaxNLocator(4))
# plt.show()

# x = np.linspace(-np.pi, np.pi, 1000, endpoint=True)
# y= np.sin(x)
# plt.plot(x,y)

# ax = plt.gca()
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['left'].set_position(('data', 0))
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# plt.show()

# x = np.linspace(1,10)
# y = [10** el for el in x]
# z = [2 *el for el in x]

# fig = plt.figure(figsize=(10,3))

# ax1 = fig.add_subplot(2,2,1)
# ax1.plot(x,y,'y')
# ax1.set_yscale('log')
# ax1.set_title('Logarithmic plot of $ {10}^{x} $')
# ax1.set_ylabel(r'${y}={10}^{x}$')
# plt.grid(b=True, which='both', axis='both')

# ax2 = fig.add_subplot(2,2,2)
# ax2.plot(x,y,'--r')
# ax2.set_yscale('Linear')
# ax2.set_title('Linear plot of $ {10}^{x} $')
# ax2.set_ylabel(r'${y}={10}^{x}$')
# plt.grid(b=True, which='both', axis='both')

# ax3 = fig.add_subplot(2,2,3)
# ax3.plot(x,y,'-.g')
# ax3.set_yscale('log')
# ax3.set_title('Logarithmic plot of $ {2}^{x} $')
# ax3.set_ylabel(r'${y}={2}^{x}$')
# plt.grid(b=True, which='both', axis='both')

# ax4 = fig.add_subplot(2,2,4)
# ax4.plot(x,y,':b')
# ax4.set_yscale('linear')
# ax4.set_title('Linear plot of $ {2}^{x} $')
# ax4.set_ylabel(r'${y}={2}^{x}$')
# plt.grid(b=True, which='both', axis='both')

# plt.show()

#스타일
# fig = plt.figure(figsize=(10,10))
# x = range(1, 11)
# columns = [np.random.randn(10)* i for i in range(1,26)]

# for n, v in enumerate(plt.style.available[1:]):
#     plt.style.use(v)
#     plt.subplot(5,5,n+1)
#     plt.title(v)

#     for c in columns:
#         plt.plot(x, c, marker='', color='royalblue', linewidth=1, alpha=0.4)
#         plt.subplots_adjust(hspace=0.5, wspace=0.4)

# plt.show()

plt.style.use(['seaborn-notebook']) #스타일 사용

#플롯

#bar plot
#y 기준으로
# height = [np.random.randn()* i for i in range(1,6)]
# names= ['A','B','C','D','C']
# y_pos = np.arange(len(names))
# plt.bar(y_pos, height)
# plt.xticks(y_pos, names, fontweight='bold')
# plt.xlabel('group')
# plt.show()

# x기준으로
# height = [np.random.randn()* i for i in range(1,6)]
# names= ['A','B','C','D','C']
# y_pos = np.arange(len(names))
# plt.barh(y_pos, height) # 수평표현
# plt.yticks(y_pos, names, fontweight='bold')
# plt.ylabel('group')
# plt.show()

#위아래스택형

# bar1 = [5, 8, 11, 14, 17]
# bar2 = [8, 11, 14, 17, 20]
# bar3 = [10, 13, 16, 19, 22]
# bars = np.add(bar1, bar2).tolist()
# print(bars)

# r = [0,1,2,3,4]
# names= ['A','B','C','D','C']
# plt.bar(r, bar1, color='royalblue', edgecolor='white')
# plt.bar(r, bar2, bottom=bar1, color='skyblue', edgecolor='white')
# plt.bar(r, bar3, bottom=bar2, color='lightblue', edgecolor='white')
# plt.xlabel('group', fontweight ='bold')
# plt.xticks(r, names, fontweight='bold')
# plt.show()

# 위드스택형

# bar_width = 0.25
# bar1 = [14, 17, 9, 8, 7]
# bar2 = [14, 7, 16, 4, 10]
# bar3 = [25, 4 ,23 ,14, 17]

# r = [0,1,2,3,4]
# names= ['A','B','C','D','C']
# r1 = np.arange(len(bar1))
# r2 = [x + bar_width for x in r1]
# r3 = [x + bar_width for x in r2]
# plt.bar(r1, bar1, color='royalblue', width=bar_width, edgecolor='white' ,label='r1')
# plt.bar(r2, bar2, color='skyblue',width=bar_width, edgecolor='white',label='r2')
# plt.bar(r3, bar3, color='lightblue',width=bar_width, edgecolor='white',label='r3')
# plt.xlabel('group', fontweight ='bold')
# plt.xticks([r + bar_width for r in range(len(bar1))], ['A','B','C','D','C'])
# plt.legend()
# plt.show()

#barbs
# x = [0,5,10,15,30,40,50,60,100]
# v = [0,-5,-10,-15,-30,-40,-50,-60,-100]
# n = len(v)
# y = np.ones(n)
# u = np.zeros(n)

# plt.barbs(x,y,u,v,length=9)
# plt.xticks(x)
# plt.ylim(0.98, 1.05)
# plt.show()

#stem plot
# x = np.linspace(0.1, 2*np.pi, 41)
# y=  np.exp(np.sin(x))

# plt.stem(x,y, linefmt='gray',bottom=1, use_line_collection=True)
# plt.show()

#박스플롯
# r1 = np.random.normal(loc=0, scale=0.5, size=100)
# r2 = np.random.normal(loc=0.5, scale=1, size=100)
# r3 = np.random.normal(loc=1, scale=1.5, size=100)
# r4 = np.random.normal(loc=1.5, scale=2, size=100)
# r5 = np.random.normal(loc=2, scale=2.5, size=100)

# f , ax = plt.subplots(1,1)
# ax.boxplot((r1,r2,r3,r4,r5))
# ax.set_xticklabels(['r1','r2','r3','r4','r5'])
# plt.show()

#scatter plot
# plt.plot(np.random.randn(50), 'o')
# plt.show()

# #markers
# plt.figure(figsize=(0,4))
# markers=['.',',','o','v','^','<','>','1','2','3','4','s','p','*']
# for m in markers:
#     plt.plot(np.random.rand(5), np.random.rand(5),m, label="'{0}'".format(m))

# plt.legend(loc='center right', ncol=2)
# plt.xlim(0,1.5)
# plt.show()

# x = np.linspace(0,10,100)
# y = np.sin(x)
# plt.scatter(x,y,marker='^')
# plt.show()

# 예쁜 산점도
# for i in range(9):
#     x = np.random.randn(1000)
#     y = np.random.randn(1000).cumsum()
#     plt.scatter(x,y,alpha=0.2, cmap='viridis')

# plt.show()

#버블차트 with 컬러맵
# x = np.random.randn(100)
# y = np.random.randn(100)
# colors = np.random.randn(100)
# sizes = 1000* np.random.randn(100)

# plt.scatter(x,y,c=colors,s=sizes,alpha=0.3,cmap='inferno')
# plt.colorbar()
# plt.show()


#correlation
# x = np.random.randn(200)
# y1= np.random.randn(len(x))
# y2= 1.1 * np.exp(x)

# ax1= plt.plot()
# plt.scatter(x,y1,color='indigo',alpha=0.3,label='no corelation')
# plt.scatter(x,y2,color='blue',alpha=0.3,label='corelation')
# plt.grid(True)
# plt.legend()
# plt.show()

#coherence(어렵)
# dt = 0.01
# t = np.arange(0,30,dt)
# n1 = np.random.randn(len(t))
# n2 = np.random.randn(len(t))
# r = np.exp(-t/0.05)
# c1 = np.convolve(n1, r, mode='same')*dt
# c2= np.convolve(n2, r, mode='same')*dt
# s1 = 0.01 * np.sin(2 *np.pi*10+t) +c1
# s2 = 0.01 * np.sin(2 *np.pi*10+t) +c2

# plt.subplot(211)
# plt.plot(t, s1, t ,s2)
# plt.xlim(0,5)
# plt.xlabel('time')
# plt.ylabel('s1 & s2')
# plt.grid(True)
# plt.subplot(212)

# plt.cohere(s1, s2**2,256,1./dt)
# plt.ylabel('coherence')
# plt.show()

#오차막대

# x = np.linspace(0,20,40)
# dy = 1
# y = np.sin(x)+dy * np.random.randn(40)

# plt.errorbar(x,y,yerr=dy,fmt = 'o', color='darkblue', ecolor='gray', elinewidth=2)
# plt.show()

#2차원 유사플롯 (히트맵)
# plt.pcolor(np.random.rand(20,20),cmap='Reds')
# plt.show()

#히스토그램, 구간화, 밀도
# data = np.random.randn(10000)
# plt.hist(data,bins=50,alpha=0.5,histtype='stepfilled',color='steelblue',edgecolor='none')
# plt.show()

# x1 = np.random.normal(0,1,10000)
# x2 = np.random.normal(-5,3,10000)
# x3 = np.random.normal(5,2,10000)
# d = dict(histtype='stepfilled', alpha=0.3, bins=50)
# plt.hist(x1, **d)
# plt.hist(x2, **d)
# plt.hist(x3, **d)
# plt.show()

#2차원 히스토그램(desinty 표현)
# x= np.random.normal(size=50000)
# y = x - np.random.normal(size=50000)
# # plt.hist2d(x,y, bins=50, cmap='OrRd')
# plt.hexbin(x,y,gridsize=20, cmap='OrRd')
# plt.colorbar()
# plt.show()

#밀도와 등고선 플롯
# a = np.arange(-1,1,0.1)
# X,Y = np.meshgrid(a,a)
# Z = np.sin(X*Y)
# CS = plt.contour(X,Y,Z, level=a)
# plt.clabel(CS,inline=2)
# plt.colorbar()
# plt.show()

# def f(x,y):
#     return (1-(x**2+y**2)) * np.exp(-y**2/2)

# x = np.arange(-1.5, 1.5, 0.1)
# y = np.arange(-1.5, 1.5, 0.1)

# X,Y = np.meshgrid(x,y)
# Z = f(X,Y)
# N = np.arange(-1,2,0.2)
# CS = plt.contour(Z,N, linewidths=2, cmap='rainbow')
# plt.clabel(CS, inline=True, fmt='%1.1f')
# plt.colorbar(CS)
# plt.show()
# ----------------------------------------------
# l = np.linspace(-1.0, 1.0 ,1000)
# X,Y = np.meshgrid(l,l)
# Z = np.sqrt(X**2 + Y**2)
# lv = np.linspace(Z.reshape(-1,1).min(),Z.reshape(-1,1).max(), 40)
# plt.contour(X,Y,Z, levels=lv)
# plt.colorbar()
# plt.show()

# plt.contourf(X,Y,Z, levels=lv)
# plt.colorbar()
# plt.show()

# plt.imshow(Z, extent=[-1,1,-1,1], origin='lower',cmap='rainbow',alpha=0.4)
# plt.show()

#스트림 플롯
# Y,X = np.mgrid[0:5:100j, 0:5:100j]
# U = X
# V = np.sin(Y)
# plt.streamplot(X,Y,U,V)
# plt.show()

# Y,X = np.mgrid[-3:3:100j, -3:3:100j]
# U = -1 - X**2 + Y
# V = 1 + X - Y **2
# speed = np.sqrt(U**2 + V**2)
# plt.figure(figsize=(12,7))
# plt.streamplot(X,Y,U,V, density=1)
# plt.show()

#화살표 2차원 필드(quiver)
# import sympy

# x, y= sympy.symbols('x y')
# f = x**2 + y**2 + x*y - sympy.sin(x) * 4
# fdx = sympy.diff(f, x)
# fdy = sympy.diff(f, y)

# sample_size=100
# xs, ys = np.meshgrid(np.linspace(-10, 10, sample_size), np.linspace(-10,10,sample_size))

# zs = [float(f.subs(x, xv).subs(y, yv)) for xv, yv in zip(xs.ravel(), ys.ravel())]
# zs = np.array(zs).reshape(sample_size,sample_size)

# plt.contour(xs,ys,zs,40,levels=np.logspace(-0.5,2.0,40),cmaps='rainbow')

# xs_q, ys_q = np.meshgrid(np.linspace(-10,10,10),np.linspace(-10,10,10))

# xsg = [-float(fdx.subs(x,xv).subs(y,yv))for xv,yv in zip(xs_q.ravel(),ys_q.ravel())]
# ysg = [-float(fdx.subs(x,xv).subs(y,yv))for xv,yv in zip(xs_q.ravel(),ys_q.ravel())]

# plt.quiver(xs_q, ys_q, xsg, ysg, width=0.005,scale=500,color='black')
# plt.show()

#파이차트
# data = [10, 50, 30, 40, 60]
# categories = ['c1', 'c2', 'c3','c4','c5']
# explode =[0.1,0.1,0.1,0.1,0.1]
# plt.pie(data, explode= explode, labels = categories, autopct ='%0.1f%%')
# plt.legend(categories)
# plt.show()

#레이다 차트
# df = pd.DataFrame({
#      'group':['A','B','C','D'],
#       'var1':[38,1.5,30,4],
#       'var2':[29,10,9,34],
#       'var3':[8, 39, 23, 24],
#       'var4':[28,15,32,14]
# })

# categories = list(df)[1:]
# N = len(categories)

# angles = [n/ float(N)*2 * np.pi for n in range(N)]
# angles += angles[:1]

# ax = plt.subplot(111,polar=True)
# ax.set_theta_offset(np.pi / 2)
# ax.set_theta_direction(-1)
# ax.set_rlabel_position(0)
# plt.xticks(angles[:-1], categories)

# plt.yticks([10,20,30], ["10","20","30"], color='gray',size=7)
# plt.ylim(0,40)

# values = df.loc[0].drop('group').values.flatten().tolist()
# values += values[:1]
# ax.plot(angles, values, linewidth=1, linestyle='solid',label='A')
# ax.fill(angles, values, 'b', alpha=0.1)

# values = df.loc[1].drop('group').values.flatten().tolist()
# values += values[:1]
# ax.plot(angles, values, linewidth=1, linestyle='solid',label='B')
# ax.fill(angles, values, 'r', alpha=0.1)

# plt.legend(bbox_to_anchor=(0.1, 0.1))
# plt.show()

#생키 다이어그램
# from matplotlib.sankey import Sankey
# Sankey(flows=[0.20, 0.15, 0.25, -0.25,-0.25, -0.15, -0.60, -0.20],
#        labels=['Zero','One','Two', 'Three', 'Four','Five','six','Seven'],
#        orientations = [-1, 1, 0 ,1, 1,1,0,-1]).finish()
# plt.show()

#3차원 플로팅
#from mpl_toolkits import mplot3d

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# x = range(1,101)
# y = np.random.randn(100) * x
# z = np.random.randn(100) * x
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(x,y,z, c='green', s =60)
# plt.show()
# -------------------------------------------
# ax = plt.axes(projection='3d')
# zline = np.linspace(0,20,1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(xline,yline,zline,'gray')

# zdata= 20 * np.random.random(100)
# xdata= np.sin(zdata) + 0.2*np.random.randn(100)
# ydata= np.cos(zdata) + 0.2*np.random.randn(100)
# ax.scatter3D(xdata,ydata,zdata, c=zdata,cmap='Blues')
# plt.show()

# ----------------CONTOUR 3D
# def f(x,y):
#     return np.cos(np.sqrt(x**2 + y**2))

# l = np.linspace(-4,4,20)
# X,Y = np.meshgrid(l,l)
# Z = f(X,Y)

# fig = plt.figure()
# ax = plt.axes(projection= '3d')
# ax.contour3D(X,Y,Z,50,cmap='BuPu')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.view_init(60, 30)
# plt.show()

# 뼈대만
# fig =plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_wireframe(X,Y,Z, color='red')
# plt.show()

#표면만
# ax = plt.axes(projection='3d')
# ax.plot_surface(X,Y,Z,rstride=1,cstride=1, cmap='viridis',edgecolor='none')
# plt.show()

# 모양변형
# r = np.linspace(0,6,20)
# theta = np.linspace(-0.8 * np.pi, 0.8 * np.pi, 40)
# r, theta = np.meshgrid(r,theta)
# X = r * np.sin(theta)
# Y = r * np.cos(theta)
# Z = f(X,Y)

# ax = plt.axes(projection='3d')
# ax.plot_surface(X,Y,Z,rstride=1,cstride=1, cmap='viridis',edgecolor='none')
# plt.show()

#히스토그램+3D (2:55부터)


