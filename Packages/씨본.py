#Seaborn
#다중 플롯 그리드 구조, 선형회귀모델 자동추정 및 표시, 컬레 팔레트

from inspect import CO_GENERATOR
from re import split
from numpy.random.mtrand import rand
from pandas.core.indexes.period import period_range
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy import stats
from seaborn import palettes
sns.set(style='whitegrid')


# fig=plt.figure(figsize=(5,5))

# x=[1,2,3,4,5]
# y=[2,4,5,1,2]

# subplot=fig.add_subplot(1,1,1)
# subplot.plot(x, y)

# fig.suptitle("Figure with only one Subplot")
# plt.show()

#scatter plot
# penguins = sns.load_dataset('penguins')
# print(penguins)
# sns.relplot(x="bill_length_mm", y="bill_depth_mm", data=penguins)
# plt.show()
# sns.relplot(x="flipper_length_mm", y="body_mass_g", data=penguins)
# sns.relplot(x="bill_length_mm", y="bill_depth_mm",hue="bill_length_mm", data=penguins)
# plt.show()
# sns.relplot(x="flipper_length_mm", y="body_mass_g", hue='species', style='species', data=penguins)

# sns.relplot(x="flipper_length_mm", y="body_mass_g", hue='species', style='island', data=penguins)
# plt.show()
#col로 따로그림 보기 편하게
# sns.relplot(x="flipper_length_mm", y="body_mass_g", hue='species', col='island', data=penguins)
# plt.show()
# sns.relplot(x="bill_length_mm", y="bill_depth_mm",hue="species", col='island', data=penguins)


# sns.relplot(x="bill_length_mm", y="bill_depth_mm",hue="body_mass_g", col='island', data=penguins)

# sns.relplot(x="bill_length_mm", y="bill_depth_mm",hue="body_mass_g", size='body_mass_g', data=penguins)
# sns.relplot(x="bill_length_mm", y="bill_depth_mm",hue="body_mass_g", sizes=(10,300), size='body_mass_g', data=penguins)

# plt.show()

#라인플롯
# flights = sns.load_dataset('flights')
# print(flights)

# sns.relplot(x='year', y='passengers', kind='line', data=flights)
# plt.show()

#dots = sns.load_dataset('dots')
# print(dots)

# sns.relplot(x='time', y='firing_rate', kind='line', data=dots)

#신뢰구간 꺼져
# sns.relplot(x='time', y='firing_rate', kind='line', ci=None, data=dots)
# #모양바꿔 표준편차로
# sns.relplot(x='time', y='firing_rate', kind='line', ci='sd', data=dots)
# plt.show()
#집계표현 꺼져
# sns.relplot(x='time', y='firing_rate', kind='line', estimator=None, data=dots)

# sns.relplot(x='time', y='firing_rate', kind='line', hue='choice', data=dots)


# sns.relplot(x='time', y='firing_rate', kind='line', hue='align',style='choice', data=dots)


# sns.relplot(x='time', y='firing_rate', kind='line', hue='align',style='choice', dashes=False, markers=True, data=dots)
# sns.relplot(x='time', y='firing_rate', kind='line', hue='align',style='choice', dashes=True, markers=False, data=dots)



# sns.relplot(x='time', y='firing_rate', kind='line', hue='align',style='choice', col='choice', dashes=False, markers=True, data=dots)

# sns.relplot(x='time', y='firing_rate', kind='line', style='choice', data=dots.query("align == 'sacc'"))


# sns.relplot(x='time', y='firing_rate', kind='line', style='choice', hue='coherence', data=dots.query("align == 'sacc'"))

# sns.relplot(x='time', y='firing_rate', kind='line', col='choice', style='choice', hue='coherence', data=dots.query("align == 'sacc'"))
# plt.show()

#fmri = sns.load_dataset('fmri')
# print(fmri)

# sns.relplot(x='timepoint', y = 'signal',kind='line', data = fmri)


# sns.relplot(x='timepoint', y = 'signal', style='region', size='event', kind='line', data = fmri)

# sns.relplot(x='timepoint', y = 'signal', hue='subject', col='region', size='event', kind='line', data = fmri)

# palettes = sns.cubehelix_palette(n_colors=14, light=0.8)
# sns.relplot(x='timepoint', y = 'signal', hue='subject', style='event', col='region', size='event', kind='line', palette=palettes, data = fmri)
# plt.show()
# sns.relplot(x='timepoint', y = 'signal', hue='subject', col='region', row='event', size='event', kind='line',  data = fmri)

# sns.relplot(x='timepoint', y = 'signal', hue='event', col='subject', col_wrap=5, linewidth=3,  style='event',  size='event', kind='line',  data = fmri.query("region=='parietal'"))


# plt.show()

#tdf = pd.DataFrame(np.random.randn(40,4),
                     index=pd.date_range('2020-01-01', periods=40),
                        columns=['A','B','C','D'])
# print(tdf)
# sns.relplot(kind='line', data=tdf)

# g = sns.relplot(kind='line', data=tdf)
# g.fig.autofmt_xdate()
# plt.show()

# g = sns.relplot(kind='line', data=tdf['A'])
# g.fig.autofmt_xdate()

# plt.show()

#범주형 데이타
# print(penguins)
# sns.catplot(x='species', y='body_mass_g', 
#                 jitter=True,data=penguins)
# plt.show()
#swarm은 겹치지 않게 벌림
# sns.catplot(x='species', y='body_mass_g', 
#                  kind='swarm',data=penguins)
# plt.show()
# sns.catplot(x='species', y='body_mass_g', hue='sex',
#                  kind='swarm',data=penguins)

# sns.catplot(x='species', y='body_mass_g', hue='species',
#                  kind='swarm',data=penguins)

# sns.catplot(x='sex', y='body_mass_g', hue='species',kind='swarm',
#                 order=["Female", "Male"],data=penguins)
# plt.show()
# sns.catplot(x='body_mass_g', y='species', hue='island',kind='swarm',
#                 data=penguins)

# sns.catplot(x='species', y='body_mass_g', hue='sex',col='island', kind='swarm', aspect=0.7,
#                 data=penguins)


# plt.show()

#범주형 분포도(distribution plots)
# sns.catplot(x='species', y='body_mass_g',hue='sex', kind='box', dodge=False, data=penguins)

# sns.catplot(x='species', y='body_mass_g',hue='sex', kind='box', col='sex', data=penguins)

# sns.catplot(x='species', y='body_mass_g',hue='sex', kind='box', row='sex', data=penguins)

# sns.catplot(x='body_mass_g', y='species',hue='sex', kind='box', row='sex', height=2, aspect=4, data=penguins)

# plt.show()

#iris = sns.load_dataset('iris')
# print(iris)
# sns.catplot(kind='box', orient='h', data=iris)
# sns.catplot(x = 'species', y='sepal_length', kind='box',data=iris)
# sns.catplot(x = 'species', y='petal_length', kind='box',data=iris)
# plt.show()

#박슨 플롯 (분포모양 표시)
# diamonds = sns.load_dataset('diamonds')
# # print(diamonds)

# sns.catplot(x='cut', y='price',kind='boxen',data=diamonds)
# # sns.catplot(x='color', y='price',kind='boxen',data=diamonds)
# sns.catplot(x='color', y='price',kind='boxen',data=diamonds.sort_values('color')) #컬러명 sort함.
# sns.catplot(x='clarity', y='price',kind='boxen',data=diamonds)


# plt.show()

#바이올린 플롯 (커널밀도추정과 박스플롯의 결합)
# print(penguins)
# sns.catplot(x='species', y='body_mass_g', hue='sex', kind='violin', data=penguins)
# # sns.catplot(x='species', y='body_mass_g', hue='sex', kind='violin', bw=.15, data=penguins) #분포까지 나옴
# sns.catplot(x='species', y='body_mass_g', hue='sex', kind='violin', split=True, data=penguins)
# sns.catplot(x='species', y='body_mass_g', hue='sex', kind='violin', inner='stick', split=True, data=penguins)

# g = sns.catplot(x='species', y='body_mass_g', kind='violin', inner=None, data=penguins)  바이올린안에
# sns.swarmplot(x='species', y='body_mass_g',color='r', size=5,data=penguins, ax=g.ax)     스캐터찍음

# sns.catplot(kind='violin',orient='h', data=iris)


# plt.show()

#범주형 추정치 도표(estimate plot) 막대플롯/포인트플롯/카운트플롯

#바플롯
# mpg = sns.load_dataset('mpg')
# # print(mpg)
# sns.catplot(x='origin', y='mpg',hue='cylinders',kind='bar',data=mpg)

# sns.catplot(x='origin', y='horsepower',hue='cylinders',palette='ch:.20', kind='bar',data=mpg)
# sns.catplot(x='cylinders', y='horsepower',palette='ch:.20', kind='bar', edgecolor=".6", data=mpg)

# plt.show()

#포인트플롯 (축의높이를 사용하여 추정값을 인코딩하여 점추정값과 신뢰구간 표시)
#titanic= sns.load_dataset('titanic')
# print(titanic)
# sns.catplot(x='who', y='survived', hue='class', kind='point',data=titanic)

# # sns.catplot(x='class', y='survived', hue='who', kind='point',data=titanic)

# sns.catplot(x='class', y='survived', hue='who', palette={'man':'b', 'woman': 'r', 'child':'g'}, markers=["^", "o","."], linestyles=['-','--',':'],kind='point',data=titanic)

# sns.catplot(x='embark_town', y='survived', hue='who', kind='point', data=titanic)

# plt.show()

#카운트플롯
# sns.countplot(y='deck', data=titanic)
# sns.countplot(y='class', data=titanic)

# plt.show()

#분포시각화 

#일변량 분포
# x = np.random.randn(200)
# print(x)
# sns.distplot(x)
# plt.show()

# #히스토그램
# sns.displot(x, kde=False, rug=True)
# plt.show()
# sns.displot(x, kde=False, rug=True, bins=3) #막대갯수를 3개로

# plt.show()
#커널 밀도추정 (kde)
# sns.distplot(x, hist=False, rug=True)
# sns.kdeplot(x, shade=True)  #색 채워짐
# sns.kdeplot(x)
# sns.kdeplot(x, bw=.2, label='bw:0.2')
# sns.kdeplot(x, bw=1, label='bw:1')
# plt.legend()

# sns.kdeplot(x, shade=True, cut=True)  #꼬다리 깎임
# sns.rugplot(x)

# x = np.random.gamma(10, size=500)
# print(x)
# sns.distplot(x, kde=False, fit=stats.gamma)
# plt.show()

#이변량 분포

#산점도
# mean = [0,1]
# cov = [(1, .3), (.3,1)]
# data = np.random.multivariate_normal(mean, cov, 2000)
# # print(data)
# df = pd.DataFrame(data, columns=["x", "y"])
# print(df)
# sns.jointplot(x='x', y='y', data=df)
# # plt.show()

# #육각 빈 플롯(헥스빈 플롯)
# x, y = np.random.multivariate_normal(mean, cov, 2000).T
# with sns.axes_style('white'):
#    sns.jointplot(x=x, y=y, kind='hex')

# plt.show()

# #커널 밀도추정
# sns.jointplot(x='x', y='y', data=df, kind='kde')
# plt.show()

# sns.kdeplot(df.x, df.y)
# sns.rugplot(df.x, color='r')
# sns.rugplot(df.y, color='g', vertical=True)

# cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
# sns.kdeplot(df.x, df.y, cmap=cmap, n_levls=60, shade=True)

# g = sns.jointplot(x='x', y='y', data=df, kind='kde')
# g.plot_joint(plt.scatter, s=30, linewidth=1, marker='.')
# g.ax_joint.collections[0].set_alpha(0)
# plt.show()

#페어와이즈 관계 시각화
# sns.plot(penguins, hue='species')

# g = sns.Grid(penguins)
# g.map_diag(sns.kdeplot) #산점도 말고 kde플롯으로
# g.map_offdiag(sns.kdeplot, n_levels = 6)
# plt.show()

#히트맵 , 클러스터맵
#udata = np.random.randn(20, 30)

# sns.heatmap(udata)

# sns.heatmap(udata, vmin=0, vmax=1)
# ndata= np.random.randn(20, 30)
# sns.heatmap(ndata, center=0)
# plt.show()

#flights = flights.pivot('month', 'year','passengers')
# print(flights)
# sns.heatmap(flights, annot=True, fmt='d') #숫자표시함 히트맵에

# sns.heatmap(flights, cmap='BuPu', linewidths=.2)

# sns.heatmap(flights, cbar=False) #컬러바 없애기

#컬러바 x축으로

# grid_kws = {'height_ratios': (.9, 0.01), 'hspace': .5}
# f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
# ax = sns.heatmap(flights, ax=ax, cbar_ax=cbar_ax,cbar_kws={'orientation': 'horizontal'} )


# plt.show()

#클러스터맵 (1:32) #댄드로그램?

# brain_networks = sns.load_dataset('brain_networks', header=[0,1,2], index_col=0)

# networks = brain_networks.columns.get_level_values('network')
# used_networks = np.arange(1, 18)
# used_columns = (networks.astype(int).isin(used_networks))
# brain_networks = brain_networks.loc[:, used_columns]

# network_pal = sns.husl_palette(17, s=.5)
# network_lut = dict(zip(map(str, used_networks), network_pal))
# network_colors = pd.Series(networks, index= brain_networks.columns).map(network_lut)

# sns. clustermap(brain_networks.corr(), center=0, cmap='RdBu_r',
#              row_colors=network_colors, col_colors=network_colors,
#           linewidth= .5, figsize=(12,12))

# plt.show()

#선형관계 시각화(리니어 릴레이션쉽)

# sns.regplot(x='flipper_length_mm', y='body_mass_g', data=penguins)

# sns.lmplot(x='flipper_length_mm', y='body_mass_g', data=penguins)

# sns.lmplot(x='bill_length_mm', y='body_mass_g',hue='island', data=penguins)

# sns.lmplot(x='bill_length_mm', y='body_mass_g',hue='island', col='sex', data=penguins)

# sns.lmplot(x='bill_length_mm', y='flipper_length_mm',  data=penguins)

# sns.lmplot(x='bill_length_mm', y='flipper_length_mm',hue='species', x_estimator=np.mean, data=penguins)

# sns.lmplot(x='bill_length_mm', y='flipper_length_mm',hue='species', col='island', row='sex', data=penguins)

# plt.show()

#다른 종류의 모델
#anscombe = sns.load_dataset('anscombe')
# # print(anscombe.describe())
# sns.lmplot(x='x', y='y', data=anscombe.query("dataset== 'II'"),
#                  ci=None, scatter_kws={'s':80})

# #비선형 맞추기
# sns.lmplot(x='x', y='y', data=anscombe.query("dataset== 'II'"),
#                order=2,  ci=None, scatter_kws={'s':80})
# #선형 맞추기
# sns.lmplot(x='x', y='y', data=anscombe.query("dataset== 'III'"),
#                 robust=True, ci=None, scatter_kws={'s':80})
# plt.show()    

#이번엔 펭귄으로 
# penguins['long_bill'] = (penguins.bill_length_mm > penguins['bill_length_mm'].mean())
# sns.lmplot(x='body_mass_g', y = 'long_bill', y_jitter=.03, data=penguins)

# sns.lmplot(x='body_mass_g', y = 'long_bill', logistic=True, y_jitter=.03, data=penguins)

# sns.lmplot(x='bill_length_mm', y= 'flipper_length_mm', lowess=True, data=penguins)

# sns.residplot(x='x', y='y', data=anscombe.query("dataset =='I'"),
#                scatter_kws={'s':80})

# sns.residplot(x='x', y='y', data=anscombe.query("dataset =='II'"),
#                scatter_kws={'s':80})

# plt.show()

#다른 상황의 회귀
# sns.jointplot(x='body_mass_g', y='flipper_length_mm', kind='reg', data=penguins)
#각축의 분포와 히스토그램 각 축에 대한 각축

# sns.plot(penguins, x_vars=['bill_length_mm', "bill_depth_mm","flipper_length_mm"],
#            y_vars= ['body_mass_g'],
#            height=4, aspect=.8,
#            kind='reg')

# sns.plot(penguins, x_vars=['bill_length_mm', "bill_depth_mm","flipper_length_mm"],
#            y_vars= ['body_mass_g'],
#            height=4, aspect=.8, hue='species',
#            kind='reg')
      
# plt.show()

#구조화된 다중 플롯 그리드 (시본의 장점)

#FacetGrid

# sns.set(style='ticks')

# g = sns.FacetGrid(penguins, col= 'sex')
# g.map(plt.hist, "body_mass_g")

# g = sns.FacetGrid(penguins, col= 'species')
# g.map(plt.hist, "body_mass_g")

# g = sns.FacetGrid(penguins, col= 'species', hue='sex')
# g.map(plt.hist, "body_mass_g")

# g = sns.FacetGrid(penguins, col= 'species', hue='sex')
# g.map(plt.scatter, "bill_length_mm", "bill_depth_mm",alpha=.7)
# g.add_legend()

# g = sns.FacetGrid(penguins, col= 'species', hue='sex', margin_titles=True)
# g.map(sns.regplot, "bill_length_mm", "bill_depth_mm")
# g.add_legend()

# g = sns.FacetGrid(penguins, col= 'species', height=4, aspect=0.5, margin_titles=True)
# g.map(sns.barplot, "sex", "body_mass_g", order=['Female','Male'])

#plt.show()

#다른데이타
#tips = sns.load_dataset('tips')
# ordered_times = tips.time.value_counts().index
# g = sns.FacetGrid(tips, row='time', row_order=ordered_times, height=2, aspect=2,)

# g.map(sns.distplot, 'tip', hist=False, rug=True)

# plt.show()

# g = sns.FacetGrid(tips, hue= 'day', height=5)
# g.map(plt.scatter, 'total_bill', 'tip', s=30, alpha=.7, linewidth=.5)
# g.add_legend()
# plt.show()

# g = sns.FacetGrid(tips, hue= 'sex', palette='BuPu',
#             hue_kws=({'marker':['^', 'v']}), height=5)
# g.map(plt.scatter, 'total_bill', 'tip', s=30, alpha=.7, linewidth=5)
# g.add_legend()
# plt.show()

# g = sns.FacetGrid(tips, col='day', col_wrap=2, height=4)
# g.map(sns.pointplot, 'sex','tip', order=['Female', 'Male'])
# plt.show()

# with sns.axes_style('darkgrid'):
#    g = sns.FacetGrid(tips, row='sex', col='day', margin_titles=True, height=2.5)
# g.map(sns.scatterplot, 'total_bill', 'tip')


# g = sns.FacetGrid(tips, col='time', margin_titles=True, height=4)
# g.map(plt.scatter, 'total_bill', 'tip')
# for ax in g.axes.flat:
#         ax.plot((0, 50), (0, .2*50), c='.2', ls=':')
# plt.show()

#존멋퍼싯
# r = np.linspace(0,10 ,num=100)
# df = pd.DataFrame({"r": r, "slow": r, "medium": 2*r, "fast": 4*r})
# print(df)
# df = pd.melt(df, id_vars=['r'], var_name='speed', value_name= 'theta')

# g = sns.FacetGrid(df, col='speed', hue='speed',
#                 subplot_kws=dict(projection='polar'), height=5,
#                 sharex=False, sharey=False, despine=False)
# g.map(sns.scatterplot, 'theta', 'r')


# plt.show()

#커스텀 함수
# def quantile_plot(x,  **kwargs):
#    qntls, xr = stats.probplot(x, fit=False)
#    plt.scatter(xr, qntls, **kwargs)

# g = sns.FacetGrid(tips, col='time', height=4)
# g.map(quantile_plot, 'total_bill')

# plt.show()

#페어와이즈 데이터 관계(2:15)
# g = sns.Grid(tips)
# g.map(plt.scatter)

# g = sns.Grid(tips)
# g.map_diag(plt.hist)  #대각 hist
# g.map_offdiag(plt.scatter) #대각 아닌 부분 scatter로

# g = sns.Grid(tips, hue='day')
# g.map_diag(plt.hist)  #대각 hist
# g.map_offdiag(plt.scatter) #대각 아닌 부분 scatter로
# g.add_legend()


#원하는 부분만
# g = sns.Grid(tips, vars=['total_bill', 'tip'], hue='day')
# g.map(plt.scatter)

# g = sns.Grid(tips)
# g.map_upper(plt.scatter)  
# g.map_lower(sns.kdeplot) 
# g.map_diag(sns.kdeplot, lw=3 ,legend=False)

# g = sns.Grid(tips, y_vars=['tip'],
#              x_vars=['size','total_bill'], height=4)
# g.map(sns.regplot)
# plt.show()

# g = sns.Grid(tips, hue='day', palette='Set1')
# g.map(plt.scatter, s=30, edgecolor='white')
# g.add_legend()

# sns.plot(tips, hue='day', palette='Set1', diag_kind='kde' ,height=3)

# plt.show()

#그림 미학 제어

# def randplot(flip=1):
#        for i in range(1,7):
#               plt.plot(np.random.randn(50).cumsum())
              

# sns.set()  #디폴트임
# sns.set_style('dark')
#sns.set_style('white')
# sns.set_style('whitegrid')
# # sns.set_style('ticks')

# randplot()
# plt.show()

#축 스핀제거

# d = np.random.randn(50).reshape(10,5)
# # sns.distplot(d)
# # sns.despine()

# # sns.violinplot(data=d)
# # sns.despine(offset=10, trim=True)

# # sns.boxplot(data=d, palette='deep')
# # sns.despine(left=True)

# plt.show()

#스타일 임시 설정
# f = plt.figure(figsize=(6,6))
# gs = f.add_gridspec(2,2)

# with sns.axes_style('darkgrid'):
#    ax = f.add_subplot(gs[0,0])
#    randplot()

# with sns.axes_style('white'):
#    ax = f.add_subplot(gs[0,1])
#    randplot()

# with sns.axes_style('ticks'):
#    ax = f.add_subplot(gs[1,0])
#    randplot()

# with sns.axes_style('whitegrid'):
#    ax = f.add_subplot(gs[1,1])
#    randplot()

# f.tight_layout()

# plt.show()

#스타일 요소 재정의
# a = sns.axes_style()
# # print(a)

# sns.set_style('darkgrid',{'axes.facecolor':'.5', 'gird.linestyle': ':'})
# randplot()

# plt.show()

#스케일링 플롯
# sns.set()
# # sns.set_context('paper')
# # randplot()


# sns.set_context('talk')
# randplot()

# sns.set_context('poster')
# randplot()

# sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth':2.5})

# randplot()
# plt.show()

#팔레트

#질적 색상 팔레트
# current_palette = sns.color_palette()
# sns.palplot(current_palette)

# plt.show()

#원형 컬러 시스템사용
# sns.palplot(sns.color_palette('hls', 8))
# sns.palplot(sns.hls_palette(8, l=.3, s=.8))
# sns.palplot(sns.color_palette('husl', 8))

# plt.show()

#범주형 컬러 브루어 팔레트
# sns.palplot(sns.color_palette('ed'))
# sns.palplot(sns.color_palette('Set1'))
# sns.palplot(sns.color_palette('Set2'))
# sns.palplot(sns.color_palette('Set3'))

# flatui = ["#99FF00", "#99CC00", "#999900","#996600", "#993300", "#990000"]  #코드표 참조
# sns.palplot(sns.color_palette(flatui))

# plt.show()

# xkcd 색상
# plt.plot(np.random.randn(50).cumsum(), sns.xkcd_rgb['pale red'],lw=3)
# plt.plot(np.random.randn(50).cumsum(), sns.xkcd_rgb['medium green'],lw=3)
# plt.plot(np.random.randn(50).cumsum(), sns.xkcd_rgb['denim blue'],lw=3)

#내가 색 설정
# colors = ["windows blue", 'amber', 'greyish', 'faded green', 'dusty purple']

# sns.palplot(sns.xkcd_palette(colors))
# plt.show()

#순차 색상 팔레트
# sns.palplot(sns.color_palette('Blues'))
# sns.palplot(sns.color_palette('BuGn_r'))
# sns.palplot(sns.color_palette('GnBu_r'))

# plt.show()

#순차적 입방체 팔레트
# sns.palplot(sns.color_palette('cubehelix', 8))
# sns.palplot(sns.cubehelix_palette(8))
# sns.palplot(sns.cubehelix_palette(8, start=.5, rot=-.75))
# sns.palplot(sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=0.95, reverse=True))

# plt.show()


# class Hello:
#     def __enter__(self):
#         # 사용할 자원을 가져오거나 만든다(핸들러 등)
#         print('enter...')
#         return self # 반환값이 있어야 VARIABLE를 블록내에서 사용할 수 있다
        
#     def sayHello(self, name):
#         # 자원을 사용한다. ex) 인사한다
#         print('hello ' + name)

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # 마지막 처리를 한다(자원반납 등)
#         print('exit...')

# with Hello() as h:
#     h.sayHello('obama')
#     h.sayHello('trump')