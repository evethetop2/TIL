import numpy as np
import pandas as pd
from pandas.core.indexes.multi import MultiIndex
from pandas.tseries import frequencies

#series 객체
# s = pd.Series([0, 0.25, 0.5, 0.75, 1.0])
# print(s)

# print(s.values) #밸류만
# print(s.index) #인덱스만

# print(s[1])

# s = pd.Series([0, 0.25, 0.5, 0.75, 1.0], index=['a','b','c','d','e'])
# print(s)
# print(s[['c','d','e']])
# print('b' in s)

# print(s.unique())
# print(s.value_counts()) # 카운트가능
# print(s.isin([0.25, 0.75])) #불리언으로 나옴

# pop_tuples = {'서울특별시': 10000000, '인천광역시':152152,
#      '대구광역시': 215215, '대전광역시': 151512, '광주광역시': 15125}

# population= pd.Series(pop_tuples)
# print(population)
# print(population['서울특별시':'대구광역시'])

#DataFrame 객체
# a = pd.DataFrame([{'A':2, 'B':4, 'D':3}, {'A':4, 'B':5,'C':7}])
# print(a)

# b = pd.DataFrame(np.random.rand(5,5),
#                   columns=['A','B','C','D','E']
#                   ,index=[1,2,3,4,5])
# print(b)

# male_tuple ={'서울특별시': 8525125, '인천광역시':152152,
# '대구광역시': 215215, '대전광역시': 151512, '광주광역시': 15125}
# male = pd.Series(male_tuple)
# # print(male)c

# female_tuple ={'서울특별시': 125125, '인천광역시':452152,
# '대구광역시': 515215, '대전광역시': 121512, '광주광역시': 5125}
# female=pd.Series(female_tuple)
# print(female)

# korea_df = pd.DataFrame({'인구수':population, '남자인구수':male_tuple, '여자인구수':female}) #인덱스 모두 동일
# print(korea_df)

# print(korea_df.index) #행
# print(korea_df.columns) #열

# print(korea_df['여자인구수'])

#index 객체
# idx = pd.Index([2,4,6,8,10])
# print(idx)

# print(idx[0:4:2])
# print(idx[-1::])
# print(idx[::2])

# print(idx)
# print(idx.size)
# print(idx.shape)
# print(idx.ndim)
# print(idx.dtype)

#인덱스연산

# idx1 = pd.Index([1,2,4,6,8])
# idx2 = pd.Index([2,4,5,6,7])
# print(idx1.append(idx)) # 합집합(중복허용)
# print(idx1.difference(idx2)) # 차집합
# print(idx1-idx2) #각 리스트들끼리 뺄셈, 차집합이랑 다름
# print(idx.intersection(idx2)) #교집합
# print(idx1 & idx2) # 교집합
# print(idx1.union(idx2)) #합집합(중복허용x)
# print(idx1 | idx2 )
# print(idx1.delete(0))
# print(idx1.drop(1))
# print(idx1 ^ idx2) #여집합(공통파트 뺴고 나머지)

#인덱싱
# s = pd.Series([0, 0.25, 0.5, 0.75, 1.0] 
#             ,index=['a','b','c','d','e'])
# print(s)
# print(s.keys())
# print(list(s.items()))

# s['f'] = 1.25 #추가
# print(s)

#슬라이싱할 때 주의!
# print(s['a':'d'])
# print(s[0:4])  #넘버링일때만 0~3임

# print(s[(s > 0.4) & (s < 0.8)])

#series 인덱싱
# s = pd.Series(['a','b','c','d','e'],
#              index=[1,3,5,7,9])
# print(s)
# print(s.iloc[0]) #정수인덱스로 접근
# print(s.iloc[2:4])

# a = s.reindex(range(10), method ='bfill') #빈곳을 채워줌
# print(a)

#DataFrame 인덱싱

# a = korea_df['남자인구수']
# print(a)
# # print(korea_df.남자인구수)

# korea_df['남녀비율'] = (korea_df['남자인구수']*100 / korea_df['여자인구수'])
# print(korea_df) # 칼럼추가


# print(korea_df.values)
# print(korea_df.T)
# print(korea_df.values[1]) #행으로 가져옴
# print(korea_df['인구수'])
# print(korea_df.loc[:'인천광역시',:'남자인구수'])
# print(korea_df.loc[korea_df.여자인구수 > 10000000])


#다중 인덱싱(3~4차원 처리가능)
# print(korea_df)
# idx_tuples = [('서울특별시',2010),('서울특별시',2020),
# ('부산광역시',2010),('부산광역시',2020),
# ('인천광역시',2010),('인천광역시',2020),
# ('대구광역시',2010),('대구광역시',2020),
# ('대전광역시',2010),('대전광역시',2020),
# ('광주광역시',2010),('광주광역시',2020)]

# pop_tuples =[1023145,6436342,
#              12535235,1241115,
#             512412,5215153,
#             4214214,21421412,
#             2142153,4363462,
#             4214125,3223625]
# population = pd.Series(pop_tuples, index=idx_tuples)
# # print(population)

# midx =pd.MultiIndex.from_tuples(idx_tuples) #위에거보다 쉬움
# # print(midx)

# population=population.reindex(midx) #중복된거 빠짐
# # print(population)

# # print(population['대전광역시',:]) #대전만

# # ------------여기부터 mdf------------------

# korea_mdf = population.unstack() #df으로 변환 #reindex 해야댐
# # print(korea_mdf)
# # print(korea_mdf.stack()) #다시 다중인덱싱구조로 변환

# male_tuple = [5111256,135315,
#               141513,1432523,
#               153252,3451353,
#               634634,4535345,
#               5423145,364336,
#               331563,4617351]
# korea_mdf = pd.DataFrame({'총인구수':population,
#                     '남자인구수': male_tuple})

# female_tuple = [213216,12415,
#               141513,1432523,
#               153252,3451353,
#               634634,4535345,
#               5423145,364336,
#               331563,4617351]

# korea_mdf = pd.DataFrame({'총인구수':population,
#                     '남자인구수': male_tuple,
#                     '여자인구수': female_tuple})


# ratio = korea_mdf['남자인구수']*100 / korea_mdf['여자인구수']
# korea_mdf = pd.DataFrame({'총인구수':population,
#                     '남자인구수': male_tuple,
#                     '여자인구수': female_tuple,
#                     '남녀비율': ratio})
# # print(korea_mdf)



# #다중인덱스 생성
# df = pd.DataFrame(np.random.rand(6,3),
#              index=[['a','a','b','b','c','c'], [1,2,1,2,1,2]],
#              columns=['c1','c2','c3'])
# # print(df)
# # a = pd.MultiIndex.from_arrays([['a','a','b','b','c','c'], [1,2,1,2,1,2]])
# # print(a)

# # a = pd.MultiIndex.from_product([['a','b','c'],[1,2]]) #곱
# # print(a)

# # a = population.index.names=['지역','인구수'] #안됌
# # print(a)

# idx = pd.MultiIndex.from_product([['a','b','c'],[1,2]],
#             names=['name1', 'name2'],)
# cols = pd.MultiIndex.from_product([['c1','c2','c3'],[1,2]],
#                       names=['col_name1', 'col_name2'])
# data = np.round(np.random.randn(6,6), 2)
# mdf = pd.DataFrame(data, index=idx, columns=cols)
# # print(mdf)

# #멀티인덱스 (슬라이싱/인덱싱)
# # a = mdf['c2', 1]
# # print(a)

# # b = mdf.iloc[:2, :3]
# # print(b)
# # c = mdf.loc[:,('c2',1)] #내가아는 리스트랑 달라
# # print(c)

# # idx_slice = pd.IndexSlice
# # a = mdf.loc[idx_slice[:,2], idx_slice[:,2]]
# # print(a)

# #다중인덱싱 재정렬 (1:09)
# # print(idx)
# # print(korea_df)
# # print(korea_mdf)
# # print(korea_df['서울특별시':'인천광역시'])
# # print(korea_mdf.unstack(level=0))
# # print(korea_mdf.reset_index(level=0))

# # idx_flat= korea_mdf.reset_index(level=(0,1))
# # print(idx_flat)

# #데이터 연산
# # s = pd.Series(np.random.randint(0,10 ,5))
# # df = pd.DataFrame(np.random.randint(0,10, (3,3)),
# #                        columns=['A','B','C'],
# #                        index=['박','형','규'])

# # print(np.exp(s))
# # print(np.cos(df*np.pi/4))

# # s1 = pd.Series([1,3,5,7,9], index=[0,1,2,3,4])
# # s2 = pd.Series([2,4,6,8,10], index=[1,2,3,4,5])
# # print(s1+s2) #인덱스끼리만 계산함, 매칭안되면 nan으로 채워짐

# # a = s1.add(s2, fill_value=0)
# # print(a)

# #데이타프레임 연산
# # df1 = pd.DataFrame(np.random.randint(0,20,(3,3)),
# #                      columns=list('ACD'))

# # df2 = pd.DataFrame(np.random.randint(0,20,(5,5)),
# #                      columns=list('BAECD'))

# # print(df1+df2) #nan 다량발생

# # fvalue=df1.stack().mean()
# # a = df1.add(df2, fill_value= fvalue)
# # print(fvalue)
# # print(a)

# #연산자 범용 함수
# # a = np.random.randint(1, 10, size=(3,3))
# # print(a)

# # df = pd.DataFrame(a, columns=list('ABC'))
# # print(df)

# # # print(df + df.iloc[0])

# # # b = df.add(df.iloc[0])
# # # print(b)

# # print(df.sub(df.iloc[0]))
# # print(df.subtract(df['B'], axis = 0))

# # print(a*a[1])
# # print(df*df.iloc[1])
# # print(df.mul(df.iloc[1]))  #나누기는 div(), 나머지mode(),power() 지수

# # r = df.iloc[0,::2]
# # print(r)

# #정렬(sort)

# # s = pd.Series(range(5), index=['A','D','B','C','E'])
# # print(s)

# # a = s.sort_index()
# # print(a)
# # b = s.sort_values()
# # print(b)

# # df = pd.DataFrame(np.random.randint(0,10,(4,4)),
# #               index=[2,4,1,3],
# #               columns=list('BDAC'))
# # print(df)
# # print(df.sort_index())
# # print(df.sort_index(axis=1))
# # print(df.sort_values(by=['A', 'C']))

# #랭킹

# # s = pd.Series([-2,4,7,3,0,7,5,-4,2,6])
# # print(s)
# # print(s.rank())
# # print(s.rank(method='first')) #먼저 적힌놈이 더 높음
# # print(s.rank(method='max')) #같은 값 가진 그룹을 높은순위로 지정

# #고성능연산
# # nrows, ncols = 10000, 100
# # df1, df2, df3, df4 = (pd.DataFrame(np.random.rand(nrows, ncols))for i in range(4))

# # df = pd.DataFrame(n)

# #데이터 결합
# s1 = pd.Series(['a', 'b'], index= [1,2])
# s2 = pd.Series(['c', 'd'], index= [3,4])
# print(pd.concat([s1,s2]))

# def creat_df(cols, idx):
#      data = {c: [str(c.lower())+ str(i) for i in idx] for c in cols}
#      return pd.DataFrame(data, idx)

# # # df1 = creat_df('AB',[1,2])
# # # # print(df1)

# # # # df2 = creat_df('AB', [3,4])
# # # # print(df2)

# # # # a = pd.concat([df1,df2])
# # # # print(a)

# # # df3 = creat_df('AB', [0,1])
# # # print(df3)

# # # df4 = creat_df('CD', [0,1])
# # # print(df4)

# # # c = pd.concat([df3, df4])
# # # print(c)      #누락값이 생성

# # # d = pd.concat([df1, df3],verify_integrity=True) #에러확인해줌
# # # print(d)

# # # e = pd.concat([df1, df3], ignore_index=True) #강제로 병합
# # # print(e)

# # # f = pd.concat([df1, df3], keys=['X','Y']) #멀티인덱스처럼 인덱스추가
# # # print(f)

# # # df5 = creat_df('ABC', [1,2])
# # # df6 = creat_df('BCD', [3,4])
# # # g = pd.concat([df5, df6])
# # # # print(g)

# # # h = pd.concat([df5, df6], join='inner') #중복존재하는 값들만 합침
# # # print(h)

# # # i = df5.append(df6) # g = pd.concat([df5, df6]) 와 same result
# # # print(i)

# # # c = pd.concat([df3, df4], axis= 1) #축 지정가능
# # # print(c)

# # #병합과 조인
# # df1 = pd.DataFrame({'학생':['홍길동','이순신','임꺽정','김유신'],
# #                     '학과':['경영학과','교육학과','컴퓨터학과','통계학과']})

# # # print(df1)

# # df2 = pd.DataFrame({'학생':['홍길동','이순신','임꺽정','김유신'],
# #                     '입학년도':[2012,2016,2019,2020]})

# # # print(df2)

# # df3 = pd.merge(df1, df2) #이거좋네
# # print(df3)

# # df4 = pd.DataFrame({'학과':['경영학과','교육학과','컴퓨터학과','통계학과'],
# #                       '학과장':['황희','장영실','안창호','정약용']})

# # print(pd.merge(df3, df4))

# # df5 = pd.DataFrame({'학과':['경영학과','교육학과','교육학과','컴퓨터학과','컴퓨터학과','통계학과'],
# #                      '과목':['경영개론','기초수학','물리학','프록래밍','운영체제','확율론']})

# # print(pd.merge(df1, df5)) # 내츄럴조인같네
# # print(pd.merge(df1, df2, on='학생')) #기준을 학생으로 갈래

# # df6 = pd.DataFrame({'이름':['홍길동','이순신','임꺽정','김유신'],
# #                     '성적':['A','A+','B','A+']})

# # print(df6)
# # print(pd.merge(df6, df1, left_on='이름', right_on='학생')) #칼럼명이 다를때 사용
# # print(pd.merge(df1, df6, left_on='학생', right_on='이름').drop('이름',axis=1))

# # mdf1 = df1.set_index('학생')
# # mdf2 = df2.set_index('학생')

# # # print(mdf1)
# # # print(mdf2)
# # # print(pd.merge(mdf1, mdf2, left_index=True, right_index=True))

# # # print(mdf1.join(mdf2)) #위랑 같음
# # # print(pd.merge(df6, mdf1, right_index=True, left_on='이름'))
# # # print(pd.merge(mdf1, df6, left_index=True, right_on='이름'))

# # df7 = pd.DataFrame({'이름':['홍길동','이순신','임꺽정'],
# #                          '주문음식':['햄버거','피자','짜장면']})
# # print(df7)   

# # df8 = pd.DataFrame({'이름':['홍길동','이순신','김유신'],
# #                          '주문음료':['콜라','사이다','커피']})
# # print(df8)
# # print(pd.merge(df7, df8)) #홍길동 이순신만 나옴, 누락된거 무시
# # # print(pd.merge(df7, df8, how='inner')) #위랑같음
# # print(pd.merge(df7, df8, how='outer')) #누락된거도 나옴
# # print(pd.merge(df7, df8, how = 'left')) #df7에 있고 ,df8엔 없는
# # print(pd.merge(df7, df8, how= 'right'))

# # df9 = pd.DataFrame({'이름':['홍길동','이순신','임꺽정','김유신'],
# #                     '순위':[3,2,4,1]})

# # df10 = pd.DataFrame({'이름':['홍길동','이순신','임꺽정','김유신'],
# #                     '순위':[4,1,3,2]})

# # print(pd.merge(df9, df10, on='이름')) #순위_x ,순위_y
# # print(pd.merge(df9, df10, on='이름', suffixes=['_인기', '_성적']))

# #데이터집계/그룹연산
# # df = pd.DataFrame([[1,1.2,np.nan],
# #                   [2.4, 5.5, 4.2],
# #                  [np.nan, np.nan, np.nan],
# #                  [0.44, -3.1, -4.1]],
# #                   index=[1,2,3,4],
# #                   columns=['A','B','C'])                

# # print(df)
# # print(df.head(2))
# # print(df.tail(2))
# # print(df.describe())

# # print(df)
# # print(np.argmin(df))
# # print(np.argmax(df))
# # print(df.idxmin())
# # print(df.idxmax())
# # print(df.std())
# # print(df.var())
# # print(df.skew())
# # print(df.kurt())
# # print(df.sum())
# # print(df.cumsum())
# # print(df.prod())
# # print(df.cumprod())
# # print(df.diff()) #차
# # print(df.quantile())
# # print(df.pct_change())
# # print(df.corr())
# # print(df.corrwith(df.B))
# # print(df.cov()) #공분산
# # print(df['B'].unique())
# # print(df['A'].value_counts())

# #group by 연산
# df = pd.DataFrame({'c1':['a','a','b','b','c','d','b'],
#                    'c2':['A','B','B','A','D','C','C'],
#                    'c3': np.random.randint(7),
#                    'c4': np.random.random(7)})
# # print(df)
# # print(df.dtypes)
# # a = df['c3'].groupby(df['c1']).mean()
# # print(a)

# # b = df['c4'].groupby(df['c2']).std()
# # print(b)

# # c = df['c4'].groupby([df['c1'], df['c2']]).mean().unstack
# # print(c)

# # d = df.groupby('c1').mean()
# # print(d)

# # e = df.groupby(['c1', 'c2']).mean()
# # print(e)

# # f = df.groupby(['c1','c2']).size()
# # print(f)

# # for c1, group in df.groupby('c1'):
# #      print(c1)
# #      print(group)

# # for (c1,c2), group in df.groupby(['c1', 'c2']):
# #      print(c1)
# #      print(group)

# # g = df.groupby(['c1','c2'])[['c4']].mean()
# # print(g)
# # h = df.groupby('c1')['c3'].quantile()
# # print(h)

# # h = df.groupby('c1')['c3'].count()
# # print(h)

# # #  한번에 출력
# # i =df.groupby(['c1','c2'])['c4'].agg(['mean','min','max'])
# # print(i) 

# # j = df.groupby(['c1','c2'], as_index=False)['c4'].mean()
# # print(j)

# # k = df.groupby(['c1','c2'], group_keys=False)['c4'].mean()
# # print(k)

# # def top(df, n=3, column='c1'):
# #      return df.sort_values(by=column)[-n:]

# # l= top(df, n=5)
# # print(l)

# # m = df.groupby('c1').apply(top) #함수사용
# # print(m)

# # #피벗 테이블
# # a = df.pivot_table(['c3','c4'],
# #               index=['c1'],
# #               columns=['c2'])
# # print(a)

# # b = df.pivot_table(['c3','c4'],
# #               index=['c1'],
# #               columns=['c2'],
# #               margins=True)  #부분합 추가 
# # print(b)     

# # c = df.pivot_table(['c3','c4'],
# #               index=['c1'],
# #               columns=['c2'],
# #               margins=True,
# #               aggfunc=sum)
# # print(c)

# # d = df.pivot_table(['c3','c4'],
# #               index=['c1'],
# #               columns=['c2'],
# #               margins=True,
# #               aggfunc=sum,
# #               fill_value=0)
# # print(d)

# # e = pd.crosstab(df.c1, df.c2)
# # print(e)

# # d = pd.crosstab(df.c1, df.c2, values=df.c3, aggfunc=sum, margins=True)
# # print(d)

# # #범주형 데이터 #----------추가 학습 필요

# # s = pd.Series(['c1','c2','c1','c2','c1']* 2)
# # print(s)
# # e = pd.unique(s)
# # print(pd.value_counts(s))

# # code = pd.Series([0,1,0,1,0]* 2)
# # print(code)

# # d = pd.Series(['c1','c2'])
# # print(d.take(code))

# # df = pd.DataFrame({'id': np.arange(len(s)),
# #                    'c':s,
# #                    'v':np.random.randint(1000, 5000, size=len(s))})
# # print(df)   

# # c = df['c'].astype('category')
# # print(c.values.codes)

# # print(df['c'])

# # c = pd.Categorical(['c1','c2','c3','c2'])
# # print(c) 

# #문자열함수
# #문자열 관련된건 다 str로 접근.
# # name_tuple = ['Suan Lee','Steven Jobs','Larry PAGE','Elon Musk', None,'Bill gates','Mark Zuckerburg','Jeff bessus']
# # names= pd.Series(name_tuple)
# # print(name_tuple)

# # print(names.str.lower())

# # print(names.str.len())
# # print(names.str.split())


# # #기타연산자

# # print(names.str.split().str.get(-1))
# # print(names.str.repeat(2))
# # print(names.str.join('*'))

# # print(names.str.match('([A-Za-z]+)'))
# # print(names.str.findall('([A-Za-z]+)'))

# #시계열 처리 (금융,주식 많이쓰임)
# # idx = pd.DatetimeIndex(['2019-01-01','2020-01-01','2020-02-01','2020-02-02','2020-03-01'])
# # s = pd.Series([0,1,2,3,4], index=idx)
# # print(s['2020-01-01':])
# # print(s[:'2020-01-01'])
# # print([s['2019']]) #년도만 인덱싱가능

# #타임스탬프/기간/시간델타 or 지속 기간

# from datetime import date, datetime
# dates = pd.to_datetime(['12-12-2019', datetime(2020,1,1), '2nd of Feb, 2020','2020-Mar-04','20200707'])
# # print(dates)

# # print(dates.to_period('D')) #인덱스구조가 바뀜
# # print(dates - dates[0]) #타임델타인덱스됌
# # print(pd.date_range('2020-01-01', '2020-07-01')) #D단위로 레인지계산
# # print(pd.date_range('2020-01-01', periods=7))
# # print(pd.date_range('2020-01-01', periods=7, freq='M'))
# # print(pd.date_range('2020-01-01', periods=7, freq='H'))

# # idx = pd.to_datetime(['2020-01-01 12:00:00', '2020-01-02 00:00:00']+ [None] )
# # print(idx[2])
# # print(pd.isna(idx))

# #시계열 기본
# # dates = [datetime(2010,1,1), datetime(2020,1,2), datetime(2020,1,4), datetime(2020,1,7),
# # datetime(2020,1,10),datetime(2020,1,11),datetime(2020,1,15)]

# # print(dates)
# # ts = pd.Series(np.random.randn(7), index=dates)
# # print(ts)
# # print(ts.index) 
# # print(ts.index[0]) 
# # print(ts[ts.index[2]])
# # print(ts['20200104'])
# # print(ts['1/4/2020'])

# # ts = pd.Series(np.random.randn(1000),
# #                   index=pd.date_range('2017-10-01',periods=1000))
# # print(ts)  
# # print(ts['2020'])
# # print(ts['2020/06'])
# # print(ts[datetime(2020,6,20):])
# # print(ts['2020-06-10':'2020-06-20'])

# # -----------3:11:58

# tdf = pd.DataFrame(np.random.randn(1000,4),
#                    index=pd.date_range('2017-10-01', periods=1000),
#                    columns=['A','B','C','D'])
# # print(tdf['2018':])
# # print(tdf['C'])

# # ts = pd.Series(np.random.randn(10),
# #              index=pd.DatetimeIndex(['2020-01-01','2020-01-01','2020-01-02','2020-01-02','2020-01-03','2020-01-04','2020-01-05','2020-01-05','2020-01-06','2020-01-07']))

# # print(ts)    
# # print(ts.index.is_unique) #false 나옴 , 중복되니까
# # print(ts['2020-01-01'])

# # a = ts.groupby(level=0).mean() #중복날짜의 mean값 출력
# # print(a)

# # b = pd.date_range('2020-01-01', '2020-07-01')
# # b = pd.date_range(start='2020-01-01', periods=10)
# # b = pd.date_range(end='2020-07-01', periods=10)
# # b1 = pd.date_range('2020-07-01', '2020-07-7', freq='B') #주말은 뺌
# # print(b1)

# #주기와 오프셋

# # a = pd.timedelta_range(0, periods=12, freq='H')
# # a = pd.timedelta_range(0, periods=60, freq='T')
# # a = pd.timedelta_range(0, periods=10, freq='1H30T')
# # a = pd.date_range('2020-01-01', periods=20, freq='B')
# # a = pd.date_range('2020-01-01', periods=30, freq='S')
# # print(a)

# #시프트 연산
# # ts = pd.Series(np.random.randn(5),
# #                index=pd.date_range('2020-01-01', periods=5, freq='B'))
# # print(ts)
# # print(ts.shift(1)) #위에꺼에서 값들이 1칸씩 밀림
# # print(ts.shift(3))
# # print(ts.shift(-1)) #음수는 반대방향
# # print(ts.shift(3, freq='B'))
# # print(ts.shift(2, freq='W')) #주 단위

# #시간대 처리
# import pytz
# # print(pytz.common_timezones)

# tz = pytz.timezone('Asia/Seoul')
# dinx = pd.date_range('2020-01-01 09:00', periods=7, freq='B')
# ts = pd.Series(np.random.randn(len(dinx)), index=dinx)
# # print(ts)

# a = pd.date_range('2020-01-01 09:00',periods=7, freq='B', tz='UTC')
# # print(a)

# ts_utc = ts.tz_localize('UTC')
# # print(ts_utc)
# # print(ts_utc.tz_convert('Asia/Seoul')) #9시간 플러스됌

# ts_seoul = ts.tz_localize('Asia/Seoul')
# # print(ts_seoul)
# # print(ts_seoul.tz_convert('UTC'))
# # print(ts_seoul.tz_convert('Europe/Berlin'))
# # print(ts.tz_localize('America/New_York'))

# stamp = pd.Timestamp('2020-01-01 12:00')
# stamp_utc = stamp.tz_localize('UTC')
# # print(stamp_utc)

# # print(stamp_utc.value)
# # print(stamp_utc.tz_convert('Asia/Seoul'))
# # print(stamp_utc.tz_convert('Asia/Seoul').value)

# # stamp_ny = pd.Timestamp('2020-01-01 12:00', tz='America/New_York')
# # print(stamp_ny)
# # print(stamp_ny.value)
# # print(stamp_utc.tz_convert('Asia/Shanghai'))

# # stamp = pd.Timestamp('2020-01-01 12:00', tz='Asia/Seoul')

# # from pandas.tseries.offsets import Hour
# # print(stamp + Hour())
# # print(stamp + 3 * Hour())

# # ts1= ts_utc[:5].tz_convert('Europe/Berlin')
# # ts2= ts_utc[2:].tz_convert('America/New_York')
# # ts = ts1 + ts2
# # print(ts.index)

# #기간과 기간 연산

# # p = pd.Period(2020, freq= 'A-JAN')
# # print(p)
# # print(p+2)

# # p1 = pd.Period(2010, freq='A-JAN')
# # p2 = pd.Period(2020, freq='A-JAN')
# # print(p2-p1)

# # pr = pd.period_range('2020-01-01', '2020-06-30', freq='M')
# # print(pr)

# # a = pd.Series(np.random.randn(6), index=pr)
# # print(a)

# # pidx = pd.PeriodIndex(['2020-1', '2020-2','2020-4'], freq='M')
# # print(pidx)

# # p = pd.Period('2020', freq='A-FEB')
# # print(p)

# # a = p.asfreq('M', how='start')
# # print(a)
# # b = p.asfreq('M', how='end')
# # print(b)

# # p = pd.Period('2020', freq='A-OCT')
# # print(p)
# # print(p.asfreq('M', how='start'))
# # print(p.asfreq('M', how='end'))

# pr = pd.period_range('2010', '2020', freq='A-JAN')
# ts = pd.Series(np.random.randn(len(pr)), index=pr)
# # print(ts)

# # print(ts.asfreq('M', how='start'))
# # print(ts.asfreq('M', how='end'))

# # print(ts.asfreq('B', how='end'))

# # p = pd.Period('2020Q2', freq='Q-JAN')
# # print(p)
# # print(p.asfreq('D', 'start'))
# # print(p.asfreq('D', 'end'))

# # pr = pd.period_range('2020-01-01',periods=5, freq='Q-JAN')
# # ts = pd.Series(np.random.randn(5), index=pr)
# # print(ts)

# # pr = pd.date_range('2020-01-01', periods=5, freq='D')
# # ts = pd.Series(np.random.randn(5), index=pr)
# # print(ts)

# # p = ts.to_period('M')
# # print(p)
# # a = p.to_timestamp(how='start')
# # print(a)


# #리샘플링 (freq를변환)
# # 업샘플링, 다운샘플링

# # dr = pd.date_range('2020-01-01', periods=200, freq='D')
# # ts = pd.Series(np.random.randn(len(dr)),index=dr)
# # print(ts)

# # print(ts.resample('M').mean())
# # print(ts.resample('M', kind='period').mean())

# # dr = pd.date_range('2020-01-01', periods=10, freq='T')
# # ts = pd.Series(np.arange(10), index=dr)
# # print(ts)
# # print(ts.resample('2T', closed='left').sum())
# # print(ts.resample('2T', closed='right', label='right').sum()) #기준점이 다름 왼이냐 오냐

# # print(ts.resample('2T', closed='right', label='right', loffset='-1s').sum())
# # print(ts.resample('2T').ohlc())

# # df = pd.DataFrame(np.random.randn(10, 4),
# #                     index=pd.date_range('2019-10-01',periods=10, freq='M')
# #                     ,columns=['C1','C2','C3','C4'])
# # print(df)

# # print(df.resample('Y').asfreq())
# # print(df.resample('W-FRI').asfreq())
# # print(df.resample('H').asfreq()) # 개많음 안해
# # print(df.resample('H').ffill(limit=2)) #두개만 채움
# # print(df.resample('Q-DEC').mean())

# #무빙윈도우
# # df = pd.DataFrame(np.random.randn(300, 4),
# #                index=pd.date_range('2020-01-01', periods=300, freq='D'),
# #                columns=['C1','C2','C3','C4'])

# # print(df)
# # print(df.rolling(30).mean())
# # print(df.C1.rolling(60, min_periods=10).std().plot())

# #데이터 읽기 및 저장
# # import csv    
# # f = open('C:\\Users\\Hyeongkyu Park\\Desktop\\독학\\output.csv', 'w', encoding='utf-8', newline='')
# # wr = csv.writer(f)
# # wr.writerow(['a','b','c','d','e','txt'])
# # wr.writerow([1,2,3,4,5,'hi'])
# # wr.writerow([6,7,8,9,10,'pandas'])
# # wr.writerow([11,12,13,14,15,'csv'])
# # f.close()

# a = pd.read_csv('output.csv')
# # print(a)

# #헤더가 없을때
# # f = open('C:\\Users\\Hyeongkyu Park\\Desktop\\독학\\output2.csv', 'w', encoding='utf-8', newline='')
# # wr = csv.writer(f)
# # wr.writerow([1,2,3,4,5,'hi'])
# # wr.writerow([6,7,8,9,10,'pandas'])
# # wr.writerow([11,12,13,14,15,'csv'])
# # f.close()

# # a = pd.read_csv('output2.csv', header=None)
# # print(a)  #알아서 헤더생김

# # a = pd.read_csv('output2.csv', names=['a','b','c','d','e'])
# # print(a)

# # a = pd.read_csv('output2.csv', names=['a','b','c','d','e', 'text'], index_col='text')
# # print(a)

# # csv로 만들기
# # dr = pd.date_range('2020-01-01', periods=10)
# # ts = pd.Series(np.arange(10), index=dr)

# # a = ts.to_csv('ts.csv', header=['value'])
# # print(a)

# # import json
# # from collections import OrderedDict

# # file_data = OrderedDict()

# # file_data["name"] = "computer"
# # file_data["language"] = "kor"
# # file_data["words"] = {'a': 1, 'b': 2, 'processr':'프로세서'}
# # file_data["number"] = 4
# # file_data["a"] = 1
# # file_data["b"] = 2
# # # print(json.dumps(file_data, ensure_ascii=False, indent="\t"))

# #이진 데이터 파일 읽기 쓰기

# # df = pd.read_csv('output.csv')
# # df.to_pickle('df_pickle')
# # a = pd.read_pickle('df_pickle')
# # print(a)

# df = pd.DataFrame({'a': np.random.randn(100),
#                    'b': np.random.randn(100),
#                    'c': np.random.randn(100)
#                                     })
# # h = pd.HDFStore('data.h5')
# # h['obj1'] = df
# # h['obj1_col1'] = df['a']
# # h['obj1_col2'] = df['b']
# # h['obj1_col3'] = df['c']
# # # print(h['obj1'])

# # h.put('obj2', df, format='table')
# # # print(h.select('obj2', where = ['index > 50 and index <=60']))
# # h.close()

# # df.to_hdf('data.h5', 'obj3', format='table')
# # # print(pd.read_hdf('data.h5', 'obj3', where= ['index <10']))

# # df.to_excel('example.xlsx', 'Sheet1')
# # a = pd.read_excel('example.xlsx', 'Sheet1')
# # print(a)

# #데이터 정제
# #누락 데이터 처리
# # a = np.array([1,2,None,4,5])
# # print(a.sum())  #집계가 안댐;;

# # a = np.array([1, 2, np.nan, 4, 5])
# # # print(a.dtype)
# # # print(0 + np.nan)
# # # print(a.sum(), a.max())
# # # print(np.nansum(a))  #이건 가능

# # b = pd.Series([1,2,np.nan,4,None]) #NONE도 NAN됌
# # # print(b)

# # c = pd.Series(range(5), dtype=int)
# # c[0] = None
# # # c[3] = np.nan  # 또 none이 nan으로 바뀜
# # # print(c)

# # s = pd.Series([True, False, None, np.nan])
# # print(s)

# #Null값 처리
# # s = pd.Series([1,2,np.nan,'String',None])
# # print(s.isnull())
# # print(s[s.notnull()])
# # print(s.dropna())

# # print(df.dropna(axis='columns'))
# # df[3]= np.nan
# # print(df.dropna(axis='columns', how='all')) #null값 다 지움
# # print(df.dropna(axis='rows', thresh=3)) #쓰레쉬3이라 제거안댐

# # print(s)
# # print(s.fillna(0))
# # print(s.fillna(method='ffill')) #forwardfill
# # print(s.fillna(method='bfill'))

# # print(df)
# # print(df.fillna(method='ffill', axis=0))
# # print(df.fillna(method='ffill', axis=1))

# # print(df.fillna(method='bfill', axis=0))
# # print(df.fillna(method='bfill', axis=1))

# #중복제거
# # df = pd.DataFrame({'c1': ['a','b','c']*2+['b']+['c'],
# #                       'c2':[1,2,1,1,2,3,3,4]})
# # print(df)      
# # print(df.duplicated())  #중복확인 불리언으로 
# # print(df.drop_duplicates())  #중복제거     

# #값 치환

# # s = pd.Series([1., 2., -999., 3., -1000., 4.])
# # print(s)
# # print(s.replace(-999, np.nan))
# # print(s.replace(-1000, np.nan))
# # print(s.replace([-999, -1000], [np.nan, 0]))
