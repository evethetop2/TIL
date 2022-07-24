import numpy as np
# a1 = np.array([1,2,3,4,5])
# print(a1)
# print(type(a1))
# print(a1.shape) #(5,) = 1차원배열임
# print(a1[0],a1[1],a1[2])
# a1[0]=4
# a1[1]=5
# print(a1)


# 2차원
# a2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
# # print(a2)
# # print(a2.shape)
# # print(a2[0,0],a2[1,1],a2[2,2])

# # 3차원
# a3 = np.array([[[1,2,3],[4,5,6],[7,8,9]],
                 [[1,2,3],[4,5,6],[7,8,9]],
                 [[1,2,3],[4,5,6],[7,8,9]] ])
# print(a3)
# print(a3.shape)

# 배열 생성 및 초기화
# a = np.zeros(10,10,10)
# print(a)

# b = np.ones(10) # 1로 채운거
# c = np.ones((3,3))
# print(c)

# d = np.full((3,3), 1.23)
# print(d)

# # 단위행렬
# f =np.eye(3)
# print(f)
# g = np.tri(3) #삼각행렬

# h = np.empty(10) # 초기화되지 않은 배열생성
# print(h)

# print(a1)
# b = np.zeros_like(a1) # a1 쉐입따서 만들어줘
# print(b)

# 생성한 값으로 배열 생성
# a =np.arange(0, 30, 2)
# print(a)
# b = np.linspace(0, 1, 5)
# print(b)
# c = np.logspace(0.1, 1, 20)
# print(c)

# 랜덤값으로 배열 생성
# a = np.random.random((3,3))
# print(a)
# b = np.random.randint(0, 10, (3,3))
# print(b)
# c = np.random.normal(0,1,size=(3,3)) #정규분포
# print(c)
# f = np.random.randn(3,3) #표준정규분포
# print(f)

#데이터타입
# a = np.zeros(20, dtype=int)
# print(a)
# b = np.ones((3,3), dtype=bool)
# print(b)
# c = np.full((3,3), 1.0 ,dtype=float)
# print(c)

#날짜/시간 배열 생성
# date = np.array('2020-01-01', dtype=np.datetime64)
# print(date)
# b = date + np.arange(12)
# print(b)

# datetime = np.datetime64('2020-06-01 12:00')
# print(datetime)

#배열조회

# def array_info(array):
#     print(array)
#     print("ndim: ", array.ndim)
#     print("shape: ", array.shape)
#     print("dtype: ", array.dtype)
#     print("size: ", array.size)
#     print("itemsize: ", array.itemsize)
#     print("nbytes: ", array.nbytes)
#     print("strides: ", array.strides)



# 인덱싱
# print(a1)
# print(a1[0])
# print(a1[2])
# print(a1[-1])
# print(a1[-2])

# print(a2)
# print(a2[0, 0])
# print(a2[0, 2])
# print(a2[1,1])
# print(a2[2,-1])

# print(a3)
# print(a3[0, 0, 0])
# print(a3[1,1,1])
# print(a3[2,2,2])
# print(a3[2,-1,-1])

# 슬라이싱
# print(a1)
# print(a1[0:2])
# print(a1[0:])
# print(a1[:1])
# print(a1[::2])
# print(a1[::-1])

# print(a2)
# print(a2[1])
# print(a2[1, :])
# print(a2[2, :2])
# print(a2[1:, ::-1])
# print(a2[::-1, ::-1])

#불리언 인덱싱
# print(a1)
# bi = [False, True, True, False, True]
# print(a1[bi])
# bi = [True, False, True, True, False]
# print(a1[bi])

# print(a2)
# bi = np.random.randint(0,2,(3,3), dtype=bool)
# print(bi)
# print(a2[bi])

#팬시 인덱싱
# print(a1)
# print(a1[0], a1[2])
# ind = [0,2]
# print(a1[ind])
# print(a1)
# ind = np.array([[0,1],
#                [2,0]])
# print(a1[ind])
# print(a2)
# row = np.array([0,2])
# col = np.array([1,2])
# print(a2[row,col])
# print(a2[row, :])
# print(a2[:, col])
# print(a2[row, 1])
# print(a2[2, col])
# print(a2[row, 1:])
# print(a2[1:, col])

#배열값 삽입/수정/삭제/복사

# print(a1)
# b1 = np.insert(a1, 0, 10)
# print(b1)
# print(a1) #원본변경안됌

# c1 = np.insert(a1, 2, 10)
# print(c1)

# print(a2)
# b2 = np.insert(a2, 2, 10, axis = 0) #row
# print(b2)

# c2 = np.insert(a2, 2, 10, axis = 1) #col
# print(c2)

#배열 값 수정
# print(a1)
# a1[0]= 1
# a1[1] = 2
# a1[2] = 3
# print(a1)

# a1[:1] = 9
# print(a1)
# i = np.array([1,3,4])
# a1[i] = 0
# print(a1)
# a1[i] += 4
# print(a1)

# 2차원
# print(a2)
# a2[0,0] = 1
# a2[1,1] = 2
# a2[2,2] = 3
# a2[0] = 1 # 전체 열 바뀜
# print(a2)
# row = np.array([0,1])
# col = np.array([1,2])
# a2[row,col]=0
# print(a2)

#배열값 삭제
# print(a1)
# b1 = np.delete(a1, 1)
# print(b1) #원본 그대로임
# print(a1)

# print(a2)
# b2 = np.delete(a2, 1, axis = 0)
# print(b2)
# c2 = np.delete(a2, 1, axis = 1)
# print(c2)


#배열 복사
# print(a2)
# print(a2[:2, :2])
# a2_sub = a2[:2, :2]
# print(a2_sub)
# a2_sub[:, 1] = 0
# print(a2_sub)
# print(a2) #원본도 바뀌었네?, 리스트 = 슬라이싱 하면 원본과 상관없는 놈이 리턴됨, 반면 넘파이는 원본이 지켜짐(동일한 메모리 위치)

# print(a2)
# a2_sub_copy= a2[:2,:2].copy() #배열이나 하위배열내의 값을 명시적으로 복사
# print(a2_sub_copy)
# a2_sub_copy[:, 1] = 1
# print(a2_sub_copy)
#   print(a2) #원본은 안바뀜!

#배열 변환

#배열 전치 및 축 변경 
# print(a2)
# print(a2.T) 

# # print(a3)
# # print(a3.T)

# print(a2)
# print(a2.swapaxes(1,0))
# print(a2.swapaxes(0,1))

#배열 재구조화
# n1 = np.arange(1,10)
# # print(n1)
# # print(n1.reshape(3, 3))

# print(n1)
# print(n1[np.newaxis, :5])
# print(n1[:5, np.newaxis])

#배열 크기변경
# n2 = np.random.randint(0, 10, (2, 5))
# print(n2)
# n2.resize(5,2)
# print(n2)

# n2.resize(5,5)
# print(n2) # 남은 공간에 0이 추가

# n2.resize(3,3)
# print(n2) # 삭제됌

#배열추가
# a2 = np.arange(1,10).reshape(3,3)
# print(a2)
# b2 = np.arange(10,19).reshape(3,3)
# print(b2)

# c2 = np.append(a2, b2) #axis지정 안하면 1차원형태로 됌
# print(c2)

# c2 = np.append(a2, b2, axis = 0) #axis지정 안하면 1차원형태로 됌
# print(c2)

# c2 = np.append(a2, b2, axis = 1) #axis지정 안하면 1차원형태로 됌
# print(c2)

#배열연결
# a1 = np.array([1,3,5])
# b1 = np.array([2,4,6])
# c1 = np.concatenate([a1, b1])
# print(c1)

# a2 = np.array([[1,2,3],[4,5,6]])
# c2 = np.concatenate([a2,a2], axis = 1)
# print(c2)

#stack 도 있음

#배열분할
# a1 = np.arange(0,10)
# print(a1)
# b1, c1 = np.split(a1, [5])
# print(b1, c1)

# b1 , c1, d1, f1, g1 = np.split(a1, [2,4,6,8])
# print(b1 , c1, d1, f1, g1)

#vsplit, hsplit, dsplit 도 있음

#배열연산 (넘파이는 벡터연산 실행)

#브로드캐스팅
# a1 = np.array([1,2,3])
# a2 = np.arange(1,10).reshape(3,3)
# print(a1+a2)

# b2 = np.array([1,2,3]).reshape(3,1)
# print(b2)
# print(a1+b2)

#Artihmetic Operators (산술 연산) one of the unverisial functions

# a1 = np.arange(1,10)
# print(a1)
# print(a1+1)
# print(np.add(a1, 10))
# print(a1 - 2)
# print(np.subtract(a1, 10))
# print(-a1)
# print(np.negative(a1))
# print(a1 *2)
# print(np.multiply(a1, 2))
# print(np.divide(a1, 2))
# print(np.floor_divide(a1, 2)) # 반내림
# print(a1 **2)
# print(np.power(a1, 2))
# print(np.mod(a1,2)) #나머지 

# a1 = np.arange(1, 10)
# print(a1)
# b1 = np.random.randint(1,10, size=9)
# print(b1)
# print(a1 + b1)
# print(a1 - b1)
# print(a1 * b1)
# print(a1 / b1)
# print(a1 // b1)
# print(a1 % b1)
# print(a1 ** b1)

# a2 = np.arange(1,10).reshape(3,3)
# print(a2)
# b2 = np.random.randint(1,10, size=(3,3))
# print(b2)
# print(a2 + b2)
# print(a2 - b2)
# print(a2 * b2)
# print(a2 / b2)
# print(a2 // b2)
# print(a2 - b2)

#절대값 함수
# a1 = np.random.randint(-10, 10, size = 5)
# print(a1)
# print(np.absolute(a1))

# #제곱/제곱근 함수
# print(a1)
# print(np.square(a1))
# print(np.sqrt(a1))

#지수와 로그함수
# a1 = np.random.randint(1,10,size=5)
# print(a1)
# print(np.exp(a1))
# print(np.exp2(a1))
# print(np.power(a1, 2))

# print (np.log(a1))
# print(np.log2(a1))
# print(np.log10(a1))

#삼각함수
# t = np.linspace(0, np.pi, 3)
# print(t)
# print(np.sin(t))
# print(np.cos(t))
# print(np.cos(t))

# x = [-1, 0, 1]
# print(x)
# print(np.arcsin(x))
# print(np.arccos(x))
# print(np.arctan(x))

#집계함수(aggregate function)
# a2 = np.random.randint(1,10, size = (3,3))
# # print(a2)
# # print(a2.sum(),np.sum(a2))
# # print(a2.sum(axis=0), np.sum(a2,axis= 0))
# # print(a2.sum(axis=1), np.sum(a2,axis= 1))

# #누적함수
# print(a2)
# print(np.cumsum(a2))
# print(np.cumsum(a2,axis= 0))
# print(np.cumsum(a2,axis= 1))

# 차분계산
# print(a2)
# print(np.diff(a2))
# print(np.diff(a2, axis=0))
# print(np.diff(a2, axis=1))

#곱 계산
# print(a2)
# print(np.prod(a2))
# print(np.prod(a2, axis=0))
# print(np.prod(a2, axis=1))

#점곱/행렬곱
# print(a2)
# b2 = np.ones_like(a2)
# print(b2)
# print(np.dot(a2,b2))
# print(np.matmul(a2,b2))

#텐서곱 계싼
# print(a2)
# print(b2)
# print(np.tensordot(a2,b2)) #그냥 곱
# print(np.tensordot(a2, b2, axes = 0))
# print(np.tensordot(a2, b2, axes = 1)) #dotpro랑 같음

#벡터곱
# x = [1,2,3]
# y = [4,5,6]
# print(np.cross(x,y))

#내적/외적
# print(a2)
# print(b2)
# print(np.inner(a2, b2)) #dotpro랑 같은 결과
# print(np.outer(a2, b2))

#평균 계싼
# print(a2)
# print(np.mean(a2))
# print(np.mean(a2, axis=0))
# print(np.mean(a2, axis=1))

#std(),var(),min(),max(),argmin(),argmax() 가 있다

#백분위수
# a1 = np.array([0,2,4,6])
# print(a1)
# print(np.percentile(a1, [0,20,40,60,80,100], interpolation='linear'))
# print(np.percentile(a1, [0,20,40,60,80,100], interpolation='higher'))
# print(np.percentile(a1, [0,20,40,60,80,100], interpolation='lower'))
# print(np.percentile(a1, [0,20,40,60,80,100], interpolation='nearest'))
# print(np.percentile(a1, [0,20,40,60,80,100], interpolation='midpoint'))

#any() #하나라도 참이면 참
# a2=np.array([[False,False, False],
#              [False, True, True],
#              [False,True,True]])
# print(a2)
# print(np.any(a2))
# print(np.any(a2, axis=0))
# print(np.any(a2, axis=1))

#all #다 참이여야 참
# print(np.all(a2))
# print(np.all(a2, axis=0))
# print(np.all(a2, axis=1))

#비교연산
# a1 = np.arange(1,10)
# print(a1)
# print(a1 == 5)
# print(a1 != 5)
# print(a1 > 5)
# print(a1 >=5)
# print(a1 <5)
# print(a1 <= 5)

#불리언연산자
# a2 = np.arange(1,10).reshape(3,3)
# print(a2)

# print((a2>5) & (a2 < 8))
# print(a2[(a2>5) & (a2 < 8)])

# print((a2>5) | (a2 < 8))
# print(a2[(a2>5) | (a2 < 8)])

# print((a2>5) ^ (a2 < 8))
# print(a2[(a2>5) ^ (a2 < 8)])

# print(~(a2>5))
# print(a2[~(a2>5)])

#배열정렬
# a1 = np.random.randint(1,10,size=10)
# print(a1)
# print(np.sort(a1))
# print(a1)  #원본 그대로나옴
# print(np.argsort(a1))
# print(a1) #원본 유지
# print(a1.sort())
# print(a1)

#부분정렬
# a1 = np.random.randint(1,10, size=10)
# print(a1)
# print(np.partition(a1, 3))

#배열입출력
# a2 = np.random.randint(1,10,size=(5,5))
# print(a2)
# np.save("a", a2)

# b2 = np.random.randint(1,10, size=(5,5))
# print(b2)
# np.savez("ab",a2, b2)

# npy = np.load("a.npy")
# print(npy) # 불러오기
# npz = np.load("ab.npz")
# print(npz.files)
# print(npz['arr_0'])
# print(npz['arr_1'])

# print(a2)
# np.savetxt("a.csv", a2, delimiter= ',')
# !ls
# !cat.a.csv

# csv = np.loadtxt("a.csv", delimiter=',') #불러오기
# print(csv)
# print(b2)
# np.savetxt("b.csv", b2,  delimiter=',',fmt='%.2e', header='c1,c2,c3,c4,c5')
# # !cat b.csv
# csv= np.loadtxt("b.csv", delimiter=',')
# print(csv)









