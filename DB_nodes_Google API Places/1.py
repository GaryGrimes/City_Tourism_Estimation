a, b, c = 1, 2, 'john'
type(a)
type(c)
 

list = [ 'runoob', 786 , 2.23, 'john', 70.2 ]
tinylist = [123, 'john']
 
print(list)               # 输出完整列表
print(list[0])            # 输出列表的第一个元素
print(list[1:3])          # 输出第二个至第三个元素 
print(list[::2])           
print(list[2:], 'like this')           # 输出从第三个开始至列表末尾的所有元素
print(tinylist * 2)       # 输出列表两次
print(list + tinylist)    # 打印组合的列表

dict = {}
dict['one'] = "This is one"
dict[2] = "This is two"
 
tinydict = {'name': 'john','code':6734, 'dept': 'sales'}
 
 
print(dict['one'])          # 输出键为'one' 的值
print(dict[2])              # 输出键为 2 的值
print(tinydict)             # 输出完整的字典
print(tinydict.keys())      # 输出所有键
print(tinydict.values())    # 输出所有值

x = 'myname'
for i in range(len(x)):
    print(ord(x[i]))

a = 10
b = 20
list = [1, 2, 3, 4, 5 ];
 
if ( a in list ):
   print("1 - 变量 a 在给定的列表中 list 中")
elif (str(b)[0] == '2'): 
    print("elif test successful") # str是一个内置函数，定义变量的时候千万不要把它覆盖了
else:
   print("1 - 变量 a 不在给定的列表中 list 中")
 

print("=======欢迎进入狗狗年龄对比系统========")
while True: # 无限循环
    try:
        age = int(input("请输入您家狗的年龄:"))
        print(" ")
        age = float(age)
        if age < 0:
            print("您在逗我？")
        elif age == 1:
            print("相当于人类14岁")
            break  # 直到有结果，跳出循环
        elif age == 2:
            print("相当于人类22岁")
            break
        else:
            human = 22 + (age - 2)*5
            print("相当于人类：",human)
            break
    except ValueError:
        print("输入不合法，请输入有效年龄")
###退出提示
input("点击 enter 键退出")

print('----欢迎使用BMI计算程序----')
name=input('请键入您的姓名:')
height=eval(input('请键入您的身高(m):'))
weight=eval(input('请键入您的体重(kg):'))
gender=input('请键入你的性别(F/M)')
BMI=float(float(weight)/(float(height)**2)) # input()函数默认为str类型。需要获得完整的除法结果需要为float，非整形
#公式
if BMI<=18.4:
    print('姓名:',name,'身体状态:偏瘦')
elif BMI<=23.9:
    print('姓名:',name,'身体状态:正常')
elif BMI<=27.9:
    print('姓名:',name,'身体状态:超重')
elif BMI>=28:
    print('姓名:',name,'身体状态:肥胖')
import time;

#time模块
nowtime=(time.asctime(time.localtime(time.time())))
if gender=='F' or gender=='f':
    print('感谢',name,'女士在',nowtime,'使用本程序,祝您身体健康!')
if gender=='M' or gender=='m':
    print('感谢',name,'先生在',nowtime,'使用本程序,祝您身体健康!')
