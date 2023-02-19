#---print and Variables----#
''' print ("hello Ali ")
print ("hello \n Ali ")

name=("Ali")
print ("hello" + name)
 '''
#--- input and Variables ---#
''' x=input("x= ") '''
''' y=input("y= ") '''
''' z=int(y)+int(x) '''
''' print(z) '''

#----Function-----#

''' def Douha( ):
    name=input ("Enter you name: ")
    job=input ("Enter you job: ")
    print ("Hello"+ name+ "you are" +job)
    Douha()


def cube(num):
    return num*num*num#
    print (cube(2)) '''


#-----lest and Matrices ---#
"""  Computer=["HP","TOSHIBA", "MACBOOK"]
 Computer[0]
 Computer[0:2]
 Computer[1:]
 Computer[-1]
 Computer[-2]
 Computer[2]="MACBOOK2"
 print(Computer)  """ 

''' cars= ["BMW", "NASSAN","MOSTOBESHE" , "CADELACK"]
speed=[100,200,400]
 print(cars)

index_car=cars.index("BMW")
count_car=car.count("NASSAN")
cars.pop()
cars.remove("BMW")
cars.clear()
cars.extend(speed)
cars.append("kia")
cars.insert (2, "kia ")
cars.sort()
cars.reverse()
cars.copy()
coloer("red", "geen", "black ") '''

#--------IF stetment --------#
''' is_man= False 

if is_man:
    print("your are man")

man=True
arab=False
if man and arab: 
    print("your are a Arab man")
elif man and not (arab):
    print("your are a man but not arab")
elif not (man)and (arab):
    print("your are a Arab woman")
elif not (arab)and not (man):
    print("your are a not Arab woman")
else:
    print("your are not a Arab man")                    



def max(x,y,z):
    if x>=y and x>=z:
      return x
    elif  y>=x and y>=z:
      return x

    else:
        return z
    print(max(100,20,12)) '''

''' x=float(input("Enter x: "))
op= (input("operetion: "))
y=float(input("Enter y: "))

if op=="+":
   print (x+y)
elif op=="-":
     print (x+y)
elif op=="*":
    print (x*y)
elif op=="/":
    print (x/y)
else:
    print ("Please Enter Operetion") '''

#---القاموس------#
''' Gared={100:"A+",90:"A",80:"B",70:"C"}
print(Gared[100])
print(Gared.get(110, "Please Try aign")) '''

#-------loop------#
''' i=0
while i <= 10:
   print(i)
    i+=2
if i > 10:
    print ("END")
 
for letter in "Ali Alhajj":
    print (letter) '''
    
#-----for-----#
''' ars= ["BMW", "NASSAN","MOSTOBESHE" , "CADELACK"]
for car in (cars):
    print (car)
   
for x in range(5,100):
   print (x)
 '''
#----lop ex----#
''' passworld ="Ali"
word=" "
count=0
limit= 3
out= False
while passworld != word and not(out):
    if count <limit:
         word=input("Enter word: ")
        count+=1
    else:
        out= True
if out:
    print ("out of try")   
else:
      print ("correct ") '''

#=---2Dlist ----#
''' gid = [ 
    [1,2,3],
    [4,5,6],
    [7,8,9]
    ]
print(gid[2][0])

for row in gid:
    for col in row:
     print()
 '''
#------file in pythom----#

''' file = open("file for python.txt","w")
file = open("index.txt","w")
print (file.readable())
print (file.read())
print(file.readline())
print(file.readlines())
file.write("Ahmed=lawer \nAbdurahman = CEO")
file.write("welcome in company ")
print (file.writable)
for hummn in file.readlines():
    print(hummn)

file.close() 

#-------Import -------#

''' import Man_file
print(Man_file.country)
 '''
#-----Class----#

''' ''' ''' class phone:
    def __init__(self, modle, size , is_toush , price):
        self.modele =modle
        self.is_toush = is_toush
        self.size = size
        self.price = price
    
    def is_toushale(self):
        if self.is_toush:
            return True
        else:
           return False

class tablet(phone): #للوراثة نستخدم قوسين بعد الكلاس ونكتب الكلاس الام
    def __init__(self, modle, size , is_toush , coloer):
        self.modele =modle
        self.is_toush = is_toush
        self.size = size
        self.coloer = coloer

    def is_toushale(self):
        if self.is_toush:
            return True
        else:
           return False '''



