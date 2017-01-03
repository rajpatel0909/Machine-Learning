import xlrd
import numpy
#import scipy
import math
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

file_location = "university data.xlsx"
workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_index(0)
#print(sheet.cell_value(0,0))
#print(sheet.nrows)
#print(sheet.ncols)
#for col in range(sheet.ncols):
#    print(sheet.cell_value(0, col))

#for row in range(sheet.nrows):
#    print(sheet.cell_value(row, 0))

print 'UBitName = rajjaysu'
print 'personNumber = 50208278'
#---------m1-cs------------------------------------------
cs_list = sheet.col_values(2)[1:(len(sheet.col_values(2))-1)]
cs_mu = round(numpy.mean(cs_list),3)
print 'mu1 =', cs_mu

#---------m2-ro------------------------------------------
ro_list = sheet.col_values(3)[1:(len(sheet.col_values(2))-1)]
ro_mu = round(numpy.mean(ro_list),3)
print 'mu2 =', ro_mu

#---------m3-abp------------------------------------------
abp_list = sheet.col_values(4)[1:(len(sheet.col_values(2))-1)]
abp_mu = round(numpy.mean(abp_list),3)
print 'mu3 =', abp_mu

#---------m2-tui------------------------------------------
tui_list = sheet.col_values(5)[1:(len(sheet.col_values(2))-1)]
tui_mu = round(numpy.mean(tui_list),3)
print 'mu4 =', tui_mu

#---------var1-CS-SCORE------------------------------------------
#cs_var = (sum(((value - cs_mu)**2) for value in cs_list))/(len(cs_list)-1)
cs_var = round(numpy.var(cs_list),3)
print 'var1 =', cs_var
#print("var1 = ",numpy.var(cs_list, ddof=1))

#---------var1-ro------------------------------------------
#ro_var = (sum(((value - ro_mu)**2) for value in ro_list))/(len(ro_list)-1)
ro_var = round(numpy.var(ro_list),3)
print 'var2 =', ro_var

#---------var1-abp------------------------------------------
#abp_var = (sum(((value - abp_mu)**2) for value in abp_list))/(len(abp_list)-1)
abp_var = round(numpy.var(abp_list),3)
print 'var3 =', abp_var

#---------var1-tui------------------------------------------
#tui_var = (sum(((value - tui_mu)**2) for value in tui_list))/(len(tui_list)-1)
#ddof=1
tui_var = round(numpy.var(tui_list),3)
print 'var4 =', tui_var

#---------sigma1-cs------------------------------------------
#cs_sigma = round(cs_var**(1/2.0),3)
cs_sigma = round(numpy.std(cs_list),3)
print 'sigma1 =', cs_sigma

#---------sigma1-ro------------------------------------------
#ro_sigma = round(ro_var**(1/2.0),3)
ro_sigma = round(numpy.std(ro_list),3)
print 'sigma2 =', ro_sigma

#---------sigma1-abp------------------------------------------
#abp_sigma = round(abp_var**(1/2.0),3)
abp_sigma = round(numpy.std(abp_list),3)
print 'sigma3 =', abp_sigma

#---------sigma1-tui------------------------------------------
#tui_sigma = round(tui_var**(1/2.0),3)
tui_sigma = round(numpy.std(tui_list),3)
print 'sigma4 =', tui_sigma

#---------covarianceMat and correlationMat------------------------------------------
data_mat = [cs_list, ro_list, abp_list, tui_list]
covarianceMat = numpy.around(numpy.cov(data_mat),3)
print 'covarianceMat ='
print covarianceMat
correlationMat = numpy.around(numpy.corrcoef(data_mat),3)
print 'correlationMat ='
print correlationMat
'''
plt.figure(1)
plt.subplot(321)
plt.scatter(cs_list,ro_list)
plt.xlabel("CS Score")
plt.ylabel("Research Overhead")
plt.subplot(322)
plt.scatter(cs_list,abp_list)
plt.xlabel("CS Score")
plt.ylabel("Admin Base Pay")
plt.subplot(323)
plt.scatter(cs_list,tui_list)
plt.xlabel("CS Score")
plt.ylabel("Tuition")
plt.subplot(324)
plt.scatter(ro_list,abp_list)
plt.xlabel("Research Overhead")
plt.ylabel("Admin Base Pay")
plt.subplot(325)
plt.scatter(ro_list,tui_list)
plt.xlabel("Research Overhead")
plt.ylabel("Admin Base Pay")
plt.subplot(326)
plt.scatter(abp_list,tui_list)
plt.xlabel("Admin Base Pay")
plt.ylabel("Tuition")
plt.show()


# Generate a mask for the upper triangle
#mask = numpy.zeros_like(correlationMat, dtype=numpy.bool)
#mask[numpy.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlationMat, cmap=cmap, vmax=0.2,square=True, xticklabels=5, yticklabels=5,linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.xlabel("adf")
plt.show()
'''
#p1 = scipy.stats.norm.logpdf(cs_mu, cs_sigma)
#p2 = scipy.stats.norm.logpdf(ro_mu, ro_sigma)
#p3 = scipy.stats.norm.logpdf(abp_mu, abp_sigma)
#p4 = scipy.stats.norm.logpdf(tui_mu, tui_sigma)

p1 = []
for i in cs_list:
	p1.append(math.exp(-1*((((i - cs_mu)/(cs_sigma))**2)/2))/(((2*math.pi)**0.5)*cs_sigma))
	

p1 = math.log(reduce(lambda x, y: x*y, p1))

p2 = []
for i in ro_list:
	p2.append(math.exp(-1*((((i - ro_mu)/(ro_sigma))**2)/2))/(((2*math.pi)**0.5)*ro_sigma))
	
p2 = math.log(reduce(lambda x, y: x*y, p2))

p3 = []
for i in abp_list:
	p3.append(math.exp(-1*((((i - abp_mu)/(abp_sigma))**2)/2))/(((2*math.pi)**0.5)*abp_sigma))
	
p3 = math.log(reduce(lambda x, y: x*y, p3))

p4 = []
for i in tui_list:
	p4.append(math.exp(-1*((((i - tui_mu)/(tui_sigma))**2)/2))/(((2*math.pi)**0.5)*tui_sigma))
	
p4 = math.log(reduce(lambda x, y: x*y, p4))
loglike = round(p1+p2+p3+p4,3)
print 'loglikelihood =', round(p1+p2+p3+p4,3)


BNgraph = numpy.matrix('0,0,0,0;1,0,0,1;0,1,0,1;1,0,0,0')
print 'BNgraph ='
print BNgraph

x0 = []
i = 0
for x in cs_list:
	x0.append(1);
#0-x0 1- cs_list 2-ro_list 3-abp_list 4-tui_list	
#------1--------------------------------------------------------------------
test1 = [x0,ro_list,tui_list]
#print 'test1[1]',numpy.shape(test1[1])
A1 = [[0 for x in range(len(test1))] for y in range(len(test1))]
for i in range(0,len(test1)):
	for j in range(0,len(test1)):
		A1[i][j] = sum(numpy.multiply(test1[i],test1[j]))
#print 'A1=',numpy.shape(A1)
#print A1

y1 = [0 for x in range(len(test1))]
for i in range(0,len(y1)):
	y1[i] = sum(numpy.multiply(cs_list,test1[i]))

#print 'y1', y1
#print 'y1=',numpy.shape(y1)
	

b1 = numpy.dot(numpy.linalg.inv(A1),y1)
#print 'b1=',numpy.shape(b1)	
#print 'b1=', b1
#print 'b1=', b1[0], b1[1], b1[2]

l1 = 0
bx_sum = 0
bx = 0
for i in range(0,len(cs_list)):
	bx_sum += ((b1[0] + b1[1]*test1[1][i] + b1[2]*test1[2][i] - cs_list[i])**2)
l1 += ((-49*(math.log(2*math.pi*cs_var))/2) - ((bx_sum)/(2*cs_var)))
'''

for i in range(0,49):
	l1 =+ (((-1*(math.log(2*math.pi*cs_var)))/2) - (((b1[0] + b1[1]*ro_list[i] + b1[2]*tui_list[i] - cs_list[i])**2)/(2*cs_var)))

'''
#print 'l1=', l1	

#----2------------------------------------------------------------------
	
test2 = [x0,abp_list]
A2 = [[0 for x in range(len(test2))] for y in range(len(test2))]
for i in range(0,len(test2)):
	for j in range(0,len(test2)):
		A2[i][j] = sum(numpy.multiply(test2[i],test2[j]))
#print 'A2=',numpy.shape(A2)
#print 'A2=',A2
y2 = [0 for x in range(len(test2))]
for i in range(0,len(test2)):
	y2[i] = sum(numpy.multiply(ro_list,test2[i]))

#print 'y2=',numpy.shape(y2)	
#print 'y2=',y2
b2 = numpy.dot(numpy.linalg.inv(A2),y2)
#print 'b2=',numpy.shape(b2)		
#print 'b2=',b2

l2 = 0
bx_sum = 0
bx = 0
for i in range(0,len(ro_list)):
	bx_sum += ((b2[0] + b2[1]*test2[1][i] - ro_list[i])**2)

l2 += ((-49*(math.log(2*math.pi*ro_var))/2) - ((bx_sum)/(2*ro_var)))

#print 'l2=', l2		

#------3------------------------------------------------------------------

test3 = [x0]
A3 = [[0 for x in range(len(test3))] for y in range(len(test3))]
for i in range(0,len(test3)):
	for j in range(0,len(test3)):
		A3[i][j] = sum(numpy.multiply(test3[i],test3[j]))
#print 'A3=',numpy.shape(A3)
#print 'A3=',A3
y3 = [0 for x in range(len(test3))]
for i in range(0,len(test3)):
	y3[i] = sum(numpy.multiply(abp_list,test3[i]))

#print 'y3=',numpy.shape(y3)	
#print 'y3=',y3
b3 = numpy.dot(numpy.linalg.inv(A3),y3)
#print 'b3=',numpy.shape(b3)		
#print 'b3=',b3	

l3 = 0
bx_sum = 0
bx = 0
for i in range(0,len(abp_list)):
	bx_sum += ((b3[0] - abp_list[i])**2)

l3 += ((-49*(math.log(2*math.pi*abp_var))/2) - ((bx_sum)/(2*abp_var)))

#print 'l3=', l3	
#print 'p3=',p3
	
#--------4----------------------------------------------------------------------	
test4 = [x0,ro_list, abp_list]
A4 = [[0 for x in range(len(test4))] for y in range(len(test4))]
for i in range(0,len(test4)):
	for j in range(0,len(test4)):
		A4[i][j] = sum(numpy.multiply(test4[i],test4[j]))
#print 'A4=',numpy.shape(A4)
#print 'A4=',A4
y4 = [0 for x in range(len(test4))]
for i in range(0,len(test4)):
	y4[i] = sum(numpy.multiply(tui_list,test4[i]))

#print 'y4=',numpy.shape(y4)
#print 'y4=',y4
b4 = numpy.dot(numpy.linalg.inv(A4),y4)
#print 'b4=',numpy.shape(b4)	
#print 'b4=',b4	

l4 = 0
bx_sum = 0
bx = 0
for i in range(0,len(tui_list)):
	bx_sum += ((b4[0] + b4[1]*test4[1][i] + b4[2]*test4[2][i] - tui_list[i])**2)

l4 += ((-49*(math.log(2*math.pi*tui_var))/2) - ((bx_sum)/(2*tui_var)))


#print 'l4=', l4	
	
print 'BNloglikelihood =',round(l1+l2+l3+l4,3)
