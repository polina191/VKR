import distributions as d
from scipy.optimize import fsolve
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

class TCL: #Theoretical Cramer-Lundberg Model
	def __init__(self, eta=d.distr(), N=d.distr()):
		if N.name != 'poisson':
			raise ValueError('only poisson frequency is supported')
	
		self.eta = eta
		self.N = N
		self.exact_ruin = -1
		self.adj_coef = -1
		self.u = 0
		self.theta = 0
		self.approx_ruin = -1

	def set_u(self, theta, psi):
		m1 = self.eta.moment(1)
		m2 = self.eta.moment(2)
		m3 = self.eta.moment(3)
		
		self.theta = theta 
		self.exact_ruin = psi
		
		if self.eta.mgf(0):
			self.adj_coef = fsolve(lambda x: self.eta.mgf(x)-(1+theta)*m1*x-1, 0.005*(1+theta))[0]
			
		self.u = fsolve(lambda u: psi - 1/(1+theta)*np.exp(-2*theta*m1*u/((1+theta)*m2))*(1+(2*m1*m3/(3*m2**2) -1)*(2*theta*m1*u/((1+theta)*m2) -1)*theta/(1+theta)), 1/psi)
	
	def set_psi(self, u, theta):
		m1 = self.eta.moment(1)
		m2 = self.eta.moment(2)
		m3 = self.eta.moment(3)

		self.u = u
		self.theta = theta
	
		if self.eta.mgf(0):
			self.adj_coef = fsolve(lambda x: self.eta.mgf(x)-(1+theta)*m1*x-1, 0.01*(2+theta))[0]
		
		if self.eta.name == 'exponential':
			self.exact_ruin = 1/(1+theta)*np.exp(-theta*u/(m1*(1+theta)))
		if u == 0:
			self.exact_ruin = 1/(1+theta)
			
		self.approx_ruin = 1/(1+theta)*np.exp(-2*theta*m1*u/((1+theta)*m2))*(1+(2*m1*m3/(3*m2**2) -1)*(2*theta*m1*u/((1+theta)*m2) -1)*theta/(1+theta))
			
		#self.temp = d.recursive(lambda x: 1/self.eta.moment(1)*(1-self.eta.cdf(x)), 1/(1+theta), 0,  u, self.eta.limit)
			
	def draw(self, theta, psi0=0.01, psi1=0.1):
		m1 = self.eta.moment(1)
		m2 = self.eta.moment(2)
		m3 = self.eta.moment(3)
	
		adj_exist = False
		
		if self.eta.mgf(0):
			adj_exist = True
			adj_coef = fsolve(lambda x: self.eta.mgf(x)-(1+theta)*m1*x-1, 0.005*(1+theta))[0]
			
		plt.figure(figsize=(20,10))
		
		approx_ruin = lambda u: 1/(1+theta)*np.exp(-2*theta*m1*u/((1+theta)*m2))*(1+(2*m1*m3/(3*m2**2) -1)*(2*theta*m1*u/((1+theta)*m2) -1)*theta/(1+theta))
			
		if adj_exist:
			u0 = -np.log(psi0)/adj_coef
			u1 = -np.log(psi1)/adj_coef
			
			u = np.linspace(u0, u1, 100)
			
			plt.plot(u, np.exp(-u*adj_coef), "red", alpha=0.4, label="Неравенство Лундберга")
		else:
			u0 = fsolve(lambda u: psi0 - approx_ruin(u), 1/psi0**2)[0]
			u1 = fsolve(lambda u: psi1 - approx_ruin(u), 2/psi1**2)[0]
			
			u = np.linspace(u0, u1, 100)
			
		plt.plot(u, approx_ruin(u), "green", alpha=0.6, label="Приближенная вероятность")
		
		if self.eta.name == 'exponential':
			exact_ruin = lambda u: 1/(1+theta)*np.exp(-theta*u/(m1*(1+theta)))
			plt.plot(u, exact_ruin(u), "blue", alpha=0.2, label="Точная вероятность")
			
		plt.title("Вероятность разорения", fontsize=16)
		plt.legend()
		plt.grid()
		plt.xlabel("u", fontsize=14)
		plt.ylabel("$\\psi(u)$", fontsize=14)
		plt.show()
			
	def result(self):
		c = (1+self.theta)*self.N.moment(1)*self.eta.moment(1)
		
		if self.approx_ruin != -1:
			approx_ruin = " Приближенная вероятность разорения: "
			ruin = self.approx_ruin
		else:
			approx_ruin = " Вероятность разорения: "
			ruin = self.exact_ruin
			
		lundb = " Неравенство Лундберга: "
		
		approx_ruin = approx_ruin + ' '*(37-len(approx_ruin)) + "{r:.5f}".format(r = round(ruin, 5))
		
		if self.adj_coef != -1:
			lundb = lundb + ' '*(37-len(lundb)) + "{r:.5f}".format(r = round(np.exp(-self.u*self.adj_coef), 5))
		else:
			lundb = lundb + ' '*(37-len(lundb)) + "{r:.5f}".format(r = 1)
	
		prem = " Размер премии: "
		start = " Стартовый капитал: "

		u_len = np.floor(np.log10(int(self.u)))+3
		c_len = np.floor(np.log10(int(c)))+3
		
		prem = prem + ' '*(33-len(prem)-int(c_len)) + "{r:.2f}".format(r = round(c, 5))
		start = start + ' '*(33-len(start)-int(u_len)) + "{r:.2f}".format(r = round(self.u, 5))
	
		print(' '+'-'*80+' ')
		print('|'+(' '*20)+"Модель Крамера-Лундберга (теоретическая)" + (' '*20) + '|')
		print('|'+(' '*80)+'|')
		print('|'+lundb+' |'+prem+'|')
		print('|'+approx_ruin+' |'+start+'|')
		print(' '+'-'*80+' ')
		
