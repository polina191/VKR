import numpy as np
import scipy.integrate as integrate
from scipy.special import comb, gammainc, gamma
from scipy.fft import fft, ifft

norm_const = 1/np.sqrt(2*np.pi)

def discretize_pdf(pdf, c=1000, n=10000):
	g = []
	
	for k in range(n):
		g.append(pdf(c*(k+1/2)/n)*c/n)
	
	g.append(1-sum(g))
	
	return g

def recursive(pdf, a, b, u, limit=np.inf):
	if limit == np.inf or limit <= 0:
		return None
	
	n=1000
	
	g = discretize_pdf(pdf, limit, n)
	g.extend([0 for i in range(int(u*n/limit)-n)])

	f = []
	
	if a != 0 and (a != -b):
		f.append(((1-a)/(1-a*g[0]))**(1+b/a))
	elif a == -b:
		f.append(np.log(1-a*g[0])/np.log(1-a))
		
	for k in range(1, int(u*n/limit)+1):
		t = 0
		for j in range(1, k+1):
			t += (a+j/k*b)*g[j]*f[k-j]
			
		f.append(t/(1-a*g[0]))
		
	return f
	
def inverse(pdf, gf, u, limit=np.inf):
	if limit == np.inf or limit <= 0:
		return None
	
	n=2**15
	r = n
	
	while r < 32*u*n/limit:
		r *= 2
	
	g = discretize_pdf(pdf, limit, n)
	g.extend([0 for i in range(r-n)])
	
	temp = fft(g)
	temp = [gf(t) for t in temp]
	
	return ifft(temp)

class distr:
	def __init__(self, name="", limit=np.inf):
		self.name = name
		self.limit = limit
		
	def pdf(self, x):
		print("Not specified or doesn't exist")
		return False
		
	def cdf(self, x):
		print("Not specified or doesn't exist")
		return False
		
	def pmf(self, x):
		print("Not specified or doesn't exist")
		return False
		
	def mgf(self, x):
		print("Not specified or doesn't exist")
		return False

	def moment(self, n):
		if self.pdf(0):
			return round(integrate.quad(lambda x: (x**n)*self.pdf(x), 0, np.inf)[0], 5)
		elif self.cdf(0):
			return round(n*integrate.quad(lambda x: x**(n-1)*(1-self.cdf(x)), 0, np.inf)[0], 5)
		else:
			raise AttributeError("No method to calculate moments")
			
		
class discrete(distr):
	def __init__(self, name):
		super().__init__(name)

	def cdf(self, x):
		if (abs(x-round(x,0))<10**(-6)):
			x = int(round(x,0))
		else:
			x = int(np.floor(x))
		
		return sum([self.pmf(k) for k in range(x+1)])
		
	def gf(self, x):
		print("Not specified or doesn't exist")
		return False
		
	def moment(self, n):
		return round(sum([k**n*self.pmf(k) for k in range(5000)]), 3)
	
class expon(distr):
	def __init__(self, l, quote = 1, limit=np.inf):
		super().__init__("exponential", limit)
		if l > 0:
			self.l = l*quote
		else:
			raise AttributeError("Incorrect domain of l (l > 0)")
		
	def pdf(self, x):
		if x >= 0 and x <= self.limit:
			return np.exp(-x/self.l)/self.l
		else:
			return 0
	
	def cdf(self, x):
		if x>=0 and x <= self.limit:
			return 1-np.exp(-x/self.l)
		elif x > self.limit:
			return 1
		else:
			return 0
			
	def mgf(self, x):
		if self.limit == np.inf:
			if x >= 1/self.l:
				return np.inf
			else:
				return 1/(1-self.l*x)
		else:
			if x == 1/self.l:
				return self.limit/self.l + np.exp(self.limit*x)*(1-self.cdf(self.limit))
			else:
				return 1/(x*self.l-1)*(np.exp(self.limit*(x-1/self.l))-1) + np.exp(self.limit*x)*(1-self.cdf(self.limit))
			
	def moment(self, n):
		if self.limit == np.inf:
			if n == 1 or n == 2:
				return n*self.l**n
			elif n == 3:
				return 6*self.l**n
			else:
				super().moment(n)
		else:
			t = self.limit*(1-self.cdf(self.limit))
			if n == 1:
				return self.l-np.exp(-self.limit/self.l)*(self.limit+self.l)+t
			elif n == 2:
				return 2*self.l**2 - np.exp(-self.limit/self.l)*(2*self.l**2 + 2*self.l*self.limit+self.limit**2)+t
			elif n == 3:
				return 6*self.l**3 - np.exp(-self.limit/self.l)*(6*self.l**3 + 6*self.l**2*self.limit + 3*self.l*self.limit**2 +self.limit**3)+t
			else:
				super().moment(n)

class nb(discrete):
	# r - number of unsuccessful tries with probability 1-p
	def __init__(self, r, p):
		if r > 0:
			self.r = r
		else:
			raise AttributeError("Incorrect domain of r (r > 0)")
		if 0 < p and p < 1:
			self.p = p
		else:
			raise AttributeError("Incorrect domain of  (0 < p < 1)")
			
		super().__init__("negative binomial")
		self.a = p
		self.b = p*(r-1)
	
	def pmf(self, n):
		return comb(n+self.r-1, n)*(self.p**n)*(1-self.p)**self.r
		
	def gf(self, x):
		if abs(x)>1:
			return np.inf
		else:
			return ((1-self.p)/(1-self.p*x))**self.r

class poisson(discrete):
	def __init__(self, l):
		if l > 0:
			self.l = l
		else:
			raise AttributeError("Incorrect domain of l (l > 0)")
		super().__init__("poisson")
		self.a = 0
		self.b = l
		
	def pmf(self, n):
		ans = 1
		
		for i in range(1, n+1):
			ans *= self.l/i
			
		return ans*np.exp(-self.l)
		
	def gf(self, x):
		if abs(x) > 1:
			return np.inf
		else:
			return np.exp(self.l*(x-1))
		
class lognormal(distr):
	def __init__(self, mu, sigma, quote = 1):
		self.mu = mu +np.log(quote)
		if sigma > 0:
			self.sigma = sigma
		else:
			raise AttributeError("Incorrect domain of sigma (sigma > 0)")
		super().__init__("lognormal")
		
	def pdf(self, x):
		if x > 0:
			return norm_const/(x*self.sigma)*np.exp(-(np.log(x)-self.mu)**2/(2*self.sigma**2))
		else:
			return 0
			
	def mgf(self, x):
		return 0
			
	def moment(self, n):
		return np.exp(n*self.mu+(n*self.sigma)**2/2)
		
		
class pareto(distr):
	def __init__(self, xm, k, quote = 1, limit = np.inf):
		super().__init__("pareto", limit)
		self.xm = xm*quote

		if (k == 1) or (k == 2) or (k == 3):
			raise AttributeError("Parameter k should not be equal to 1,2,3")
		
		self.k = k
		
	def mgf(self, x):
		return 0
		
	def moment(self, n):
		return self.k*self.xm**n/(self.k-n)
		
class gamma(distr):
	def __init__(self, alpha, beta, quote=1, limit=np.inf):
		super().__init__("gamma", limit)
		self.alpha = alpha
		self.beta = beta/quote
		
	def cdf(self, x):
		return gammainc(self.alpha, x*self.beta)
		
	def pdf(self, x):
		return (self.beta**self.alpha)*x**(self.alpha-1)*np.exp(-self.beta*x)/gamma(self.alpha)
		
	def mgf(self, x):
		if (x < self.beta):
			return (1-x/self.beta)**(-self.alpha)
		else:
			return np.inf
			
	def moment(self, n):
		if (n == 1):
			return self.alpha/self.beta
		if (n == 2):
			return self.alpha*(self.alpha+1)/self.beta**2
		if (n == 3):
			return self.alpha*(self.alpha+1)*(self.alpha+2)/self.beta**3
			
