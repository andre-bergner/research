from pylab import *


#   ----------------------------------------------------------------
#  continous space, continous time
#   ----------------------------------------------------------------

def alpha_csct( t , gamma = 1 , kappa = 0.09 ):

   return exp( gamma*t - 0.5*kappa*t*t )



#   ----------------------------------------------------------------
#  discrete space, continous time
#   ----------------------------------------------------------------

def alpha_dsct( t , gamma = 1 , kappa = 0.09 , w = 0.1 ):      # laplacian formulation

   return exp( (gamma-2*kappa)*t + 2*kappa*sin(w*t) )


def alpha_dsct_adj( t , gamma = 1 , kappa = 0.09 , w = 0.1 ):      # adjecency formulation

   return exp( gamma*t + 2*kappa*sin(w*t) )



#   ----------------------------------------------------------------
#  discrete space, discrete time
#   ----------------------------------------------------------------

def alpha_dsdt_adj( t , gamma = -0.2 , kappa1 = 0.2 , kappa2 = 0.2 , w = 1 ):

   return  cumprod( exp(gamma) * ( 1 + 2*kappa1*cos(w*t)) / ( 1 + 2*kappa2*cos(w*t)) )



def alpha_dsdt_euler( t , gamma = -0.2 , kappa = 0.2 , w = 1 ):

   return  alpha_dsdt_adj( t , gamma-2.*kappa , kappa , 0 , w )



def alpha_dsdt_tustin( t , gamma = -0.2 , kappa = 0.2 , w = 1 ):

   return  alpha_dsdt_adj( t , gamma-2.*kappa , kappa , -kappa , w )



def alpha_dsdt_adj_euler( t , gamma = -0.2 , kappa = 0.2 , w = 1 ):

   return  alpha_dsdt_adj( t , gamma , kappa , 0 , w )



def alpha_dsdt_adj_tustin( t , gamma = -0.2 , kappa = 0.2 , w = 1 ):

   return  alpha_dsdt_adj( t , gamma , kappa , -kappa , w )



t = arange(128)



figure()

plot( alpha_csct( t, 1, 0.09 ) , 'k' )
plot( alpha_dsct( t, 1, 0.6 ) , 'k' )

show()

"""
figure()

subplot(211)
plot( array([ alpha_dsdt_adj_euler( t , 3*pi/N , 3*pi/N ,  2*pi/N ) for N in arange(30,3000,20) ]).T , 'k', alpha=0.7 )

subplot(212)
plot( array([ alpha_dsdt_euler( t , 3*pi/N , 3*pi/N ,  3*pi/N ) for N in arange(30,3000,20) ]).T , 'k', alpha=0.7 )



figure()

subplot(211)
plot( array([ alpha_dsdt_adj_tustin( t , 3*pi/N , 3*pi/N ,  2*pi/N ) for N in arange(30,3000,20) ]).T , 'k', alpha=0.7 )

subplot(212)
plot( array([ alpha_dsdt_tustin( t , 3*pi/N , 3*pi/N ,  3*pi/N ) for N in arange(30,3000,20) ]).T , 'k', alpha=0.7 )
"""



"""
exp( k*( < + > ) ) = 1 + k*( < + > ) + k^2/2 * ( < + > )^2 + k^3/3! * ( < + > )^3 + ...

= 1 
+  k      * ( < + > ) 
+  k^2/2! * ( << + 2 + >> )
+  k^3/3! * ( <<< + 3< + 3> + >>> )
+  k^4/4! * ( <<<< + 4<< + 6 + 4>> + >>>> )
+  k^5/5! * ( ... + 10< + 10> + ... )
+  k^6/6! * ( ... + 20 + ... )

1 + k^2 + k^4/4. + k^6/36. + ... +  2*(<+>)*( k + k^3/2. + k^5/12.)


def alpha_dsdt( t , gamma , k , w ): return  exp(gamma*t) * cumprod(  1 + k*k - k**4/4. + (k+k**3/2.)*2*cos(w*t) - 2*k*k*cos(2*w*t) )
plot( array([ alpha_dsdt( t , -3*pi/N , 3*pi/N , 2*pi/N ) for N in arange(20,800,20) ]).T , 'k', alpha=0.7 )

"""

