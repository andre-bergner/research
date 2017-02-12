#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
np.set_printoptions(linewidth=140)


N = 128
k =  0.2
damp = -0.1

#N = 256
#k =  0.1
#damp = -0.05

#N = 512
#k =  0.05
#damp = -0.025



f = arange( N ) * 2*pi/N   
a = exp( 1j*f + damp )
e = exp( -1j*f )
b = ones(N)

I = eye(len(a))
A = diag(a)
B = b / N
#B = 1 - a*e

K = np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1)
K[-1,0] = 1
K[0,-1] = 1

M = dot( A , I + k*K )


def H ( A , B , z ):    # state space transfer function
   return  linalg.solve( z * I  +  M  , B )


i = 1.j
W = linspace(-pi,pi,2000)

Hz = array( map( lambda w: H(A,B,exp(i*w)) , W ) )

S = linalg.solve( I , M )  # ??? S=M
print sort( abs(eig(S)[0]) )

Hs = np.sum( Hz[:,:] , axis = 1 )
#Hs = np.sum( exp(-24j*f) * Hz[:,:] , axis = 1 )


figure()
plot( W , abs(Hz)[:,:] ,'k' , alpha = 0.6 , linewidth = 0.5 )
plot( W , abs(Hz[:,N/2]) , 'k' , linewidth = 1.5 )
xticks( [-pi,0,pi] , [r"$-\pi$",0,r"$\pi$"] )
xlim(( -pi , pi ))
ylim((  0 , 12 ))

figure()
plot( W , abs(Hs) , 'k' )
xticks( [-pi,0,pi] , [r"$-\pi$",r"$0$",r"$\pi$"] , fontsize=24 )
yticks( [0,1] , [r"$0$",r"$1$"] , fontsize=24 )
xlim(( -pi , pi ))
ylim((  0 , 1.1 ))

plot( (-0.2 , -1.5) , (1,.6) , color="#333333" , linewidth=0.5 )
plot( ( 0.2 ,  1.5) , (1,.6) , color="#333333" , linewidth=0.5 )

def relative_inset( x , y , dx , dy , parent_ax ):
   pac = copy( parent_ax.get_position() )
   xl = xlim()    # TODO get from parent_ax
   yl = ylim()    # TODO get from parent_ax
   print xl
   print yl
   ax = pac[0][0] + (pac[1][0]-pac[0][0])*(x+pi) / (xl[1]-xl[0])
   ay = pac[0][1] + (pac[1][1]-pac[0][1])*(y-dy-0) / (yl[1]-yl[0])
   aw = (pac[1][0]-pac[0][0])*dx / (xl[1]-xl[0])
   ah = (pac[1][1]-pac[0][1])*dy / (yl[1]-yl[0])
   inset_ax = axes([ ax , ay , aw, ah ])
   return  inset_ax

ia = relative_inset( -1.5 , 0.6 , 3.0 , 0.3 , gca() )
plot( W , abs(Hs) , 'k' )
xlim(( -0.2 , 0.2 ))
xticks( [-0.2,0,0.2] , [r"$-0.2$",r"$0$","$0.2$"] , fontsize=14 )
yticks( [0.99999999,1,1.00000001] , [r"$1-10^{-8}$",r"$1$","$1+10^{-8}$"] , fontsize=14 )
gca().yaxis.set_ticks_position('right');



show();

