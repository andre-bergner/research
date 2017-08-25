from pylab import *


def  phase_sync ( p1 , p2 , a = 0.002 ):
   pc = zeros( p1.shape )
   for n in range( 0 , len(p1) ):
      pc[n]  =  pc[n-1] + a*(mod( p2[n] - p1[n] - 2.*pc[n-1] + pi , 2.*pi ) - pi)
   return pc

# instantanious freq
# inst_freq = imag(0.5*(x1[2:]-x1[:-2])/x1[1:-1])
# recontruct
# p1 = zeros(inst_freq.shape)
# p1[0::2] = cumsum(inst_freq[0::2])
# p1[1::2] = cumsum(inst_freq[1::2])

N = 2**16
noise1 = randn(N)
x1 = ifft(fft(noise1) * exp(-0.01*(arange(N)-40)**2))
noise2 = randn(N)
x2 = ifft(fft(noise2) * exp(-0.01*(arange(N)-40)**2))

p1 = angle(x1)
p2 = angle(x2)

pc  = phase_sync( p1 , p2 , 0.0005)
p1_ = mod ( p1 + pc + pi , 2.*pi ) - pi
p2_ = mod ( p2 - pc + pi , 2.*pi ) - pi

diff      = mod( p2_- p1_+ pi , 2.*pi ) - pi
diff_orig = mod( p2 - p1 + pi , 2.*pi ) - pi


figure(1)

#plot( p1  , 'b' , label=r"$\phi_1$" )
#plot( p2  , 'g' , label=r"$\phi_2$" )
plot( p1_ , 'r' , label=r"$\psi_1$" )
plot( p2_ , 'y' , label=r"$\psi_2$" )
plot( diff , 'k' , linewidth=2 )
plot( diff_orig , 'b' , linewidth=2 )
legend()


x1_ = abs(x1)*cos(p1_)
x2_ = abs(x2)*cos(p2_)


figure(2)
plot( real(x1_) , 'k' )
plot( real(x2_) , 'r' )
#plot( abs(x1)   , 'b' )

figure(3)
plot( real(x1+x2) , 'k' )
plot( real(x1_+x2_) , 'r' )


show()

