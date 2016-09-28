#include <iostream>
#include "matrix.h"

int main()
{
   using namespace std;

   matrix<float>  m( 2, 3, 2.f );
   m(0,0) = 0.f;
   m(1,0) = 1.f;

   cout << m(0,0) << " " << m(0,1) << endl;
   cout << m(1,0) << " " << m(1,1) << endl;
   cout << m(2,0) << " " << m(2,1) << endl;

   std::vector<float> v({ 1.f, 2.f });
   std::vector<float> u(3);

   dot( m, v, u );

   for ( auto x : u ) std::cout << x << ", "; std::cout << std::endl;

   matrix<float>  w(2,3);
   outer(u,v,w);

   cout << w(0,0) << " " << w(0,1) << " " << w(0,2) << endl;
   cout << w(1,0) << " " << w(1,1) << " " << w(1,2) << endl;

}
