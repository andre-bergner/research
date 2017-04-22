#include "pop_back_iterator.h"
#include <iostream>
#include <vector>


int main()
{
   std::vector<int> xs = {1,2,3,4};
   auto it = iter::back_remover(xs);

   std::cout << *it << std::endl;
   std::cout << *++it << std::endl;
   std::cout << *++it << std::endl;
   std::cout << *++it << std::endl;
}