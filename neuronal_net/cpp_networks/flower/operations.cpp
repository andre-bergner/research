#include "operations.hpp"
#include <iostream>


struct Add
{
   static constexpr size_t num_inputs = 2;
   static constexpr size_t num_outputs = 1;
};

int main()
{
   using AddPar = Parallel<Add,Add>;
   using AddSeq = Sequence<AddPar,Add>;
   AddSeq s;
   std::cout << num_inputs_v<AddPar> << std::endl;
   /*
   auto a1 = Var(0.8);
   auto b1 = Var(-0.2);
   auto a2 = Var(0.3);
   auto b2 = Var(0.1);

   auto l1 = a1 * _1 + b1;   // expression with one placeholder
   auto l2 = a2 * _1 + b2;   // expression with one placeholder
   */
   //  x*x + sin(x)
   //         x
   //  _1*_1    sin(_1)
   //     _1 + _2
}