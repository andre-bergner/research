#pragma once
#include <cstddef>

//    ------------------------------------------------------------------------------------
//   Traits
//    ------------------------------------------------------------------------------------


template <typename Box>
struct num_inputs
{
   static constexpr size_t value = Box::num_inputs;
};

template <typename Box>
inline constexpr size_t num_inputs_v = num_inputs<Box>::value;



template <typename Box>
struct num_outputs
{
   static constexpr size_t value = Box::num_outputs;
};

template <typename Box>
inline constexpr size_t num_outputs_v = num_outputs<Box>::value;


//    ------------------------------------------------------------------------------------
//   Combiners
//    ------------------------------------------------------------------------------------

template <typename Box1, typename Box2>
struct Parallel
{
   static constexpr size_t num_inputs = num_inputs_v<Box1> + num_inputs_v<Box2>;
   static constexpr size_t num_outputs = num_outputs_v<Box1> + num_outputs_v<Box2>;
};


template <typename Box1, typename Box2>
struct Sequence
{
   static_assert(num_outputs_v<Box1> == num_inputs_v<Box2>);
   static constexpr size_t num_inputs = num_inputs_v<Box1>;
   static constexpr size_t num_outputs = num_outputs_v<Box2>;
};


template <typename Box1, typename Box2>
struct Split
{
   static_assert(num_outputs_v<Box1> == num_inputs_v<Box2>);
   static constexpr size_t num_inputs = num_inputs_v<Box1>;
   static constexpr size_t num_outputs = num_outputs_v<Box2>;
};


template <size_t N_wires>
struct Identity
{
   static constexpr size_t num_inputs = N_wires;
   static constexpr size_t num_outputs = N_wires;
};







