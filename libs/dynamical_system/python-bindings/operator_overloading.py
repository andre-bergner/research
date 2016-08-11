from numbers import Number


class Operation ( object ):
   def __repr__( self ):   return NotImplemented

class Addition ( Operation ):
   def __repr__( self ):   return '+'

class Subtraction ( Operation ):
   def __repr__( self ):   return '-'

class Multiplication ( Operation ):
   def __repr__( self ):   return '*'

class Division ( Operation ):
   def __repr__( self ):   return '/'

class Negation ( Operation ):
   def __repr__( self ):   return '-'



class Expression ( object ):

   def eval( self ):
      return self.__repr__()

   def eval( self ):
      return self.__repr__()

   def __add__( self , rhs ):
      return BinaryExpression( self , Addition , rhs )

   def __radd__( self , lhs ):
      return BinaryExpression( lhs , Addition , self )

   def __sub__( self , rhs ):
      return BinaryExpression( self , Subtraction , rhs )

   def __rsub__( self , lhs ):
      return BinaryExpression( lhs , Subtraction , self )

   def __mul__( self , rhs ):
      return BinaryExpression( self , Multiplication , rhs )

   def __rmul__( self , lhs ):
      return BinaryExpression( lhs , Multiplication , self )

   def __div__( self , rhs ):
      return BinaryExpression( self , Division , rhs )

   def __rdiv__( self , lhs ):
      return BinaryExpression( lhs , Division , self )

   def __neg__( self ):
      return UnaryExpression( Negation , self )

   @staticmethod
   def __check_type( obj ):
      assert isinstance( obj , Number ) or      \
             isinstance( obj , Expression ) ,   \
             "'" + type(obj).__name__ +  "' is not a supported type!"



class UnaryExpression ( Expression ):

   def __init__( self , op , rhs ):
      self.__check_type( rhs )
      self.__op  = op
      self.__rhs = rhs

   def __repr__( self ):
      return  repr( self.__op() )  +  repr( self.__rhs )

   @staticmethod
   def __check_type( obj ):
      assert isinstance( obj , Number ) or      \
             isinstance( obj , Expression ) ,   \
             "'" + type(obj).__name__ +  "' is not a supported type!"



class BinaryExpression ( Expression ):

   def __init__( self , lhs , op , rhs ):
      self.__check_type( lhs )
      self.__check_type( rhs )
      self.__lhs = lhs
      self.__op  = op
      self.__rhs = rhs


   def __repr__( self ):

      open  = ['','(']  # opening parenthesis
      close = ['',')']  # closing parenthesis

      distributive = self.__op is Multiplication or self.__op is Division
      dl = distributive and isinstance( self.__lhs , Expression )
      dr = distributive and isinstance( self.__rhs , Expression )

      return  open[dl]   + repr( self.__lhs )  + close[dl]   \
                 + ' '   + repr( self.__op() ) + ' '         \
              + open[dr] + repr( self.__rhs )  + close[dr]

   @staticmethod
   def __check_type( obj ):
      assert isinstance( obj , Number ) or      \
             isinstance( obj , Expression ) ,   \
             "'" + type(obj).__name__ +  "' is not a supported type!"



class State ( Expression ):

   def __init__( self , index ):
      assert  isinstance( index , int ) and index >= 0 ,\
              "State must be initilized with an positive integer"
      self.__index = index

   def __repr__( self ):
      return "x[" + str(self.__index) + "]"



__state_count = 0

def states( num ):

   global __state_count

   _states = []

   for n in range( __state_count , __state_count + num ):
      _states.append( State(n) )

   __state_count += num
   
   return _states



class Parameter ( Expression ):

   def __init__( self , index ):
      assert  isinstance( index , int ) and index >= 0 ,\
              "State must be initilized with an positive integer"
      self.__index = index

   def __repr__( self ):
      return "c[" + str(self.__index) + "]"

