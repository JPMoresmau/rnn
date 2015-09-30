{-# LANGUAGE PatternGuards #-}
-----------------------------------------------------------------------------
--
-- Module      :  AI.Network.RNN.Expr
-- Copyright   :  (c) JP Moresmau
-- License     :  BSD3
--
-- Maintainer  :  JP Moresmau <jp@moresmau.fr>
-- Stability   :  experimental
-- Portability :
--
-- | Expression language to use for symbolic differentiation
--   <http://jtobin.ca/blog/2014/07/06/automasymbolic-differentiation/>
-----------------------------------------------------------------------------

module AI.Network.RNN.Expr (
    Expr(..)
  , eval
  , close
  , autoEval
  , fullSimplify
) where


import Text.PrettyPrint.HughesPJClass

import Numeric.AD as AD
import qualified Data.IntMap as I

-- | The expression type, pretty self explanatory
data Expr a =
    Lit a
  | Var Int
  | Neg (Expr a)
  | Add (Expr a) (Expr a)
  | Sub (Expr a) (Expr a)
  | Mul (Expr a) (Expr a)
  | Div (Expr a) (Expr a)
  | Recip (Expr a)
  | Exp (Expr a)
  | Sin (Expr a)
  | Cos (Expr a)
  | Tan (Expr a)
  | SinH (Expr a)
  | CosH (Expr a)
  | TanH (Expr a)
  | Log (Expr a)
  deriving (Eq, Show)

-- | (partial) Num instance
instance Num a => Num (Expr a) where
  fromInteger = Lit . fromInteger
  e0 + e1 = Add e0 e1
  e0 - e1 = Sub e0 e1
  e0 * e1 = Mul e0 e1
  negate e0 = Neg e0

-- | Fractional instance
instance Fractional a => Fractional (Expr a) where
  fromRational = Lit . fromRational
  e0 / e1 = Div e0 e1
  recip e0 = Recip e0

-- | (Partial) Floating instance
instance Floating a => Floating (Expr a) where
  exp e0 = Exp e0
  sin e0 = Sin e0
  cos e0 = Cos e0
  tan e0 = Tan e0
  sinh e0 = SinH e0
  cosh e0 = CosH e0
  tanh e0 = TanH e0
  log e0 = Log e0

-- | Pretty print as a numeric expression
instance Pretty a => Pretty (Expr a)
    where
        pPrint (Lit x) = pPrint x
        pPrint (Var x) = char 'x' <> int x
        pPrint (Neg e0)  = unaryPrint "negate" e0
        pPrint (Add e0 e1) = binaryPrint '+' e0 e1
        pPrint (Sub e0 e1) = binaryPrint '-' e0 e1
        pPrint (Mul e0 e1) = binaryPrint '*' e0 e1
        pPrint (Div e0 e1) = binaryPrint '/' e0 e1
        pPrint (Recip e0)  = unaryPrint "recip" e0
        pPrint (Exp e0)  = unaryPrint "exp" e0
        pPrint (Sin e0)  = unaryPrint "sin" e0
        pPrint (Cos e0)  = unaryPrint "cos" e0
        pPrint (Tan e0)  = unaryPrint "tan" e0
        pPrint (SinH e0)  = unaryPrint "sinh" e0
        pPrint (CosH e0)  = unaryPrint "cosh" e0
        pPrint (TanH e0)  = unaryPrint "tanh" e0
        pPrint (Log e0)  = unaryPrint "log" e0


-- | Helper to print binary expression
binaryPrint :: (Pretty a,Pretty b) => Char -> a -> b -> Doc
binaryPrint c e0 e1= parens $ pPrint e0 <+> char c <+> pPrint e1

-- | Helper to print unary expression
unaryPrint :: Pretty a => String -> a -> Doc
unaryPrint c e0 = parens $ text c <+> pPrint e0

-- | Simplify Expression
simplify :: (Floating a,Eq a)=>Expr a -> Expr a
simplify (Neg (Lit x)) = Lit (-x)
simplify (Neg e0) = Neg (simplify e0)
simplify (Add (Lit 0) e1) = simplify e1
simplify (Add e0 (Lit 0)) = simplify e0
simplify (Add (Lit x) (Lit y)) = Lit (x+y)
simplify (Add e0 e1) = Add (simplify e0) (simplify e1)
simplify (Sub e0 (Lit 0)) = simplify e0
simplify (Sub (Lit x) (Lit y)) = Lit (x-y)
simplify (Sub (Lit 0) e0) = Neg (simplify e0)
simplify (Sub e0 e1) | e0==e1 = Lit 0
simplify (Sub e0 e1) = Sub (simplify e0) (simplify e1)
simplify (Mul (Lit 1) e1) = simplify e1
simplify (Mul e0 (Lit 1)) = simplify e0
simplify (Mul (Lit (-1)) e1) = Neg (simplify e1)
simplify (Mul e0 (Lit (-1))) = Neg (simplify e0)
simplify (Mul (Lit 0) _) = Lit 0
simplify (Mul _ (Lit 0)) = Lit 0
simplify (Mul (Lit x) (Lit y)) = Lit (x*y)
simplify (Mul e0 e1) = Mul (simplify e0) (simplify e1)
simplify (Div e0 (Lit 1)) = simplify e0
simplify (Div e0 e1) | e0==e1 = Lit 1
simplify (Div (Lit x) (Lit y)) = Lit (x/y)
simplify (Div e0 e1) = Div (simplify e0) (simplify e1)
simplify (Recip (Lit x)) = Lit (recip x)
simplify (Recip e0) = Recip (simplify e0)
simplify (Exp (Lit x)) = Lit (exp x)
simplify (Exp e0) = Exp (simplify e0)
simplify (Sin (Lit x)) = Lit (sin x)
simplify (Sin e0) = Sin (simplify e0)
simplify (Cos (Lit x)) = Lit (cos x)
simplify (Cos e0) = Cos (simplify e0)
simplify (Tan (Lit x)) = Lit (tan x)
simplify (Tan e0) = Tan (simplify e0)
simplify (SinH (Lit x)) = Lit (sinh x)
simplify (SinH e0) = SinH (simplify e0)
simplify (CosH (Lit x)) = Lit (cosh x)
simplify (CosH e0) = CosH (simplify e0)
simplify (TanH (Lit x)) = Lit (tanh x)
simplify (TanH e0) = TanH (simplify e0)
simplify (Log (Lit x)) = Lit (log x)
simplify (Log e0) = Log (simplify e0)
simplify e0 = e0

-- | Simplify Expression as much as possible
fullSimplify :: (Floating a,Eq a, Show a)=>Expr a -> Expr a
fullSimplify expr = fullSimplify' expr (Lit 0) -- placeholder
  where fullSimplify' cur lastE | cur == lastE = cur
                                | otherwise = let cur' = simplify cur
                                             in fullSimplify' cur' cur

-- | Close an expression over some variable.
close :: Expr a -> I.IntMap a -> Expr a
close e0 m | I.null m = e0
close (Neg e0) m = Neg (close e0 m)
close (Add e0 e1) m = Add (close e0 m) (close e1 m)
close (Sub e0 e1) m = Sub (close e0 m) (close e1 m)
close (Mul e0 e1) m = Mul (close e0 m) (close e1 m)
close (Div e0 e1) m = Div (close e0 m) (close e1 m)
close (Recip e0) m = Recip (close e0 m)
close (Exp e0) m = Exp (close e0 m)
close (Sin e0) m = Sin (close e0 m)
close (Cos e0) m = Cos (close e0 m)
close (Tan e0) m = Tan (close e0 m)
close (SinH e0) m = SinH (close e0 m)
close (CosH e0) m = CosH (close e0 m)
close (TanH e0) m = TanH (close e0 m)
close (Log e0) m = Log (close e0 m)
close (Var v) m
  | Just x <- I.lookup v m   = Lit x
  | otherwise = Var v
close e0 _ = e0 -- Lit

-- | Evaluate a closed expression.
eval :: (Num a,Floating a) => Expr a -> a
eval (Lit d) = d
eval (Var _) = error "expression not closed"
eval (Neg e0) = -(eval e0)
eval (Add e0 e1) = eval e0 + eval e1
eval (Sub e0 e1) = eval e0 - eval e1
eval (Mul e0 e1) = eval e0 * eval e1
eval (Div e0 e1) = eval e0 / eval e1
eval (Recip e0) = recip $ eval e0
eval (Exp e0) = exp $ eval e0
eval (Sin e0) = sin $ eval e0
eval (Cos e0) = cos $ eval e0
eval (Tan e0) = tan $ eval e0
eval (SinH e0) = sinh $ eval e0
eval (CosH e0) = cosh $ eval e0
eval (TanH e0) = tanh $ eval e0
eval (Log e0) = log $ eval e0

-- | Evaluate for AD
autoEval :: (AD.Mode a,Floating a) =>  Expr (Scalar a) -> I.IntMap a -> a
autoEval expr m = go expr where
  go (Lit d) = auto d
  go (Var s)
    | Just x <- I.lookup s m    = x
    | otherwise = error "expression not closed"
  go (Neg e0) = - (go e0)
  go (Add e0 e1) = go e0 + go e1
  go (Sub e0 e1) = go e0 - go e1
  go (Mul e0 e1) = go e0 * go e1
  go (Div e0 e1) = go e0 / go e1
  go (Recip e0)  = recip $ go e0
  go (Exp e0)    = exp $ go e0
  go (Sin e0)    = sin $ go e0
  go (Cos e0)    = cos $ go e0
  go (Tan e0)    = tan $ go e0
  go (SinH e0)    = sinh $ go e0
  go (CosH e0)    = cosh $ go e0
  go (TanH e0)    = tanh $ go e0
  go (Log e0)    = log $ go e0

