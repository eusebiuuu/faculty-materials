{-# LANGUAGE ConstrainedClassMethods #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}

module Unification (unify, unifyAsList) where

import Syntax ( Expr(..), Equation(..) )
import Data.List (nub, intercalate)

import Parser (Parser, parseFirst, pequations, pexprs, pequation, symbol, pexpr, commaSep0, variable)

{-| Class for types for which the set of variables can be computed.
-}
class HasVars e where
  vars :: e -> [String]

{-| Extends a 'HasVars' instance to lists
-}
instance HasVars e => HasVars [e] where
  vars = nub . concatMap vars

instance HasVars Expr where
  vars (Var x) = [x]
  vars (App _ es) = vars es

instance HasVars Equation where
  vars (e1 :=: e2) = vars [e1, e2]

-- | Substitution as a function from names of variables to expressions
newtype Substitution = Substitution { vapply :: String -> Expr }

-- | Substitution as a list of pairs between names of variables and expressions
newtype SubstitutionGraph = SubstitutionGraph [(String, Expr)]

instance Show SubstitutionGraph where
  show (SubstitutionGraph sg) = intercalate ", " (map (\(x, e) -> x ++ " |-> " ++ show e) sg)

-- | Given a list of variables of interest computes the 'SubstitutionGraph' associated to
-- them from a given 'Substitution'
makeGraph :: [String] -> Substitution -> SubstitutionGraph
makeGraph vs subst = SubstitutionGraph (filter (\(x, e) -> e /= Var x) substPairs)
  where
    substPairs = vs `zip` map (vapply subst) vs


{-| Class for operation which substitution-like structures should support.
We will make our 'unifyStep' algorithm below generic in the choice of substitution type.

We assume that our substitutions are always in solved form, that is, for any substitution s
if s x =/= x then x does not appear in s y for any variable y
-}
class SubstitutionLike s where
  -- | the identity substitution
  idSubst :: s
  -- | the singleton substitution. default implementation uses 'extend' on 'idSubst'
  singletonSubst :: String -> Expr -> s
  singletonSubst = extend idSubst
  -- | Composition / combination of substitutions
  -- (s1 >>> s2) x means that first we apply s1 on x then s2 on the result
  (>>>) :: s -> s -> s
  -- | Obtaining an actual 'Substitution' from self
  asSubstitution :: s -> Substitution

  -- | Substitution application. Defaults to applying the corresponding 'Substitution'
  apply :: s -> Expr -> Expr
  apply = apply . asSubstitution
  -- | Substitution application to an equation. Implemented based on 'apply'
  eapply :: s -> Equation -> Equation
  eapply s (e1 :=: e2) = apply s e1 :=: apply s e2
  -- | Extending a substitution with x |-> e. Should substitute x with e in all
  -- expessions assigned to variables and add the map x |-> e to the substitution.
  -- see corresponding step in the unification algorithm.
  -- Defaults to composing the substitution with the singleton substitution x |-> e
  extend :: s -> String -> Expr -> s
  extend s x e = s >>> singletonSubst x e


{-| A 'SubstitutionLike' instance for 'SubstitutionGraph'

Examples: 

>>> idSubst :: SubstitutionGraph

>>> singletonSubst "x" (Var "y") :: SubstitutionGraph
x |-> y

>>> extend (SubstitutionGraph [("x", Var "y")]) "y" (App "a" [])
y |-> a, x |-> a

>>> (SubstitutionGraph [("x", Var "y")]) >>> singletonSubst "y" (App "a" [])
y |-> a, x |-> a
-}

instance SubstitutionLike SubstitutionGraph where
  idSubst :: SubstitutionGraph
  idSubst = SubstitutionGraph []
  (>>>) :: SubstitutionGraph -> SubstitutionGraph -> SubstitutionGraph
  s1 >>> SubstitutionGraph s2 = foldr (\(x, e) sub -> extend sub x e) s1 s2
  asSubstitution :: SubstitutionGraph -> Substitution
  asSubstitution (SubstitutionGraph s) = foldr (\(x, e) sub -> extend sub x e) idSubst s
  -- Because we implemented '(>>>)' using extend, we cannot use here the default implementation of extend
  extend :: SubstitutionGraph -> String -> Expr -> SubstitutionGraph
  extend (SubstitutionGraph s) x e = SubstitutionGraph ((x, e): map (\(x', e') -> (x', apply xe e')) s)
    where
      xe :: SubstitutionGraph
      xe = singletonSubst x e

instance SubstitutionLike Substitution where
  idSubst :: Substitution
  idSubst = Substitution Var

  singletonSubst :: String -> Expr -> Substitution
  singletonSubst x e = Substitution subst
    where
      subst y
          | y == x = e
          | otherwise = Var y

  (>>>) :: Substitution -> Substitution -> Substitution
  s1 >>> s2 = Substitution (apply s2 . vapply s1)

  asSubstitution :: Substitution -> Substitution
  asSubstitution = id

  -- Because the default implementation uses 'toSubstitution' we cannot use it here.
  apply :: Substitution -> Expr -> Expr
  apply subst = go
    where
      go (Var x) = vapply subst x
      go (App f es) = App f (go <$> es)

{-| A configuration for the unification algorithm.

It consists of the list of equations (still) to be reasolved
and the current computed substitution.

The substitution is parameterized to allow any substitution-like type.
-}
data Config s = Config [Equation] s

instance Show s => Show (Config s) where
  show (Config eqs s) = "< " ++ show eqs ++ ", " ++ show s ++ " >"

{-| Implements a step of the unification algorithm.

If successful, it yields the new configuration.
If failing it returns an appropriate error message.

Examples:

>>> testUnifyStep "g(Y) = X, f(X,h(X),Y) = f(g(Z),W,Z)" ""
Right < [f(g(Y), h(g(Y)), Y) = f(g(Z), W, Z)], X |-> g(Y) >

>>> testUnifyStep "f(g(Y), h(g(Y)), Y) = f(g(Z), W, Z)" "X |-> g(Y)"
Right < [g(Y) = g(Z),h(g(Y)) = W,Y = Z], X |-> g(Y) >

>>> testUnifyStep "g(Y) = g(Z),h(g(Y)) = W,Y = Z" "X |-> g(Y)"
Right < [Y = Z,h(g(Y)) = W,Y = Z], X |-> g(Y) >

>>> testUnifyStep "Y = Z,h(g(Y)) = W,Y = Z" "X |-> g(Y)"
Right < [h(g(Z)) = W,Z = Z], Y |-> Z, X |-> g(Z) >

>>> testUnifyStep "h(g(Z)) = W,Z = Z" "Y |-> Z, X |-> g(Z)"
Right < [Z = Z], W |-> h(g(Z)), Y |-> Z, X |-> g(Z) >

>>> testUnifyStep "Z = Z" "Y |-> Z, X |-> g(Z)"
Right < [], Y |-> Z, X |-> g(Z) >


-}
unifyStep :: SubstitutionLike s => Config s -> Either String (Config s)
unifyStep (Config (Var x :=: Var y : eqs) subst)
  | x == y = Right $ Config eqs subst

unifyStep (Config (Var x :=: t : eqs) subst)
  | x `elem` vars t = Left $ "Cycle on" x
  | otherwise = Right $ Config (map (eapply (singletonSubst x t :: SubstitutionGraph)) eqs) (extend subst x t)

unifyStep (Config (t :=: Var x : eqs) subst)
  | x `elem` vars t = Left $ "Cycle on" ++ x
  | otherwise = Right $ Config (map (eapply (singletonSubst x t :: SubstitutionGraph)) eqs) (extend subst x t)

unifyStep (Config ((App f args_f) :=: (App g args_g) : eqs) subst)
  | f =/= g = Left $ "Conflict on" ++ f
  | length args_f =/= length args_g = Left $ "Conflict on" ++ args_f ++ args_g
  | otherwise = Right $ Config (zipWith (:=:) args_f args_g ++ eqs) subst

class HasUnify eqs where
  unify :: eqs -> Either String Substitution

  unifyAsList :: HasVars eqs => eqs -> Either String SubstitutionGraph
  unifyAsList eqs = makeGraph (vars eqs) <$> unify eqs

{-| HasUnify instance for lists of equations uses 'unifyStep' to
implement the actual unification algorithm described in the class.

Examples:

>>> testUnifyEqs "g(Y) = X, f(X,h(X),Y) = f(g(Z),W,Z)"
Right Y |-> Z, X |-> g(Z), W |-> h(g(Z))

>>> testUnifyEqs "g(Y) = X, f(X,h(Y),Y) =  f(g(Z),b,Z)"
Left "Conflict between h/1 and b/0"

>>> testUnifyEqs "g(Y) = X, f(X,h(X),Y) = f(Y,W,Z)"
Left "Cycle on Y"
-}
instance HasUnify [Equation] where
  unify eqs = go (Config eqs idSubst)
    where
      go (Config [] s) = Right s
      go cfg = unifyStep cfg >>= go

{-| Special instance of the unification algorithm for a single equation.

Examples:

>>> testUnifyEq "plus(X,mul(Y,Y)) = plus(X,mul(Y,X))"
Right Y |-> X
-}
instance HasUnify Equation where
  unify eq = unify [eq]

{-| Special instance of the unification algorithm for lists of expressions.
We transform them in equations each with the next in sequence.

Examples:

>>> testUnifyExprs "f(X,Y), f(h(X),X), f(X,b)"
Left "Cycle on X"

>>> testUnifyExprs "f(X,f(X,g(Y))), f(U,Z), f(g(Y), Y)"
Left "Cycle on Y"

>>> testUnifyExprs "f(f(X,Y),X), f(g(Y), Z), f(U,h(Z))"
Left "Conflict between f/2 and g/1"

>>> testUnifyExprs "f(f(X,Y),X), f(V,U), f(U,h(Z))"
Left "Cycle on U"

>>> testUnifyExprs "f(f(X,Y),X), f(V,U), f(U,Z)"
Left "Cycle on U"

>>> testUnifyExprs "f(f(g(X),h(Y)),h(Z)), f(f(U,h(h(X))),h(Y)), f(V,W)"
Right Y |-> h(X), Z |-> h(X), U |-> g(X), V |-> f(g(X), h(h(X))), W |-> h(h(X))

>>> testUnifyExprs "p(X, X,Z), p(f(a,a), Y,Y), p(f(X,a), b,Z)"
Left "Conflict between a/0 and f/2"

>>> testUnifyExprs "p(X,X,Z), p(f(a,a), Y,Y), p(X,b,Z)"
Left "Conflict between f/2 and b/0"

>>> testUnifyExprs "p(X,X,Z), p(f(a,a), Y,Y), p(X,f(a,a), Z)"
Right X |-> f(a, a), Z |-> f(a, a), Y |-> f(a, a)

>>> testUnifyExprs "p(f(X,a), g(Y), Z), p(f(a,a), Z,U), p(V,U,Z)"
Right X |-> a, Z |-> g(Y), U |-> g(Y), V |-> f(a, a)
-}
instance HasUnify [Expr] where
  unify es = unify (zipWith (:=:) es (tail es))

-- Extra stuff for testing

testUnify :: (HasUnify e, HasVars e, Show e) => Parser e -> String -> Either String SubstitutionGraph
testUnify p s = parseFirst p s >>= unifyAsList 
    
testUnifyEqs :: String -> Either String SubstitutionGraph
testUnifyEqs = testUnify pequations

testUnifyExprs :: String -> Either String SubstitutionGraph
testUnifyExprs = testUnify pexprs

testUnifyEq :: String -> Either String SubstitutionGraph
testUnifyEq = testUnify pequation

testUnifyStep :: String -> String -> Either String (Config SubstitutionGraph)
testUnifyStep eqss substs 
  = do
    eqs <- parseFirst pequations eqss
    subst <- parseFirst pSubstitutionGraph substs
    unifyStep (Config eqs subst)

{-| parser for  a singleton substitution pair

Examples:

>>> parseFirst psingletonSubst "X |-> f(a, T)"
Right ("X",f(a, T))
-}
psingletonSubst :: Parser (String, Expr)
psingletonSubst = (,) <$> variable <*> (symbol "|->" *> pexpr)

{-| parser for a substitution represented as a 'SubstitutionGraph'

Examples:
>>> parseFirst pSubstitutionGraph "X |-> a, Z |-> g(Y), U |-> g(Y), V |-> f(a, a)"
Right X |-> a, Z |-> g(Y), U |-> g(Y), V |-> f(a, a)
-}
pSubstitutionGraph :: Parser SubstitutionGraph
pSubstitutionGraph = SubstitutionGraph <$> commaSep0 psingletonSubst
