{-# LANGUAGE MultiParamTypeClasses, AllowAmbiguousTypes, FlexibleContexts, FlexibleInstances #-}


data Tree = Empty | Node Int Tree Tree Tree

extree :: Tree
extree = Node 4 (Node 5 Empty Empty Empty) (Node 3 Empty Empty (Node 1 Empty Empty Empty)) Empty

class ArbInfo t where
  level :: t -> Int -- intoarce inaltimea arborelui; pt un arbore vid
                      -- se considera ca are inaltimea 0
  sumval :: t -> Int -- intoarce suma valorilor din arbore
  nrFrunze :: t -> Int -- intoarce nr de frunze al arborelui
-- level extree
-- 3
-- sumval extree
-- 13
-- nrFrunze extree
-- 2


instance ArbInfo Tree where
    level Empty = 0
    level (Node k left mid right) = maximum[level left, level mid, level right] + 1
    sumval Empty = 0
    sumval (Node k left mid right) = k + sumval left + sumval right + sumval mid
    nrFrunze (Node k left mid right)
        | level(Node k left mid right) == 1 = 1
        | otherwise = nrFrunze left + nrFrunze mid + nrFrunze right


class Scalar a where
  zero :: a
  one :: a
  adds :: a -> a -> a
  mult :: a -> a -> a
  negates :: a -> a
  recips :: a -> a


class (Scalar a) => Vector v a where
    zerov :: v a
    onev :: v a
    addv :: v a -> v a -> v a -- adunare vector
    smult :: a -> v a -> v a  -- inmultire cu scalare
    negatev :: v a -> v a -- negare vector


data Vec2D a = Vec2D a a

instance (Scalar a) => Vector Vec2D a where
    zerov = Vec2D zero zero
    onev = Vec2D one one
    addv (Vec2D x1 y1) (Vec2D x2 y2) = Vec2D (adds x1 x2) (adds y1 y2)
    smult k (Vec2D x1 y1) = Vec2D (mult k x1) (mult k y1)
    negatev (Vec2D x1 y1) = Vec2D (negates x1) (negates y1)


disjElim :: Either a b -> (a -> c) -> (b -> c) -> c
disjElim (Right b) f g = g b
disjElim (Left a) f g = f a