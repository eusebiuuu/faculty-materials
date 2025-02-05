

data D a b c = T a b c c
instance Applicative (D a b) where
    pure a = D a
    (<*>) (T x y z t) f = T x y (f z t)

