data Dogs a = Huski a | Mastiff a

power :: Maybe Int -> Int -> Int
power Nothing n = 2 ^ n
power (Just m) n = m ^ n


my_elem :: Eq a => a -> [a] -> Bool
my_elem x ys = or [ x == y | y <- ys ]


data List a = Nil | a ::: List a

showMyList :: Show a => List a -> String
showMyList Nil = " N i l "
showMyList (x ::: xs) = show x ++ " : : : " ++ showMyList xs

list = (1 ::: (3 ::: Nil))

instance (Show a ) => Show ( List a ) where
    show = showMyList



data Point = Pt Int Int


instance Eq Point where
    (Pt x1 y1) == (Pt x2 y2) = x1 == x2 && y1 == y2

-- data D a b c = D a b c c
-- instance Functor (D a b) where
--     fmap f (D x y z t) = D x y (f z) (f t)

data D a b = D1 a | D2 b
instance Functor (D a) where
    fmap f (D1 a) = D1 a
    fmap f (D2 b) = D2 (f b)

(<+) :: String -> [Int] -> Bool
(<+) str list = True

data Tree a b = Empty | Branch a ( Tree a b ) ( Tree a b )

l1 = [2,4..]
l2 = ['a','b'..]
l3 = zip l1 l2
h x = x + g x + y
    where g x = x + 1; y = x

z :: a -> (a -> b) -> a
z f a = f


