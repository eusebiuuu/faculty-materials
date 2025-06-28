-- link: https://github.com/FMI-Materials/FMI-Materials/blob/main/Year%20II/Semester%20I/Programare%20Functionala/Modele%20Examen/2023%20-%202024/Subiect%2004.jpg

import Data.Char
import Data.Either

-- Ex 1

transformString :: String -> String
transformString str = do
    ch <- str
    if isLower ch then
        return ('*')
    else if isDigit ch then
        [ch] ++ [ch]
    else if elem ch "ADT" then
        []
    else
        return (ch)

-- Ex 2

data Pereche a b = MyP a b deriving Show

data Lista a = MyL [a] deriving Show

class MyOp m where
    myFilter :: (a -> Bool) -> (b -> Bool) -> m (Pereche a b) -> m (Pereche a b)

instance MyOp Lista where
    myFilter _ _ (MyL []) = MyL []
    myFilter f1 f2 (MyL list) = MyL (filter (\(MyP x y) -> (f1 x) && (f2 y)) list)

lp :: Lista (Pereche Int Char)
lp = MyL [MyP 97 'a', MyP 3 'b', MyP 100 'd']

-- Ex 3


newtype AE a = AE { getAE :: (Either String a, String) } deriving Show

extractRight :: Either a b -> b
extractRight (Right x) = x
extractRight (Left _) = error "Tried to extract Right from Left!"

instance Monad AE where
    return x = AE (Right x, "")
    AE (Right val, msg1) >>= g =
        let (result, msg2) = getAE (g val)
        in AE (result, msg1 ++ msg2)

    AE (Left err, msg) >>= _ = AE (Left err, msg)

instance Applicative AE where
    pure = return
    aef <*> aev = do
        f <- aef
        f <$> aev

instance Functor AE where
    fmap f aev = pure f <*> aev

testAE :: AE Int
testAE = ma >>= f
    where
        ma = AE (Right 8, "ana are ")
        f x = AE (if x `mod` 2 == 0 then Right x else Left "error", "mere si pere")

