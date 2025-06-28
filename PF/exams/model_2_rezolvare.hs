-- link: https://github.com/FMI-Materials/FMI-Materials/blob/main/Year%20II/Semester%20I/Programare%20Functionala/Modele%20Examen/2023%20-%202024/Subiect%2001.pdf

import Data.Maybe -- to use fromJust
-- Ex 1

data Expr = Var String | Val Int | Expr :+: Expr | Expr :*: Expr deriving (Show, Eq)

class Operations exp where
    simplify :: exp -> exp

instance Operations Expr where
    simplify (Var s) = Var s
    simplify (Val num) = Val num
    simplify ((Val 0) :+: e2) = simplify e2
    simplify (e1 :+: (Val 0)) = simplify e1
    simplify (e1 :+: e2)
        | (simplify e1) == (Val 0) = (simplify e2)
        | (simplify e2) == (Val 0) = (simplify e1)
        | otherwise = (simplify e1) :+: (simplify e2)
    simplify (e1 :*: e2)
        | (simplify e1) == (Val 1) = (simplify e2)
        | (simplify e2) == (Val 1) = (simplify e1)
        | (simplify e1) == (Val 0) = (Val 0)
        | (simplify e2) == (Val 0) = (Val 0)
        | otherwise = (simplify e1) :*: (simplify e2)


ex1 = ((Val 1) :+: (Var "x")) :*: (Val 1)
ex2 = ex1 :+: (Val 3)
ex3 = (((Val 0) :*: (Val 2)) :+: (Val 3)) :*: (Val 0)
ex4 = ex3 :*: Val 5


-- Ex 2

convertString :: String -> String
convertString [] = []
convertString (ch : rest)
    | elem ch "AEIOUaeiou" = [ch] ++ "p" ++ [ch] ++ (convertString rest)
    | otherwise = [ch] ++ convertString rest


convertStringM :: String -> String
convertStringM [] = []
convertStringM (ch : rest) = do
    let res = convertStringM rest
    if elem ch "AEIOUaeiou" then
        return (ch) ++ "p" ++ return (ch) ++ res
    else
        return (ch) ++ res


-- Ex 3

newtype ReaderM env a = ReaderM { runReaderM :: env -> Maybe a }

instance Monad (ReaderM env) where
    return x = ReaderM (\_ -> Just x)
    frm >>= g = ReaderM f
        where f env = let var = (runReaderM frm) env
                      in if isNothing(var) then Nothing else (runReaderM (g (fromJust var))) env


instance Applicative (ReaderM env) where
    pure = return
    rmf <*> rmv = do
        f <- rmf
        v <- rmv
        return (f v)

instance Functor (ReaderM env) where
    fmap f rmv = pure f <*> rmv

testReaderM :: ReaderM String String
testReaderM = ma >>= k
    where ma = ReaderM (\ str -> if length str > 10 then Just (length str) else Nothing)
          k val = ReaderM (\ str -> if val `mod` 2 == 0 then Just "par" else Nothing)

-- se testeaza cu comanda `runReaderM testReaderM <string_here>`
