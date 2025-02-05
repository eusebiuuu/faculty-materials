import Data.List

myInt = 31415926535897932384626433832795028841971693993751058209749445923

double :: Integer -> Integer
double x = x+x

maxim :: Integer -> Integer -> Integer
maxim x y = 
    if (x > y)
        then x
        else y

maxim3 :: Integer -> Integer -> Integer -> Integer
maxim3 x y z =
    let 
        curr_max = maxim y z
    in 
        maxim x curr_max

max3 x y z = let
             u = maxim x y
             in (maxim  u z)

fizzbuzz :: Integer -> String
fizzbuzz x 
    | x `mod` 2 == 0 = "par"
    | x `mod` 3 == 0 = "3 div"
    | otherwise = "wtf"

fibonacciCazuri :: Integer -> Integer
fibonacciCazuri n
    | n < 2     = n
    | otherwise = fibonacciCazuri (n - 1) + fibonacciCazuri (n - 2)
    
fibonacciEcuational :: Integer -> Integer
fibonacciEcuational 0 = 0
fibonacciEcuational 1 = 1
fibonacciEcuational n =
    fibonacciEcuational (n - 1) + fibonacciEcuational (n - 2)
    
tribonacci :: Integer -> Integer
tribonacci 1 = 1
tribonacci 2 = 1
tribonacci 3 = 2
tribonacci n = tribonacci (n - 1) + tribonacci (n - 2) + tribonacci (n - 3)

binomial :: Integer -> Integer -> Integer
binomial n 0 = 1
binomial 0 k = 0
binomial n k = binomial (n - 1) k + binomial (n - 1) (k - 1)


squared_sum :: Integer -> Integer -> Integer
squared_sum x y = 
    x * x + y * y

get_parity :: Integer -> [Char]
get_parity num =
    if (num `mod` 2 == 0)
        then "par"
        else "impar"

fact :: Integer -> Integer
fact 0 = 1
fact n = n * fact (n - 1)

get_max :: [Integer] -> Integer
get_max [t] = t
get_max (first:rest) = maxim first (get_max rest)

poly :: Double -> Double -> Double -> Double -> Double
poly a b c x = a * x ^ 2 + b * x + c