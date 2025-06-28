{-
[x^2 |x <- [1..10], x `rem` 3 == 2]
[(x,y) | x <- [1..5], y <- [x..(x+2)]]
[(x,y) | x <- [1..3], let k = x^2, y <- [1..k]]
[x | x <- "Facultatea de Matematica si Informatica", elem x ['A'..'Z']]
[[x..y] | x <- [1..5], y <- [1..5], x < y]
-}
factori :: Int -> [Int]
factori = undefined
prim :: Int -> Bool
prim = undefined
numerePrime :: Int -> [Int]
numerePrime = undefined
ordonataNat :: [Int] -> Bool
ordonataNat [] = True
ordonataNat [x] = True
-- ordonataNat (x:xs) = 
ordonata :: [a] -> (a -> a -> Bool) -> Bool
ordonata = undefined

myzip3 :: [Int] -> [Int] -> [Int] -> [(Int, Int, Int)]
myzip3 l1 l2 l3 = [(x, y, z) | (x, (y, z)) <- zip l1 $ zip l2 l3]

sum_sq_pos :: [Int] -> Int
-- sum_sq_pos = sum . map (^ 2) . filter (> 0)
sum_sq_pos = foldr (+) 0 . map (^ 2) . filter (> 0)

rev :: [Int] -> [Int]
rev list = foldl (\l x -> (x:l)) [] list

sqr_odd_pos :: [Int] -> [Int]
sqr_odd_pos list = map sqr $ map fst $ filter (\(x, y) -> odd y) $ zip list [1..] where sqr x = x * x

remove_cons :: [String] -> [String]
remove_cons list = map (\word -> filter (\c -> not $ elem c "AEIOUaeiou") word) list

mymap :: (a -> b) -> [a] -> [b]
mymap f [] = []
mymap f (h:rest) = ((f h) : (mymap f rest))

