verifL :: [Int] -> Bool
verifL = undefined

takefinal :: [Int] -> Int -> [Int]
takefinal l n = drop (length(l) - n) l

remove :: [Int] -> Int -> [Int]
remove l n = (take (n - 1) l) ++ (drop n l)

myreplicate :: Int -> Int -> [Int]
myreplicate 0 v = []
myreplicate n v = v : myreplicate (n - 1) v 

-- semiPareRec [0,2,1,7,8,56,17,18] == [0,1,4,28,9]
semiPareRec :: [Int] -> [Int]
semiPareRec [] = []
semiPareRec (h:t)
 | even h    = h `div` 2 : t'
 | otherwise = t'
 where t' = semiPareRec t
 

sumImp :: [Int] -> Int
sumImp [] = 0
sumImp (x:l)
    | even x    = sumImp l
    | otherwise = x + (sumImp l)


totalLen :: [String] -> Int
totalLen = undefined

countVocale :: String -> Int
countVocale [] = 0;
countVocale (c:rest)
    | c == 'a' = 1 + countVocale rest
    | c == 'e' = 1 + countVocale rest
    | c == 'i' = 1 + countVocale rest
    | c == 'o' = 1 + countVocale rest
    | c == 'u' = 1 + countVocale rest
    | otherwise = countVocale rest

nrVocale :: [String] -> Int
nrVocale [] = 0
nrVocale (str:rest)
    | str == reverse(str) = (countVocale str) + nrVocale rest
    | otherwise = nrVocale rest

-- nrVocale ["sos", "civic", "palton", "desen", "aerisirea"] == 9

-- f 3 [1,2,3,4,5,6] = [1,2,3,3,4,3,5,6,3]

inInterval :: Int -> Int -> [Int] -> [Int]
inInterval x y l = [n | n <- l, n >= x, n <= y]


pozImp :: [Int] -> [Int]
pozImp l = [fst x | x <- zip l [0..], odd(snd x)]


semiPareComp :: [Int] -> [Int]
semiPareComp l = [ x `div` 2 | x <- l, even x ]


get_perm :: Eq a => [a] -> [[a]]
get_perm [] = [[]]
get_perm list = [x : perm | x <- list, perm <- get_perm (remove_elem x list)]
    where
    remove_elem x [] = []
    remove_elem x (y : rest)
        | y == x = rest
        | otherwise = y : remove_elem x rest

get_comb :: [a] -> Int -> [[a]]
get_comb list 0 = [[]]
get_comb (h:rest) k
    | length(rest) < k = [h : comb | comb <- get_comb rest (k - 1)]
    | otherwise        = [h : comb | comb <- get_comb rest (k - 1)] ++ (get_comb rest k)

get_arrangements :: Eq a => [a] -> Int -> [[a]]
get_arrangements list 0 = [[]]
get_arrangements list k = [x : arr | x <- list, arr <- get_arrangements (remove_elem x list) (k - 1)]
    where
    remove_elem x [] = []
    remove_elem x (y : rest)
        | y == x = rest
        | otherwise = y : remove_elem x rest

-- ASA NU!
-- place_queens :: Int -> Int -> [[(Int, Int)]]
-- place_queens dim 1 = [[(x, y)] | x <- [1..dim], y <- [1..dim]]
-- place_queens dim cnt = [(x, y) : qlist | x <- [1..dim], y <- [1..dim], qlist <- place_queens dim (cnt - 1), q <- qlist, x /= (fst q) && y < (snd q) && (x - y) /= ((fst q) - (snd q)) && (x + y) /= ((fst q) + (snd q))]

isSafe :: (Int, Int) -> [(Int, Int)] -> Bool
isSafe (x, y) queens = all (\(qx, qy) -> x /= qx && y < qy && (x - y) /= (qx - qy) && (x + y) /= (qx + qy)) queens

place_queens :: Int -> Int -> [[(Int, Int)]]
place_queens dim 1 = [[(x, y)] | x <- [1..dim], y <- [1..dim]]
place_queens dim cnt = [(x, y) : qlist | x <- [1..dim], y <- [1..dim], qlist <- (place_queens dim (cnt - 1)), (isSafe (x, y) qlist)]