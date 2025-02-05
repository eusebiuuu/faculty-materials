allVerifies :: (Int -> Bool) -> [Int] -> Bool
allVerifies prop list = length (filter prop list) == length list

anyVerifies :: (Int -> Bool) -> [Int] -> Bool
anyVerifies prop list = length (filter prop list) > 0

listToInt :: [Int] -> Int
listToInt list = foldl (\num x -> num * 10 + x) 0 list

rmChar :: Char -> String -> String
rmChar c str = foldr (\x acc -> if c == x then acc else x : acc) [] str

rmStr :: String -> String -> String
rmStr rmlist str = foldr (\x acc -> rmChar x acc) str rmlist

myElem :: Eq a => a -> [a] -> Bool
myElem x list = length (filter (== x) list) > 0

myUnzip :: [(a, b)] -> ([a], [b])
myUnzip list = foldr (\(x, y) acc -> (x : fst acc, y : snd acc)) ([], []) list

intersect :: [Int] -> [Int] -> [Int]
intersect l1 l2 = filter (\x -> x `elem` l2) l1

-- getLists [Int] -> [[Int]]
-- getLists [] = [[]]
-- getLists (x:rest) = [[x] : getLists rest]


addElem :: Int -> [[Int]] -> [[Int]]
addElem x lists = map (x :) $ filter (\ll -> not $ x `elem` ll) lists


perms :: [Int] -> [[Int]]
perms [] = [[]]
perms list = concat $ map (\ e -> addElem e (perms $ filter (/= e) list)) list