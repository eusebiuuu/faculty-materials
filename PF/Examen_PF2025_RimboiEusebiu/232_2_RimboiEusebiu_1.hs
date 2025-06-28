-- fara monade

isValid :: [Int] -> Int -> Int -> Bool
isValid [] _ _ = True
isValid (elem : rest) l r
    | elem < l || elem > r = isValid rest l r
    | otherwise = False

getList :: [[Int]] -> Int -> Int -> [Int]
getList [] _ _ = []
getList (list : rest) l r
    | ((length list) `mod` 2 == 1) && (isValid list l r) = list ++ (getList rest l r)
    | otherwise = getList rest l r


-- cu monade

getListM :: [[Int]] -> Int -> Int -> [Int]
getListM list l r = do
    curr_list <- list
    if (length curr_list) `mod` 2 == 1 && (isValid curr_list l r) then
        curr_list
    else
        []

