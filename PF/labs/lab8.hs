class Collection c where
    empty :: c key value
    singleton :: key -> value -> c key value
    insert :: Ord key => key -> value -> c key value -> c key value
    lookup :: Ord key => key -> c key value -> Maybe value
    delete :: Ord key => key -> c key value -> c key value
    keys :: c key value -> [key]
    values :: c key value -> [value]
    toList :: c key value -> [(key, value)]
    fromList :: Ord key => [(key,value)] -> c key value

newtype PairList k v = PairList { getPairList :: [(k, v)] }

instance Collection PairList where
    empty = PairList []
    singleton key val = PairList [(key, val)]
    insert key val coll_pair = PairList ((key, val) : (getPairList coll_pair))
    toList coll = getPairList coll
    keys coll = [curr_key | (curr_key, curr_val) <- (toList coll)]
    values coll = [curr_val | (curr_key, curr_val) <- (toList coll)]
    lookup k coll = getElem k (getPairList coll) where
        getElem _ [] = Nothing
        getElem k ((x, y) : rest)
            | x == k = Just y
            | otherwise = getElem k rest


-----------------------

data Punct = Pt [Int]

data Arb = Vid | F Int | N Arb Arb
    deriving Show

instance Show Punct where
    show (Pt coords) = "(" ++ showTuple coords ++ ")"
      where
        showTuple [] = ""
        showTuple [x] = show x
        showTuple (x:xs) = show x ++ ", " ++ showTuple xs

class ToFromArb a where
    toArb :: a -> Arb
    fromArb :: Arb -> a

instance ToFromArb Punct where
    toArb (Pt list) = showTree (Pt list)
      where
        showTree (Pt []) = Vid
        showTree (Pt [x]) = F x
        showTree (Pt (top:rest)) = N (showTree (Pt [top])) (showTree (Pt rest))
    
    fromArb (Vid) = Pt []
    fromArb (F x) = Pt [x]
    fromArb (N t1 t2) = concatPt (fromArb t1) (fromArb t2)
      where
        concatPt (Pt x1) (Pt x2) = Pt (x1 ++ x2)
    

data Geo a = Square a | Rectangle a a | Circle a
    deriving Show

class GeoOps g where
    perimeter :: (Floating a) => g a -> a
    area :: (Floating a) => g a -> a

instance GeoOps Geo where
    perimeter (Square a) = 4 * a
    perimeter (Circle a) = 2 * a * pi
    perimeter (Rectangle a b) = 2 * (a + b)
    area (Square a) = a * a
    area (Circle a) = pi * a * a
    area (Rectangle a b) = a * b


instance (Eq a, Floating a) => Eq (Geo a) where
    g1 == g2 = (perimeter (g1)) == (perimeter (g2))


    
