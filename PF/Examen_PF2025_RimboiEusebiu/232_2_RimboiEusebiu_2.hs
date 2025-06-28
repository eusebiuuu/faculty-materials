data RGB v = RGB v v v deriving Show
newtype Image p = Img [[p]] deriving Show

class Composite c where
    blend :: (v -> v -> v) -> c (RGB v) -> c (RGB v) -> c (RGB v)


combine :: (v -> v -> v) -> [RGB v] -> [RGB v] -> [RGB v]
combine _ [] [] = []
combine _ [] list = list
combine _ list [] = list
combine f ((RGB r1 v1 a1) : rest1) ((RGB r2 v2 a2) : rest2) = [RGB (f r1 r2) (f v1 v2) (f a1 a2)] ++ (combine f rest1 rest2)

blendL :: (v -> v -> v) -> [[RGB v]] -> [[RGB v]] -> [[RGB v]]
blendL f [] [] = []
blendL f [] list = list
blendL f list [] = list
blendL f (row1 : list1) (row2 : list2) = [combine f row1 row2] ++ (blendL f list1 list2)

instance Composite Image where
    blend f (Img list1) (Img list2) = Img (blendL f list1 list2)

img1 = Img [[(RGB 255 0 0), (RGB 0 255 0), (RGB 0 0 255)]]
img2 = Img [[(RGB 0 255 0), (RGB 255 0 0), (RGB 0 0 255)]]

avg :: Integer -> Integer -> Integer
avg x y = (x + y) `div` 2
