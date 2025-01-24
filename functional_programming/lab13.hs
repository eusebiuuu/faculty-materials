-- Monada Maybe este definita in GHC.Base 

-- instance Monad Maybe where
--   return = Just
--   Just va  >>= k   = k va -- k is a function that return Maybe b
--   Nothing >>= _   = Nothing


-- instance Applicative Maybe where
--   pure = return
--   mf <*> ma = do
--     f <- mf
--     va <- ma
--     return (f va)       

-- instance Functor Maybe where
--   fmap f ma = pure f <*> ma


pos :: Int -> Bool
pos  x = if (x>=0) then True else False

fct :: Maybe Int ->  Maybe Bool
fct  mx =  mx  >>= (\x -> Just (pos x))

fctDo :: Maybe Int -> Maybe Bool
fctDo mx = do
    x <- mx
    return (pos x)

addM :: Maybe Int -> Maybe Int -> Maybe Int
addM mx my = mx >>= (\x -> (my >>= (\y -> Just (x + y))))


cartesian_product :: Monad m => m a -> m b -> m (a, b)
cartesian_product xs ys = xs >>= ( \x -> (ys >>= \y-> return (x,y)))

prod :: (a -> b -> c) -> [a] -> [b] -> [c]
-- prod f xs ys = [f x y | x <- xs, y<-ys]
prod f xs ys = do
    x <- xs
    y <- ys
    return (f x y)


myGetLine :: IO String
myGetLine = getChar >>= \x ->
      if x == '\n' then
          return []
      else
          myGetLine >>= \xs -> return (x:xs)

myGetLineDo :: IO String
myGetLineDo = do
    x <- getChar
    if x == '\n' then
        return [] -- return inserts the [] in the monad, doesn't return anything
    else do
        xs <- myGetLineDo
        return (x : xs)

prelNo noin =  sqrt noin

ioNumber :: IO Float -> IO ()
ioNumber = do
     noin  <- readLn :: IO Float
     putStrLn $ "Intrare\n" ++ (show noin)
     let  noout = prelNo noin
     putStrLn $ "Iesire"
     print noout

data Person = Person { name :: String, age :: Int }

showPersonN :: Person -> String
showPersonN = undefined
showPersonA :: Person -> String
showPersonA = undefined

{-
showPersonN $ Person "ada" 20
"NAME: ada"
showPersonA $ Person "ada" 20
"AGE: 20"
-}

showPerson :: Person -> String
showPerson = undefined 

{-
showPerson $ Person "ada" 20
"(NAME: ada, AGE: 20)"
-}


newtype Reader env a = Reader { runReader :: env -> a }


instance Monad (Reader env) where
  return x = Reader (\_ -> x)
  ma >>= k = Reader f
    where f env = let a = runReader ma env
                  in  runReader (k a) env



instance Applicative (Reader env) where
  pure = return
  mf <*> ma = do
    f <- mf
    a <- ma
    return (f a)       

instance Functor (Reader env) where              
  fmap f ma = pure f <*> ma    

mshowPersonN ::  Reader Person String
mshowPersonN = undefined
mshowPersonA ::  Reader Person String
mshowPersonA = undefined 
mshowPerson ::  Reader Person String
mshowPerson = undefined 
{-
runReader mshowPersonN  $ Person "ada" 20
"NAME:ada"
runReader mshowPersonA  $ Person "ada" 20
"AGE:20"
runReader mshowPerson  $ Person "ada" 20
"(NAME:ada,AGE:20)"
-}