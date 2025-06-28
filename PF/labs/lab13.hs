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

fct :: Maybe Int -> Maybe Bool
fct mx = do
  x <- mx
  return (pos x) -- comanda `return` doar introduce pe x in monada, nu returneaza nimic

cartesian_product :: Monad m => m a -> m b -> m (a, b)
cartesian_product xs ys = do
  x <- xs
  y <- ys
  return (x, y)

prod :: (a -> b -> c) -> [a] -> [b] -> [c]
prod f xs ys = do
    x <- xs
    y <- ys
    return (f x y)





newtype WriterS a = Writer { runWriter :: (a, [String]) }

instance  Monad WriterS where
  return va = Writer (va, [])
  ma >>= k = let (va, log1) = runWriter ma -- aici se extrage din contextul / monada ma tuplul (a, String)
                 (vb, log2)  = runWriter (k va) -- aici se aplica functia k pe valoarea de tipul a extrasa mai sus, iar apoi se extrage tuplul folosing `runWriter`
              in  Writer (vb, log1 ++ log2)


instance  Applicative WriterS where
  pure = return
  mf <*> ma = do
    f <- mf
    a <- ma
    return (f a)       

instance Functor WriterS where              
  fmap f ma = pure f <*> ma     

tell :: String -> WriterS ()
tell log = Writer ((), [log]) -- operatorul () este folosit, deoarece nu dorim sa avem niciun efect pe primul element din tuplu
  
logIncrement :: Int -> WriterS Int
logIncrement x = do
  tell (show x)
  return (x + 1)

logIncrementN :: Int -> Int -> WriterS Int
logIncrementN x n = do
  if n == 1 then do
    tell ("increment: " ++ show x)
    return (x + 1)
  else do
    new_x <- logIncrementN x (n - 1) -- extrage tuplul (Int, String) din monada, adica (\_(num, msg) -> ...)
    tell ("increment: " ++ show new_x) -- `tell` concateneaza mesajul la string-ul deja existent (se foloseste operatorul >>=)
    return (new_x + 1) -- se reintroduce in monada tuplul dupa ce a fost modificat numarul (operatorul >>= cu functia (+1) si Writer (new_val + 1, msg))



data Person = Person { name :: String, age :: Int } -- name si age sunt field accessors

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

mshowPerson ::  Reader Person String
mshowPerson = do
  person <- Reader id
  -- se extrage environment-ul (acel env din definitie) reader-ului din monada `Reader`, adica un obiect cu tipul `Person`
  -- se foloseste functia identitate `id`, astfel ca Reader id :: `Reader Person Person`
  -- e un mod smecher de a extrage environment-ul in care este aplicata mshowPerson
  return ("NAME: " ++ name person ++ ", AGE: " ++ show (age person))
  -- field accessor-ul `name` se apeleaza pe variabila person extrasa mai sus si se reintroduce in contextul `Reader Person`

{-
Explicatii:
- In comanda `runReader mshowPerson $ Person "ada" 20`, runReader extrage functia de tipul `env -> a` (adica Person -> String)
din monada returnata de mshowPerson, iar apoi o aplica cu parametrul `Person "ada" 20` returnand un String.
- Deci, `mshowPerson` trebuie doar sa returneze un `Reader (Person -> String)` (adica tipul Reader Person String (vezi definitia lui Reader)).
- Pentru aceasta, vom folosi id-ul Reader-ului pentru a extrage environment-ul in care ne aflam, adica persoana (parametrul functiei din Reader)
si apoi sa returnam un String care reprezinta, de fapt, corpul functiei din Reader
-}