-- link model: https://drive.google.com/drive/folders/1y0xF0X2VwgpVit3JI3cZjq37bRCudogG
-- Ex 1:

data Point = Pt [Int] deriving Show

data Arb = Empty | Node Int Arb Arb deriving Show

class ToFromArb a where
    toArb :: a -> Arb
    fromArb :: Arb -> a

insert :: Int -> Arb -> Arb
insert elem Empty = Node elem Empty Empty
insert elem (Node val left right)
    | elem <= val = Node val (insert elem left) right
    | otherwise = Node val left (insert elem right)

concatCoord :: Arb -> [Int]
concatCoord Empty = [] :: [Int]
concatCoord (Node val left right) = (concatCoord left) ++ [val] ++ (concatCoord right) :: [Int]

instance ToFromArb Point where
    toArb (Pt []) = Empty
    toArb (Pt (x : rest)) = insert x (toArb (Pt rest))
    fromArb arb = Pt (concatCoord arb)
    
-- Prelude> toArb (Pt [3, 4, 1])
-- Prelude> fromArb (Node 1 Empty (Node 4 (Node 3 Empty Empty) Empty)) :: Point


-- Ex 2:

-- solutie fara monade
getFromInterval :: Int -> Int -> [Int] -> [Int]
-- getFromInterval l r list = filter (\x -> l <= x && r >= x) list

-- solutie cu monade

-- varianta mea
getFromInterval _ _ [] = []
getFromInterval l r (elem : rest) = do
    let res = getFromInterval l r rest -- aici res e de tipul [Int], nu l-am scos din context
    if l <= elem && elem <= r then
        (return elem) ++ res
    else
        res

-- varianta DeepSeek (ChatGPT nu a stiu, ce prost e)
-- nu prea inteleg ce a facut aici, pare cam magie; oricum solutia mea e mai buna
-- good to know daca mai intalnim liste
-- getFromInterval l r list = do
--     x <- list -- se extrag toate elementele din context si se pun in x
--     if x >= l && x <= r -- aici se verifica conditia pentru fiecare element
--         then return x
--         else []


-- Ex 3

newtype ReaderWriter env a = RW { getRW :: env -> (a, String) }

instance Monad (ReaderWriter env) where
    return x = RW (\_ -> (x, ""))
    frw >>= g = RW f -- >>= trebuie sa returneze un obiect de tipul RW (env -> (a, String))
        where f env = let (var1, str1) = (getRW frw) env -- ca sa aplicam f, mai intai extragem functia din frw cu getRW si o aplicam
                          (var2, str2) = getRW (g (var1)) env
                          -- apoi ne folosim de pasul precedent si aplicam pe g (functia cu side effects din >>=) la rezultatul de tip `a` din f
                          -- g va returna un obiect de tip `m b`, in acest caz (RW (env -> (b, String)))
                          -- acestuia din urma i se va scoate functia cu getRW si se va apela cu `env` dat ca parametru
                          -- rezultatul va fi tot un tuplu `(a, String)`
                    in (var2, str1 ++ str2)
                    -- stiind toate informatiile, returnam tuplul cu string-urile concatenate si a 2-a valoare de tipul a (adica var2)
                    -- pentru ca nu am aplicat degeaba functia g

instance Applicative (ReaderWriter env) where
    pure = return
    mf <*> ma = do
        f <- mf
        x <- ma
        return (f x)

instance Functor (ReaderWriter env) where
    fmap f rwa = pure f <*> rwa


-- utilizare (optional)

data Person = Person { name :: String, age :: Int }

mshowPerson ::  ReaderWriter Person String
mshowPerson = do
  person <- RW (\ env -> (env, "")) -- extragem environment-ul din RW si obtinem in person tipul (Person, String)
  return ("NAME: " ++ name person ++ ", AGE: " ++ show (age person))
  -- am aplicat functia name pe un obiect de tip (Person, String), deci se va apela doar pentru primul parametru (cel de tipul `a`)
  -- de ce? vezi definitia operatorului (>>=) din instanta `Monad (ReaderWriter env)`
  -- nu uitati!!! : intre liniile din `do` este operatorul (>>=)

