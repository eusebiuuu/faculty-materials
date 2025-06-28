{-
class Functor f where
fmap :: ( a -> b ) -> f a -> f b
-}


data Constant a b = Constant b

instance Functor (Constant a) where
    fmap f (Constant x) = Constant (f x)


data Three a b c = Three a b c

data Three' a b = Three' a b b


data Four a b c d = Four a b c d

data Four'' a b = Four'' a a a b
instance Functor (Four'' a) where
    fmap f (Four'' a1 a2 a3 b1) = Four'' a1 a2 a3 (f b1)

data GoatLord a = NoGoat | OneGoat a | MoreGoats (GoatLord a) (GoatLord a) (GoatLord a)
instance Functor GoatLord where
    fmap f NoGoat = NoGoat
    fmap f (OneGoat a) = OneGoat (f a)
    fmap f (MoreGoats g1 g2 g3) = MoreGoats (fmap f g1) (fmap f g2) (fmap f g3)


data TalkToMe a = Halt | Print String a | Read (String -> a)

instance (Show a) => Show (TalkToMe a) where
    show (Read g) = show (g "heey")

instance Functor TalkToMe where
    fmap f Halt = Halt
    fmap f (Print s t) = Print s (f t)
    fmap f (Read g) = Read (f . g)
dummy :: String -> Int
dummy str = length (str)
main :: IO ()
main = do
    let var = Read dummy
    let new_var = fmap (* 2) var
    let str_to_print = show new_var
    print str_to_print


data Parappa f g a = DaWrappa (f a) (g a) deriving Show

instance (Functor f, Functor g) => Functor (Parappa f g) where
    fmap fct (DaWrappa f1 f2) = DaWrappa (fmap fct f1) (fmap fct f2)

main2 :: IO ()
main2 = do
    let var = DaWrappa [5, 6] (6, 8)
    let res = fmap (+1) var
    let string_to_print = fmap show res
    print string_to_print