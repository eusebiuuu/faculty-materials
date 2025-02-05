import Data.List (nub)
import Data.Maybe (fromJust)

type Nume = String
data Prop
  = Var Nume
  | F
  | T
  | Not Prop
  | Prop :|: Prop
  | Prop :&: Prop
  -- | Prop :->: Prop
  -- | Prop :<->: Prop
  deriving (Eq, Read)
infixr 2 :|:
infixr 3 :&:

p1 :: Prop
p1 = (Var "P" :|: Var "Q") :&: (Var "P" :&: Var "Q")

p2 :: Prop
p2 = (Var "P" :|: Var "Q") :&: (Not (Var "P") :&: Not (Var "Q"))

instance Show Prop where
  show (Var s) = s
  show F = show " false "
  show T = show " true "
  show (Not p) = " ~ " ++ show p
  show (p1 :|: p2) = "(" ++ show p1 ++ " | " ++ show p2 ++ ")"
  show (p1 :&: p2) = "(" ++ show p1 ++ " & " ++ show p2 ++ ")"
 

type Env = [(Nume, Bool)]

impureLookup :: Eq a => a -> [(a,b)] -> b
impureLookup a = fromJust . lookup a


eval :: Prop -> Env -> Maybe Bool
eval (Var s) val = if (lookup s val) == Nothing then Nothing else lookup s val
eval (Not p) val = if res == Nothing then Nothing else Just (not (fromJust res))
  where res = eval p val
eval (p1 :|: p2) val = if res1 == Nothing || res2 == Nothing then Nothing else Just ((fromJust res1) || (fromJust res2))
  where res1 = eval p1 val; res2 = eval p2 val
eval (p1 :&: p2) val = if res1 == Nothing || res2 == Nothing then Nothing else Just ((fromJust res1) && (fromJust res2))
  where res1 = eval p1 val; res2 = eval p2 val
 
test_eval = eval  (Var "P" :|: Var "Q") [("P", True), ("Q", False)] == Just True


variabile :: Prop -> [Nume]
variabile (Var s) = [s]
variabile (Not p) = variabile p
variabile (p1 :|: p2) = nub ((variabile p1) ++ (variabile p2))
variabile (p1 :&: p2) = nub ((variabile p1) ++ (variabile p2))
 
test_variabile =
  variabile (Not (Var "P") :&: Var "Q") == ["P", "Q"]

get_bits :: Int -> Int -> [Bool]
get_bits bit 1 = [bit == 1]
get_bits num len = (get_bits (num `div` 2) (len - 1)) ++ [num `mod` 2 == 1]

envs :: [Nume] -> [Env]
envs names = [zip names (get_bits curr_num len) | curr_num <- [0 .. (2 ^ len - 1)]]
  where len = length(names)


test_envs = 
    envs ["P", "Q"]
    ==
    [ [ ("P",False)
      , ("Q",False)
      ]
    , [ ("P",False)
      , ("Q",True)
      ]
    , [ ("P",True)
      , ("Q",False)
      ]
    , [ ("P",True)
      , ("Q",True)
      ]
    ]

satisfiabila :: Prop -> Bool
satisfiabila p = foldr (\v acc -> if v == Nothing then acc else acc || fromJust v) False [eval p vals | vals <- envs (variabile p)]


test_satisfiabila1 = satisfiabila (Not (Var "P") :&: Var "Q") == True
test_satisfiabila2 = satisfiabila (Not (Var "P") :&: Var "P") == False

valida :: Prop -> Bool
valida = undefined

test_valida1 = valida (Not (Var "P") :&: Var "Q") == False
test_valida2 = valida (Not (Var "P") :|: Var "P") == True

get_eval_array :: Prop -> [Nume] -> [Maybe Bool]
get_eval_array p vars = [eval p vals | vals <- envs vars]

echivalenta :: Prop -> Prop -> Bool
echivalenta p1 p2 = (get_eval_array p1 all_var) == (get_eval_array p2 all_var)
  where all_var = nub ((variabile p1) ++ (variabile p2))
 
test_echivalenta1 =
  True
  ==
  (Var "P" :&: Var "Q") `echivalenta` (Not (Not (Var "P") :|: Not (Var "Q")))
test_echivalenta2 =
  False
  ==
  (Var "P") `echivalenta` (Var "Q")
test_echivalenta3 =
  True
  ==
  (Var "R" :|: Not (Var "R")) `echivalenta` (Var "Q" :|: Not (Var "Q"))

