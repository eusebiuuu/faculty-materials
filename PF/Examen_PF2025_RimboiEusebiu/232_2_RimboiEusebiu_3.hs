
import Data.Maybe

data Fuel a = Fuel { getFuel :: Integer -> Integer -> Maybe (Integer, a) }


instance Monad Fuel where
    return x = Fuel (\a -> (\b -> Just (b, x)))
    fuelv >>= g = Fuel f where
        f x y =
            let res = (getFuel fuelv) x y
            in if isNothing res then Nothing else
                let (a, b) = fromJust res
                in getFuel (g b) x y


