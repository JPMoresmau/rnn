-----------------------------------------------------------------------------
--
-- Module      :  AI.Network.RNN.Util
-- Copyright   :  (c) JP Moresmau
-- License     :  BSD3
--
-- Maintainer  :  JP Moresmau <jp@moresmau.fr>
-- Stability   :  experimental
-- Portability :
--
-- |
--
-----------------------------------------------------------------------------

module AI.Network.RNN.Util where

import Control.DeepSeq
import Control.Parallel.Strategies

listMProd :: (Num a) => [a] -> [a] -> [a]
listMProd mdt vdt = go mdt vdt 0
  where
    go [] _  s = [s]
    go ls [] s = s : go ls vdt 0
    go (y:ys) (x:xs) ix = go ys xs (y*x+ix)

calcMeanList :: (Fractional a) => [a] -> a
calcMeanList = uncurry (/) . foldr (\e (s,c) -> (e+s,c+1)) (0,0)

takes :: [Int] -> [a] -> [[a]]
takes szs ls = go szs ls
    where
        go [] _ = []
        go (_:sz) [] = [] : go sz []
        go (s:sz) xs = let
            (one,two) = splitAt s xs
            in one : go sz two

-- | It's a simple, differentiable sigmoid function.
sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}


normalize :: (Floating b,Ord b)=> [(a,b)] -> [(a,b)]
normalize xs =
    let
        ws = map snd xs
        mi = minimum ws
        mx = maximum ws
        df = mx-mi
        --df = if mi < 0 then (-mi) else 0
        --(w-mi)/df)
    in  map (\(a,w)->(a,(w-mi)/df)) xs
     --map (\(a,w)->(a,(w+df)**4)) xs

parZipWith f xs ys = withStrategy (parList rdeepseq) $ zipWith f xs ys

binaryDigits a = go 1 1
    where go nb sz
            | nb == a = sz
            | nb > a  = sz - 1
            | otherwise = go (nb*2) (sz+1)
