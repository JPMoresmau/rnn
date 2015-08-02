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
-- | Utility functions
--
-----------------------------------------------------------------------------

module AI.Network.RNN.Util where

import Control.Parallel.Strategies

-- | Calculate the product of a matrix by a vector, with both represented by a list
listMProd :: (Num a) => [a] -> [a] -> [a]
listMProd mdt vdt = go mdt vdt 0
  where
    go [] _  s = [s]
    go ls [] s = s : go ls vdt 0
    go (y:ys) (x:xs) ix = go ys xs (y*x+ix)

-- | Calculate the mean of a list
calcMeanList :: (Fractional a) => [a] -> a
calcMeanList = uncurry (/) . foldr (\e (s,c) -> (e+s,c+1)) (0,0)

-- | Split a given list into a series of list whose length is given by the first argument
takes :: [Int] -> [a] -> [[a]]
takes [] _ = []
takes (_:sz) [] = [] : takes sz []
takes (s:sz) xs = let
    (one,two) = splitAt s xs
    in one : takes sz two

-- | It's a simple, differentiable sigmoid function.
sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}

-- | Normalize a list of values between 0 and 1
normalize :: (Floating b,Ord b)=> [(a,b)] -> [(a,b)]
normalize xs =
    let
        ws = map snd xs
        mi = minimum ws
        mx = maximum ws
        df = mx-mi
     in  map (\(a,w)->(a,(w-mi)/df)) xs

-- | Parallel zipwith
parZipWith :: (a -> b -> c) -> [a] -> [b] -> [c]
parZipWith f xs ys = withStrategy (parList rseq) $ zipWith f xs ys

-- | Parallel zipWith3
parZipWith3 :: (a -> b -> c -> d) -> [a] -> [b] -> [c] -> [d]
parZipWith3 f xs ys zs = withStrategy (parList rseq) $ zipWith3 f xs ys zs

-- | Number of binary digits necessary to store the given number
binaryDigits :: (Num a, Ord a) => a -> a
binaryDigits a = go 1 1
    where go nb sz
            | nb == a = sz
            | nb > a  = sz - 1
            | otherwise = go (nb*2) (sz+1)

-- | Number of bits needed to store in a sparse manner (with 2 bits) the given number
sparseSize :: (Integral a, Ord a) => a -> a
sparseSize a = go 1
      where
        go nb =
            let pos = nb * (nb - 1) `div` 2
            in if pos >= a
                then nb
                else go (nb+1)

-- | Sparse representation: store all numbers from 0 to n using 2 bits
-- Returns the location of each two bits
sparse :: Int -> [(Int,Int)]
sparse n = let
    sz = sparseSize n
    in take n $ iterate (gen sz) (0,1)
    where gen sz (a,b) =
            if b==sz-1 then (a+1,a+2)
                       else (a,b+1)

