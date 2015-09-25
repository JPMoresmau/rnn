{-# LANGUAGE BangPatterns #-}
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
import Control.Monad.Random

import Numeric.LinearAlgebra.HMatrix

import Data.List
import Data.Ord
import qualified Data.Set as S

-- | Calculate the product of a matrix by a vector, with both represented by a list
listMProd :: (Num a) => [a] -> [a] -> [a]
listMProd mdt vdt = go mdt vdt 0
  where
    go [] _  !s = [s]
    go ls [] !s = s : go ls vdt 0
    go (y:ys) (x:xs) !ix = go ys xs (y*x+ix)
--    map (foldr (\(a,b) c->a*b+c) 0) $ go mdt vdt []
--  where
--    go [] _  s = [s]
--    go ls [] s = s : go ls vdt []
--    go (y:ys) (x:xs) acc = go ys xs ((y,x):acc)
--listMProd mdt vdt = let (_,l,lst)= foldl' go (vdt,[],0) mdt
--    in reverse $ lst:l
--    where
--        go ([],acc,ix) m =
--            let (x:xs)=vdt
--            in (xs,ix:acc,x*m)
--        go ((x:xs),acc,ix) m = (xs,acc,x*m+ix)
--listMProd mdt vdt = let (_,l,lst)= foldr go (rvdt,[],0) mdt
--    in lst:l
--    where
--        rvdt = reverse vdt
--        go m ([],acc,ix) =
--            let (x:xs)=rvdt
--            in (xs,ix:acc,x*m)
--        go m ((x:xs),acc,ix) = (xs,acc,x*m+ix)
--listMProd mdt vdt = map sum $ takes1 (length vdt) $ zipWith (*) mdt (cycle vdt)

-- | Calculate the mean of a list
calcMeanList :: (Fractional a) => [a] -> a
calcMeanList = uncurry (/) . foldr (\e (s,c) -> (e+s,c+1)) (0,0)

takes1 :: Int -> [a] -> [[a]]
takes1 _ [] = []
takes1 s xs = let
    (one,two) = splitAt s xs
    in one : takes1 s two


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

-- | Standard Normal distribution
stdNormal :: (Monad m,RandomGen g) => RandT g m (Double,Double)
stdNormal = do
    (x1,x2,w1)<-gen
    let w  = sqrt( (-2 * log( w1 ) ) / w1 )
        y1 = x1 * w;
        y2 = x2 * w;
    return (y1,y2)
    where
        ranx = do
            r1 <- getRandomR (0,1)
            return (2 * r1 - 1)
        gen = do
            x1 <- ranx
            x2 <- ranx
            let w = x1 * x1 + x2 * x2
            if w>=1 then gen else return (x1,x2,w)

softmax :: [Double] -> [Double]
softmax is =
    let den = sum $ map exp is
    in map (( / den) . exp) is

softmaxV :: Vector Double -> Vector Double
softmaxV is =
    let den = sumElements $ cmap exp is
    in cmap (( / den) . exp) is

euclidian :: Vector Double -> Vector Double -> Double
euclidian v1 v2 = sumElements $ cmap (**2) (v1 - v2)

equilateralEncoding :: Int -> Matrix Double
equilateralEncoding n =
    let z1 = replicate (n-2) 0
        m = (n><(n-1)) (-1: z1 ++ (1 : z1) ++ repeat 0)
    in foldl' pass m [2..n-1]
    where
        r :: Double -> Double
        r k = -1 / k
        f :: Double -> Double
        f k = sqrt (k*k -1) / k
        pass :: Matrix Double -> Int -> Matrix Double
        pass m1 ik =
            let k = fromIntegral ik
                f1 = f k
                a0 = concatMap (\i->map (\j->((i,j),f1)) [0..ik-2]) [0..ik-1]
                m2 = accum m1 (*) a0
                r1 = r k
                a1 = map (\x->((x,ik-1),r1)) [0..ik-1] ++ [((ik,ik-1),1)]
                m3 = accum m2 const a1
            in m3

equilateralDecoding :: Matrix Double -> Vector Double -> Int
equilateralDecoding m os =
    let rs = zip (toRows m) [0..]
        es = map (\(vs,ix)->(euclidian vs os,ix)) rs
    in snd $ minimumBy (comparing fst) es

roundTo :: (Fractional a, Integral b, RealFrac r) =>
                 b -> r -> a
roundTo n f=  (fromInteger $ round $ f * (10^n)) / (10.0^^n)

ordNub :: (Ord a) => [a] -> [a]
ordNub l = go S.empty l
     where
       go _ []     = []
       go s (x:xs) = if x `S.member` s then go s xs
                                     else x : go (S.insert x s) xs
