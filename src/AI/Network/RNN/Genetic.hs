{-# LANGUAGE FlexibleContexts #-}
-----------------------------------------------------------------------------
--
-- Module      :  AI.Network.RNN.Genetic
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

module AI.Network.RNN.Genetic where

import Control.Monad
import Control.Monad.Random hiding (fromList)
import Control.DeepSeq
import AI.GeneticAlgorithm.Simple
import AI.Network.RNN.RNN
import Numeric.LinearAlgebra.HMatrix


data RNNData = RNNData RNNetwork [Vector Double] [Vector Double] Double
    deriving (Show,Read,Eq)

instance NFData RNNData where
    rnf (RNNData rnn _ _ _) = rnf rnn

errorThreshold :: Double
errorThreshold = 0.1

instance Chromosome RNNData where
    crossover (RNNData rnn1 is os l) (RNNData rnn2 _ _ _) = do
        rnns <- crossNetworkHalf rnn1 rnn2
        return $ map (\r->RNNData r is os l) rnns

    mutation (RNNData rnn1 is os l) = do
        rnn2 <- mutateNetwork rnn1
        return $ RNNData rnn2 is os l

    fitness (RNNData rnn1 is os l) =
        let (_,res) = evalSteps rnn1 is
            err a b    = (sum $ zipWith (\c d -> (c- d)**2 ) (toList a) (toList b)) / fromIntegral (size a)
         --   z = zip os res
         --   ok = length $ takeWhile (\(a,b)->err a b < errorThreshold) z
        in 1 / ((sum $ zipWith err os res) / l)
                --  if ok > 0
                --    then fromIntegral ok - (uncurry err $ head $ drop ok z)
                --    else - (err (head os) (head res))

crossNetworkFull :: RandomGen g =>  RNNetwork -> RNNetwork -> Rand g [RNNetwork]
crossNetworkFull rnn1 rnn2 = {-# SCC "crossNetworkFull" #-} do
    let v1 = networkToVector rnn1
        v2 = networkToVector rnn2
    idx <- getRandomR (0, size v1 - 1)
    let (v3,v4) = crossVector v1 v2 idx
        rnn3 = case createNetworkFromVector (networkDimensions rnn1) v3 of
            Right n -> n
            Left e  -> error e
        rnn4 = case createNetworkFromVector (networkDimensions rnn1) v4 of
            Right n -> n
            Left e  -> error e
    return $ [rnn3,rnn4]

crossNetworkHalf :: RandomGen g =>  RNNetwork -> RNNetwork -> Rand g [RNNetwork]
crossNetworkHalf rnn1 rnn2 = {-# SCC "crossNetworkHalf" #-} do
    let v1 = networkToVector rnn1
        v2 = networkToVector rnn2
    (v3,v4) <- mixVector v1 v2 0.5
    let
        rnn3 = case createNetworkFromVector (networkDimensions rnn1) v3 of
            Right n -> n
            Left e  -> error e
        rnn4 = case createNetworkFromVector (networkDimensions rnn1) v4 of
            Right n -> n
            Left e  -> error e
    return $ [rnn3,rnn4]

avgNetwork :: RandomGen g =>  RNNetwork -> RNNetwork -> Rand g [RNNetwork]
avgNetwork rnn1 rnn2 = {-# SCC "avgNetwork" #-} do
    let v1 = networkToVector rnn1
        v2 = networkToVector rnn2
        v3 = fromList $ zipWith (\x y -> (x+y)/2) (toList v1) (toList v2)
        rnn3 = case createNetworkFromVector (networkDimensions rnn1) v3 of
            Right n -> n
            Left e  -> error e
      --  rnn4 = case createNetworkFromVector (networkDimensions rnn1) v4 of
      --      Right n -> n
      --      Left e  -> error e
    return $ [rnn3]

crossNetwork :: RandomGen g =>  RNNetwork -> RNNetwork -> Rand g [RNNetwork]
crossNetwork rnn1 rnn2 = {-# SCC "crossNetwork" #-} do
    let (in1,in2) = crossMatrixEq (rnnMIn rnn1) (rnnMIn rnn2)
        (m1,m2) = crossMatrixEq (rnnM rnn1) (rnnM rnn2)
        (out1,out2) = crossMatrixEq (rnnMOut rnn1) (rnnMOut rnn2)
        (bck1,bck2) = case (rnnMBack rnn1,rnnMBack rnn2) of
            (Just m1,Just m2)-> let (m12,m22) = crossMatrixEq m1 m2 in (Just m12,Just m22)
            _ -> (Nothing,Nothing)
        (st1,st2) = crossVectorEq (rnnState rnn1) (rnnState rnn2)
        (os1,os2) = crossVectorEq (rnnOutput rnn1) (rnnOutput rnn2)
    return [RNNetwork (rnnDimensions rnn1) in1 m2 out1 bck1 st1 os1
           ,RNNetwork (rnnDimensions rnn1) in1 m2 out2 bck2 st2 os2
           ]



crossMatrixEq :: Matrix Double -> Matrix Double -> (Matrix Double,Matrix Double)
crossMatrixEq m1 m2 =
    let row      = upperHalf $ rows m1
        col      = upperHalf $ cols m1
        [[a1,a2],[a3,a4]] = toBlocksEvery row col m1
        [[b1,b2],[b3,b4]] = toBlocksEvery row col m2
        --(n1,n2)  = splitAt 4 $ merge blks1 blks2
    in (fromBlocks [[a1,b2],[a3,b4]],fromBlocks [[b1,a2],[b3,a4]])
    where
        upperHalf a = (if odd a then (a+1) else a) `div` 2
        merge :: [a] -> [a] -> [a]
        merge xs     []     = xs
        merge []     ys     = ys
        merge (x:xs) (y:ys) = x : y : merge xs ys

crossVectorEq :: Vector Double -> Vector Double -> (Vector Double,Vector Double)
crossVectorEq v1 v2 = crossVector v1 v2 (size v1 `div` 2)

crossVector :: Vector Double -> Vector Double -> Int -> (Vector Double,Vector Double)
crossVector v1 v2 idx = (vjoin [subVector 0 idx v1,subVector idx (size v2 - idx) v2],
                    vjoin [subVector 0 idx v2,subVector idx (size v1-idx) v1])

mixVector :: (Monad m,RandomGen g) =>  Vector Double -> Vector Double -> Double -> RandT g m (Vector Double,Vector Double)
mixVector v1 v2 prob = do
    rs <- sequence (replicate (size v1) (getRandomR (0, 1)))
    let (l1,l2) = unzip $ map swapR $ zip3 (toList v1) (toList v2) rs
    return (fromList l1,fromList l2)
    where
        swapR (a,b,idx) = if idx <= prob then (a,b) else (b,a)

mutateNetwork :: RandomGen g =>  RNNetwork -> Rand g RNNetwork
mutateNetwork rnn = do
    let v1 = networkToVector rnn
    idx <- getRandomR (0, size v1 - 1)
    dbl <- getRandom
    let v2 = accum v1 (\_ b -> b) [(idx,dbl)]
    -- vjoin [subVector 0 idx v1,fromList [dbl],subVector (idx+1) (size v1-idx-1) v1]
    return $ case createNetworkFromVector (networkDimensions rnn) v2 of
            Right n -> n
            Left e  -> error e

-- | Maximum for the fitness
maxFit :: Double
maxFit = 0


-- | Stop function
stopf ::  RNNData -> Int -> IO Bool
stopf nd gen= do
    print $ "Fitness (" ++ show gen ++"): " ++ show (fitness nd)
    return $ gen > 10 -- || fitness nd >= maxFit

-- | Build a random network data
buildNetworkData :: (Monad m,RandomGen g) =>  RNNDimensions -> [Vector Double] -> [Vector Double] -> RandT g m RNNData
buildNetworkData dim is os = do
  n <- randNetwork dim
  return $ RNNData n is os (fromIntegral $ length is)

