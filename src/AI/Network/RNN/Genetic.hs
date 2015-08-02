{-# LANGUAGE ExistentialQuantification #-}
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
-- | Code for genetic evolution of a network
--
-----------------------------------------------------------------------------

module AI.Network.RNN.Genetic where

import Control.Monad
import Control.Monad.Random hiding (fromList)
import qualified Control.Monad.Random as MR
import Control.DeepSeq
import Data.IORef
import AI.GeneticAlgorithm.Simple
import AI.Network.RNN.Types
import Numeric.LinearAlgebra.HMatrix

-- import Debug.Trace

-- | Keep all relevant data together
data RNNData a sz b = (RNNEval a sz) => RNNData !a !(TrainData b sz) !Double
   -- deriving (Show,Read,Eq)

-- | evaluate network
instance NFData (RNNData a sz b) where
    rnf (RNNData rnn _ _) = rnf rnn

-- | Error threshold (not currently used)
errorThreshold :: Double
errorThreshold = 0.1

-- | Chromosome instance
instance Chromosome (RNNData a sz b) where
    -- | cross network by taking half the values of one and half of the other
    crossover (RNNData rnn1 td l) (RNNData rnn2 _  _) = do
        rnns <- crossNetworkHalf rnn1 rnn2
        return $ map (\r->RNNData r td l) rnns

    mutation (RNNData rnn1 td l) = do
        rnn2 <- mutateNetwork rnn1
        return $ RNNData rnn2 td l
    -- | fitness is "higher is better", inverse of cost
    fitness (RNNData rnn1 td _) = 1 / cost rnn1 td

-- | Cross 2 networks by taking the first half of one, the second half of the other (not currently used)
crossNetworkFull :: (RandomGen g,RNNEval a sz) =>  a -> a -> Rand g [a]
crossNetworkFull rnn1 rnn2 = do
    let v1 = toVector rnn1
        v2 = toVector rnn2
    idx <- getRandomR (0, size v1 - 1)
    let (v3,v4) = crossVector v1 v2 idx
        rnn3 = fromVector (rnnsize rnn1) v3
        rnn4 = fromVector (rnnsize rnn1) v4
    return [rnn3,rnn4]

-- | cross 2 networks by taking half the values of one and half of the other
crossNetworkHalf :: (RandomGen g,RNNEval a sz) =>  a -> a -> Rand g [a]
crossNetworkHalf rnn1 rnn2 = do
    let v1 = toVector rnn1
        v2 = toVector rnn2
    (v3,v4) <- mixVector v1 v2 0.5
    let
        rnn3 = fromVector (rnnsize rnn1) v3
        rnn4 = fromVector (rnnsize rnn1) v4
    return [rnn3,rnn4]

-- | Cross networks by taking the average of their values (not currently used)
avgNetwork :: (RandomGen g,RNNEval a sz) =>  a -> a -> Rand g [a]
avgNetwork rnn1 rnn2 = do
    let v1 = toVector rnn1
        v2 = toVector rnn2
        v3 = fromList $ zipWith (\x y -> (x+y)/2) (toList v1) (toList v2)
        rnn3 = fromVector (rnnsize rnn1) v3
      --  rnn4 = fromVector (rnnsize rnn1) v4 of
      --      Right n -> n
      --      Left e  -> error e
    return [rnn3]


crossMatrixEq :: Matrix Double -> Matrix Double -> (Matrix Double,Matrix Double)
crossMatrixEq m1 m2 =
    let row1      = upperHalf $ rows m1
        col1      = upperHalf $ cols m1
        [[a1,a2],[a3,a4]] = toBlocksEvery row1 col1 m1
        [[b1,b2],[b3,b4]] = toBlocksEvery row1 col1 m2
    in (fromBlocks [[a1,b2],[a3,b4]],fromBlocks [[b1,a2],[b3,a4]])
    where
        upperHalf a = (if odd a then a+1 else a) `div` 2

crossVectorEq :: Vector Double -> Vector Double -> (Vector Double,Vector Double)
crossVectorEq v1 v2 = crossVector v1 v2 (size v1 `div` 2)

-- | Cross vector at the given point
crossVector :: Vector Double -> Vector Double -> Int -> (Vector Double,Vector Double)
crossVector v1 v2 idx = (vjoin [subVector 0 idx v1,subVector idx (size v2 - idx) v2],
                    vjoin [subVector 0 idx v2,subVector idx (size v1-idx) v1])

-- | Mix vector values given a probability to take a value from the first one and not the second
mixVector :: (Monad m,RandomGen g) =>  Vector Double -> Vector Double -> Double -> RandT g m (Vector Double,Vector Double)
mixVector v1 v2 prob = do
    rs <- replicateM (size v1) (getRandomR (0, 1))
    let (l1,l2) = unzip $ map swapR $ zip3 (toList v1) (toList v2) rs
    return (fromList l1,fromList l2)
    where
        swapR (a,b,idx) = if idx <= prob then (a,b) else (b,a)

-- | network mutation: select randomly from several mutation algorithms
mutateNetwork :: (RandomGen g,RNNEval a sz) =>  a -> Rand g a
mutateNetwork rnn = do
    f <- MR.fromList [(pointMutation,1),(swapMutation,1),(insertMutation,1)]
    f rnn

-- | Mutate a single value of the vector, taking a random value
pointMutation :: (RandomGen g,RNNEval a sz) =>  a -> Rand g a
pointMutation rnn = do
    let v1 = toVector rnn
    idx <- getRandomR (0, size v1 - 1)
    dbl <- getRandomR (0,1)
    let v2 = accum v1 const [(idx,dbl)]
    return $ fromVector (rnnsize rnn) v2

-- | Swap two values in the vector
swapMutation :: (RandomGen g,RNNEval a sz) =>  a -> Rand g a
swapMutation rnn = do
    let v1 = toVector rnn
    idx1 <- getRandomR (0, size v1 - 1)
    idx2 <- getRandomR (0, size v1 - 1)
    let v2 = accum v1 const [(idx1,atIndex v1 idx2),(idx2,atIndex v1 idx1)]
    return $ fromVector (rnnsize rnn) v2

-- | Insert a random value at a random point in the vector, discarding the last value
insertMutation :: (RandomGen g,RNNEval a sz) =>  a -> Rand g a
insertMutation rnn = do
    let v1 = toVector rnn
    idx <- getRandomR (0, size v1 - 1)
    dbl <- getRandomR (0,1)
    let v2 = vjoin [subVector 0 idx v1,fromList [dbl],subVector idx (size v1 - idx -1) v1]
    return $ fromVector (rnnsize rnn) v2


-- | Stop function
-- we take a list of past fitnesses and stop when the fitness 5 generations ago was not worse that the current one
stopf :: IORef [Double] -> Int -> RNNData a sz b -> Int -> IO Bool
stopf fitnessList maxGen nd gen = do
    let fit = fitness nd
    mfit <- atomicModifyIORef' fitnessList (\l->
        let l2 = take 5 (fit:l)
        in if length l2 == 5
            then (l2, Just $ last l2)
            else (l2, Nothing)
        )

    print $ "Fitness (" ++ show gen ++"): " ++ show fit
    let converged = case mfit of
                        Nothing -> False
                        Just f  -> f >= fit
    when converged $ print "Converged!"
    return ( gen >= maxGen || converged)


-- | Build a random network data
buildNetworkData :: (Monad m,RandomGen g,(RNNEval a sz)) =>  TrainData b sz -> FullSize sz -> RandT g m (RNNData a sz b)
buildNetworkData td@(TrainData is _ dim _) fullSz = do
  n <- randomNetwork dim fullSz
  return $ RNNData n td (fromIntegral $ length is)

