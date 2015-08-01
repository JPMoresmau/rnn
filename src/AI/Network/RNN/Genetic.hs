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
-- |
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

import Debug.Trace

data RNNData a sz b = (RNNEval a sz) => RNNData !a !(TrainData b sz) !Double
   -- deriving (Show,Read,Eq)

instance NFData (RNNData a sz b) where
    rnf (RNNData rnn _ _) = rnf rnn

errorThreshold :: Double
errorThreshold = 0.1

instance Chromosome (RNNData a sz b) where
    crossover (RNNData rnn1 td l) (RNNData rnn2 _  _) = do
        rnns <- crossNetworkHalf rnn1 rnn2
        return $ map (\r->RNNData r td l) rnns

    mutation (RNNData rnn1 td l) = do
        rnn2 <- mutateNetwork rnn1
        return $ RNNData rnn2 td l

    fitness (RNNData rnn1 td l) = 1 / (cost rnn1 td)
--        let (_,res) = evalSteps rnn1 is
--            err a b    = (sum $ zipWith (\c d -> (c- d)**2 ) (toList a) (toList b)) / fromIntegral (size a)
--         --   z = zip os res
--         --   ok = length $ takeWhile (\(a,b)->err a b < errorThreshold) z
--        in 1 / ((sum $ zipWith err os res) / l)
                --  if ok > 0
                --    then fromIntegral ok - (uncurry err $ head $ drop ok z)
                --    else - (err (head os) (head res))

crossNetworkFull :: (RandomGen g,RNNEval a sz) =>  a -> a -> Rand g [a]
crossNetworkFull rnn1 rnn2 = do
    let v1 = toVector rnn1
        v2 = toVector rnn2
    idx <- getRandomR (0, size v1 - 1)
    let (v3,v4) = crossVector v1 v2 idx
        rnn3 = fromVector (rnnsize rnn1) v3
        rnn4 = fromVector (rnnsize rnn1) v4
    return $ [rnn3,rnn4]

crossNetworkHalf :: (RandomGen g,RNNEval a sz) =>  a -> a -> Rand g [a]
crossNetworkHalf rnn1 rnn2 = do
    let v1 = toVector rnn1
        v2 = toVector rnn2
    (v3,v4) <- mixVector v1 v2 0.5
    let
        rnn3 = fromVector (rnnsize rnn1) v3
        rnn4 = fromVector (rnnsize rnn1) v4
    return $ [rnn3,rnn4]

avgNetwork :: (RandomGen g,RNNEval a sz) =>  a -> a -> Rand g [a]
avgNetwork rnn1 rnn2 = do
    let v1 = toVector rnn1
        v2 = toVector rnn2
        v3 = fromList $ zipWith (\x y -> (x+y)/2) (toList v1) (toList v2)
        rnn3 = fromVector (rnnsize rnn1) v3
      --  rnn4 = fromVector (rnnsize rnn1) v4 of
      --      Right n -> n
      --      Left e  -> error e
    return $ [rnn3]

--crossNetwork :: RandomGen g =>  RNNetwork -> RNNetwork -> Rand g [RNNetwork]
--crossNetwork rnn1 rnn2 = {-# SCC "crossNetwork" #-} do
--    let (in1,in2) = crossMatrixEq (rnnMIn rnn1) (rnnMIn rnn2)
--        (m1,m2) = crossMatrixEq (rnnM rnn1) (rnnM rnn2)
--        (out1,out2) = crossMatrixEq (rnnMOut rnn1) (rnnMOut rnn2)
--        (bck1,bck2) = case (rnnMBack rnn1,rnnMBack rnn2) of
--            (Just m1,Just m2)-> let (m12,m22) = crossMatrixEq m1 m2 in (Just m12,Just m22)
--            _ -> (Nothing,Nothing)
--        (st1,st2) = crossVectorEq (rnnState rnn1) (rnnState rnn2)
--        (os1,os2) = crossVectorEq (rnnOutput rnn1) (rnnOutput rnn2)
--    return [RNNetwork (rnnDimensions rnn1) in1 m2 out1 bck1 st1 os1
--           ,RNNetwork (rnnDimensions rnn1) in1 m2 out2 bck2 st2 os2
--           ]



crossMatrixEq :: Matrix Double -> Matrix Double -> (Matrix Double,Matrix Double)
crossMatrixEq m1 m2 =
    let row      = upperHalf $ rows m1
        col      = upperHalf $ cols m1
        [[a1,a2],[a3,a4]] = toBlocksEvery row col m1
        [[b1,b2],[b3,b4]] = toBlocksEvery row col m2
    in (fromBlocks [[a1,b2],[a3,b4]],fromBlocks [[b1,a2],[b3,a4]])
    where
        upperHalf a = (if odd a then (a+1) else a) `div` 2

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

mutateNetwork :: (RandomGen g,RNNEval a sz) =>  a -> Rand g a
mutateNetwork rnn = do
    f <- MR.fromList [(pointMutation,1),(swapMutation,1),(insertMutation,1)]
    f rnn

pointMutation :: (RandomGen g,RNNEval a sz) =>  a -> Rand g a
pointMutation rnn = do
    let v1 = toVector rnn
    idx <- getRandomR (0, size v1 - 1)
    dbl <- getRandom
    let v2 = accum v1 (\a _ -> a) [(idx,dbl)]
    return $ fromVector (rnnsize rnn) v2

swapMutation :: (RandomGen g,RNNEval a sz) =>  a -> Rand g a
swapMutation rnn = do
    let v1 = toVector rnn
    idx1 <- getRandomR (0, size v1 - 1)
    idx2 <- getRandomR (0, size v1 - 1)
    let v2 = accum v1 (\a _ -> a) [(idx1,atIndex v1 idx2),(idx2,atIndex v1 idx1)]
    return $ fromVector (rnnsize rnn) v2

insertMutation :: (RandomGen g,RNNEval a sz) =>  a -> Rand g a
insertMutation rnn = do
    let v1 = toVector rnn
    idx <- getRandomR (0, size v1 - 1)
    dbl <- getRandom
    let v2 = vjoin [subVector 0 idx v1,fromList [dbl],subVector idx (size v1 - idx -1) v1]
    return $ fromVector (rnnsize rnn) v2

-- | Maximum for the fitness
maxFit :: Double
maxFit = 0


-- | Stop function
stopf :: IORef [Double] -> Int -> RNNData a sz b -> Int -> IO Bool
stopf fitnessList maxGen nd gen = do
    let fit = fitness nd
    mfit <- atomicModifyIORef' fitnessList (\l->
        let l2 = take 5 $ (fit:l)
        in if length l2 == 5
            then (l2, Just $ last l2)
            else (l2, Nothing)
        )

    print $ "Fitness (" ++ show gen ++"): " ++ show fit
    let converged = case mfit of
                        Nothing -> False
                        Just f  -> f >= fit
    when converged $ print "Converged!"
    return $ ( gen >= maxGen || converged)
     -- || fitness nd >= maxFit

-- | Build a random network data
buildNetworkData :: (Monad m,RandomGen g,(RNNEval a sz)) =>  TrainData b sz -> FullSize sz -> RandT g m (RNNData a sz b)
buildNetworkData td@(TrainData is _ dim _) fullSz = do
  n <- randomNetwork dim fullSz
  return $ RNNData n td (fromIntegral $ length is)

