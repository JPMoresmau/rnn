{-# LANGUAGE ScopedTypeVariables #-}
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
import AI.Network.RNN.Util
import Numeric.LinearAlgebra.HMatrix

import Debug.Trace

-- | Keep all relevant data together
data RNNData a b = (RNNEval a) => RNNData a !(TrainData b) !Double
   -- deriving (Show,Read,Eq)

-- | evaluate network
instance NFData (RNNData a b) where
    rnf (RNNData rnn _ _) = rnf rnn

-- | Error threshold (not currently used)
errorThreshold :: Double
errorThreshold = 0.1

-- | Chromosome instance
instance Chromosome (RNNData a b) where
    -- | cross network by taking half the values of one and half of the other
    crossover (RNNData rnn1 td l) (RNNData rnn2 _ _) = do
        rnns <- crossNetworkFull rnn1 rnn2
        --idx <- getRandomR (0, size rnn1 - 1)
        --let (v3,v4) = crossVector rnn1 rnn2 idx
        return $ map (\r->RNNData r td l) rnns -- [v3,v4]

    mutation (RNNData rnn1 td l) = do
        rnn2 <- mutateNetwork rnn1
        --rnn2 <- stdNormalMutate rnn1
        return $ RNNData rnn2 td l
    -- | fitness is "higher is better", inverse of cost
    fitness (RNNData rnn1 td _) = 1 / cost rnn1 td -- (fromVector sz rnn1)

-- | Cross 2 networks by taking the first half of one, the second half of the other (not currently used)
crossNetworkFull :: (RandomGen g,RNNEval a) =>  a -> a -> Rand g [a]
crossNetworkFull rnn1 rnn2 = do
    let v1 = toVector rnn1
        v2 = toVector rnn2
    idx <- getRandomR (0, size v1 - 1)
    let (v3,v4) = crossVector v1 v2 idx
        rnn3 = fromVector (rnnsize rnn1) v3
        rnn4 = fromVector (rnnsize rnn1) v4
    return $[rnn3,rnn4]

-- | cross 2 networks by taking half the values of one and half of the other
crossNetworkHalf :: (RandomGen g,RNNEval a) =>  a -> a -> Rand g [a]
crossNetworkHalf rnn1 rnn2 = do
    let v1 = toVector rnn1
        v2 = toVector rnn2
    (v3,v4) <- mixVector v1 v2 0.5
    let
        rnn3 = fromVector (rnnsize rnn1) v3
        rnn4 = fromVector (rnnsize rnn1) v4
    return $ force [rnn3,rnn4]

crossMutate :: (RandomGen g,RNNEval a) =>  a -> a -> Rand g [a]
crossMutate rnn1 rnn2 = do
    let v1 = toVector rnn1
        v2 = toVector rnn2
    v3 <- stdNormalMutate v1
    v4 <- stdNormalMutate v2
    let
        rnn3 = fromVector (rnnsize rnn1) v3
        rnn4 = fromVector (rnnsize rnn1) v4
    return [rnn3,rnn4]


-- | Cross networks by taking the average of their values (not currently used)
avgNetwork :: (RandomGen g,RNNEval a) =>  a -> a -> Rand g [a]
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
crossVector v1 v2 idx =
--    let m1 = asRow v1
--        m2 = asRow v2
--        [[a1,a2]] = toBlocks [1] [idx,size v1-idx]  m1
--        [[b1,b2]] = toBlocks [1] [idx,size v1-idx] m2
--    in (head $ toRows $ fromBlocks [[a1,b2]],head $ toRows $ fromBlocks [[b1,a2]])
--    let sz2 = size v2 - idx - 1
--        [a1,a2] = takesV [idx,sz2] v1
--        [b1,b2] = takesV [idx,sz2] v2
--    in (vjoin [a1,b2],vjoin [a2,b1])
 let sz = size v1-idx
 in force $ (vjoin [subVector 0 idx v1,subVector idx sz v2],
                    vjoin [subVector 0 idx v2,subVector idx sz v1])

-- | Mix vector values given a probability to take a value from the first one and not the second
mixVector :: (Monad m,RandomGen g) =>  Vector Double -> Vector Double -> Double -> RandT g m (Vector Double,Vector Double)
mixVector v1 v2 prob = do
    -- rs <- replicateM (size v1) (getRandomR (0, 1))
    let (l1,l2) = swapL (toList v1) (toList v2) True
        -- unzip $ map swapR $ zip3 (toList v1) (toList v2) [0..]
    return (fromList l1,fromList l2)
    where
        swapR (a,b,idx) = if (idx `mod` 2) == 0 then (a,b) else (b,a)
        --if idx <= prob then (a,b) else (b,a):
        swapL :: [a] -> [a] -> Bool -> ([a],[a])
        swapL [] _ _ = ([],[])
        swapL (x:xs) (y:ys) True = let
            (rs1,rs2)=swapL xs ys False in (x:rs1,y:rs2)
        swapL (x:xs) (y:ys) False = let
            (rs1,rs2)=swapL xs ys True in (y:rs1,x:rs2)


-- | Mix vector values given a probability to take a value from the first one and not the second
mixList :: (Monad m,RandomGen g) =>  [Double] -> [Double] -> Double -> RandT g m ([Double],[Double])
mixList v1 v2 prob = do
    --rs <- getRandomRs (0,1)
    return $ unzip $ map swapR $ zip3 v1 v2 [0..]
    where
        swapR (a,b,idx) = if (idx `mod` 2) == 0 then (a,b) else (b,a)



stdNormalMutate :: (Monad m,RandomGen g) =>  Vector Double -> RandT g m (Vector Double)
stdNormalMutate v1 = do
    --rs <- replicateM (size v1) stdNormal
    -- let mx = maxElement v1
    -- let mn = minElement v1
    --let sz= trace ("v1:"++ show (mn,mx)) $ size v1
    seed <- getRandom
    let rs = randomVector seed Gaussian $ size v1
    let v2 = (rs) + v1
--    let nv2 = cmap (\x->
--                            if x < 0
--                                then 0
--                                else if x > 1
--                                    then 1
--                                    else x)
--         v2
    -- let mx2 = maxElement v2
    -- let mn2 = minElement v2
    return v2
    -- return $ trace ("v2:"++show (mn2,mx2)) v2

-- | network mutation: select randomly from several mutation algorithms
mutateNetwork :: (RandomGen g,RNNEval a) =>  a -> Rand g a
mutateNetwork rnn = do
    f <- MR.fromList [(stdNormalMutation,1),(pointMutation,1),(swapMutation,1),(insertMutation,1),(flipMutation,0.5)]
    f rnn

-- | Mutate values around a standard deviation
stdNormalMutation :: (RandomGen g,RNNEval a) =>  a -> Rand g a
stdNormalMutation rnn = do
    let v1 = toVector rnn
    v2 <- stdNormalMutate v1
    return $ force $ fromVector (rnnsize rnn) v2

-- | Mutate a single value of the vector, taking a random value
pointMutation :: (RandomGen g,RNNEval a) =>  a -> Rand g a
pointMutation rnn = do
    let v1 = toVector rnn
    idx <- getRandomR (0, size v1 - 1)
    dbl <- getRandomR (0,1)
    let v2 = accum v1 const [(idx,dbl)]
    return $ fromVector (rnnsize rnn) v2

-- | Swap two values in the vector
swapMutation :: (RandomGen g,RNNEval a) =>  a -> Rand g a
swapMutation rnn = do
    let v1 = toVector rnn
    idx1 <- getRandomR (0, size v1 - 1)
    idx2 <- getRandomR (0, size v1 - 1)
    let v2 = accum v1 const [(idx1,atIndex v1 idx2),(idx2,atIndex v1 idx1)]
    return $ fromVector (rnnsize rnn) v2

-- | Insert a random value at a random point in the vector, discarding the last value
insertMutation :: (RandomGen g,RNNEval a) =>  a -> Rand g a
insertMutation rnn = do
    let v1 = toVector rnn
    idx <- getRandomR (0, size v1 - 1)
    dbl <- getRandomR (0,1)
    let v2 = vjoin [subVector 0 idx v1,fromList [dbl],subVector idx (size v1 - idx -1) v1]
    return $ fromVector (rnnsize rnn) v2

flipMutation :: (RandomGen g,RNNEval a) =>  a -> Rand g a
flipMutation rnn = do
    let v1 = toVector rnn
    let v2 = cmap (\a->(-a)) v1
    return $ fromVector (rnnsize rnn) v2

-- | Stop function
-- we take a list of past fitnesses and stop when the fitness 5 generations ago was not worse that the current one
stopf :: IORef [Double] -> Int -> (RNNData a b,Double) -> Int -> IO Bool
stopf fitnessList maxGen (nd,fit) gen = do
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
    return ( gen >= maxGen)


-- | Build a random network data
buildNetworkData :: (RNNEval re,Monad m,RandomGen g) => TrainData b -> Size re -> FullSize re -> RandT g m (RNNData re b)
buildNetworkData td@(TrainData is _ _ _) dim fullSz = do
  --n <- randomNetwork dim fullSz
  let n = startNetwork dim fullSz
  return $ RNNData n td (fromIntegral $ length is)

