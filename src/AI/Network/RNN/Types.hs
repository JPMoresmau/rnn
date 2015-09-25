{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses, RankNTypes #-}
-----------------------------------------------------------------------------
--
-- Module      :  AI.Network.RNN.Types
-- Copyright   :  (c) JP Moresmau
-- License     :  BSD3
--
-- Maintainer  :  JP Moresmau <jp@moresmau.fr>
-- Stability   :  experimental
-- Portability :
--
-- | Types and some basic functions on them
--
-----------------------------------------------------------------------------

module AI.Network.RNN.Types where

import Control.DeepSeq
import Control.Monad.Random
import Data.List
import Data.Tuple
import Numeric.LinearAlgebra.HMatrix as M

import AI.Network.RNN.Util

import Debug.Trace

-- | Class for RNN evaluation
class (NFData a) => RNNEval a where
    type Size a
    -- | Eval one step
    evalStep   :: a -> Vector Double -> (a,Vector Double)
    -- | Transform to vector
    toVector   :: a -> Vector Double
    -- | Build from vector representation
    fromVector :: Size a -> Vector Double -> a
    -- | Transform to vector
    toDList   :: a -> [Double]
    toDList = toList . toVector
    -- | Build from vector representation
    fromDList :: Size a -> [Double] -> a
    fromDList sz = fromVector sz . M.fromList
    -- | Get the size
    rnnsize    :: a -> Size a
    -- | full size
    fullSize   :: a -> Int

instance (RNNEval a) => RNNEval [a] where
    type Size [a]   = [Size a]
    evalStep xs v = swap $ mapAccumL e v xs
        where e v1 x1 =
                let (x2,v2) = evalStep x1 v1
                in (v2,x2)
    toVector = vjoin . map toVector
    fromVector sz vs = snd $ mapAccumL e vs sz
        where e v1 s1 =
                let x1 = fromVector s1 v1
                    fs1 = fullSize x1
                in (subVector fs1 (size v1 - fs1) v1,x1)
    rnnsize = map rnnsize
    fullSize = sum . map fullSize

-- | Full size of a network: the total number of weights required
type FullSize sz = Size sz -> Int

-- | Eval several steps
evalSteps :: (Traversable t,RNNEval a) => a -> t (Vector Double) -> (a,t (Vector Double))
evalSteps = mapAccumL evalStep

-- | Cost function: lower is better
cost :: (RNNEval a) => a -> TrainData b -> Double
cost rnn td =
    let (_,res) = evalSteps rnn $ tdInputs td
    in (sum $ zipWith err (tdOutputs td) res)
    where
      err ::  Vector Double -> Vector Double -> Double
      --err tgt res =let
      --  norm = cmap (\x->if x>=0.5 then 1 else 0) res
      --  in euclidian  tgt norm
      err = euclidian
       --sum ( zipWith (\c d -> (c- d)**2 ) (toList a) (toList b)) / fromIntegral (size a)
      errsf a b = sum ( zipWith (\c d -> if c==1 then (c- d) else 0) (toList a) (softmax $ toList b))
      sf :: Vector Double -> Vector Double -> Double
      --sf a b = sum ( zipWith (\c d -> if c==1 then log d else 0 ) (toList a) (softmax $ toList b))
      sf a b = log $ (softmaxV b) ! (maxIndex a)
       -- sum ( zipWith (\c d -> if c==1 then log d else 0 ) (toList a) (softmax $ toList b))


-- | Build a random network of a given size, with weights uniformly between 0 and 1
randomNetwork
    :: (RNNEval a,Monad m,RandomGen g)
    => Size a
    -> FullSize a
    -> RandT g m a
randomNetwork sz fsz = do
    s <- getRandom
    return $ fromVector sz $ randomVector s Uniform $ fsz sz

startNetwork :: (RNNEval a)
    => Size a
    -> FullSize a
    -> a
startNetwork sz fsz = fromVector sz $ konst 0.1 (fsz sz)

-- | Training data
data TrainData a = TrainData
    { tdInputs  :: ![Vector Double]
    , tdOutputs :: ![Vector Double]
    , tdRecSize :: !Int
    , tdData    :: a}

class Contains a where
    type ContainsSize a
    buildContains :: ContainsSize a -> a
    toCList :: a -> [Int]

--data Training a c = Training (ContainsSize a) c
--
--randomContains :: (Contains a) => ContainsSize a -> a
--randomContains = buildContains
--
--fromTraining :: (Contains a)=>Training a c -> a
--fromTraining (Training sz d)= buildContains sz
--
--fromTrainingToList :: (Contains a)=>Training a c -> [Int]
--fromTrainingToList (Training sz d)=toCList $ buildContains sz

