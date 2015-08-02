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
import Numeric.LinearAlgebra.HMatrix

-- | Class for RNN evaluation
class (NFData a) => RNNEval a sz | a -> sz where
    -- | Eval one step
    evalStep   :: a -> Vector Double -> (a,Vector Double)
    -- | Transform to vector
    toVector   :: a -> Vector Double
    -- | Build from vector representation
    fromVector :: sz -> Vector Double -> a
    -- | Get the size
    rnnsize    :: a -> sz

-- | Full size of a network: the total number of weights required
type FullSize sz = sz -> Int

-- | Eval several steps
evalSteps :: (Traversable t,RNNEval a sz) => a -> t (Vector Double) -> (a,t (Vector Double))
evalSteps = mapAccumL evalStep

-- | Cost function: lower is better
cost :: (RNNEval a sz) => a -> TrainData b sz -> Double
cost rnn td =
    let (_,res) = evalSteps rnn $ tdInputs td
    in sum $ zipWith err (tdOutputs td) res
    where
      err ::  Vector Double -> Vector Double -> Double
      err a b    = sum ( zipWith (\c d -> (c- d)**2 ) (toList a) (toList b)) / fromIntegral (size a)

-- | Build a random network of a given size, with weights uniformly between 0 and 1
randomNetwork
    :: (RNNEval a sz,Monad m,RandomGen g)
    => sz
    -> FullSize sz
    -> RandT g m a
randomNetwork sz fsz = do
    s <- getRandom
    return $ fromVector sz $ randomVector s Uniform $ fsz sz

-- | Training data
data TrainData a sz = TrainData
    { tdInputs  :: ![Vector Double]
    , tdOutputs :: ![Vector Double]
    , tdRecSize :: !sz
    , tdData    :: a}
