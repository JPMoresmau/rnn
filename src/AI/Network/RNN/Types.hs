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
-- |
--
-----------------------------------------------------------------------------

module AI.Network.RNN.Types where

import Control.DeepSeq
import Control.Monad.Random
import Data.List
import Numeric.LinearAlgebra.HMatrix

class (NFData a) => RNNEval a sz | a -> sz where
    evalStep   :: a -> (Vector Double) -> (a,Vector Double)
    toVector   :: a -> Vector Double
    fromVector :: sz -> Vector Double -> a
    rnnsize    :: a -> sz

type FullSize sz = sz -> Int

evalSteps :: (Traversable t,RNNEval a sz) => a -> t (Vector Double) -> (a,t (Vector Double))
evalSteps rnn = mapAccumL evalStep rnn

cost :: (RNNEval a sz) => a -> TrainData b sz -> Double
cost rnn td =
    let (_,res) = evalSteps rnn $ tdInputs td
    in ((sum $ zipWith err (tdOutputs td) res))
    where
      err ::  Vector Double -> Vector Double -> Double
      err a b    = (sum $ zipWith (\c d -> (c- d)**2 ) (toList a) (toList b)) / fromIntegral (size a)

randomNetwork
    :: (RNNEval a sz,Monad m,RandomGen g)
    => sz
    -> FullSize sz
    -> RandT g m a
randomNetwork sz fsz = do
    s <- getRandom
    return $ fromVector sz $ randomVector s Uniform $ fsz sz

data TrainData a sz = TrainData
    { tdInputs  :: !([Vector Double])
    , tdOutputs :: !([Vector Double])
    , tdRecSize :: !sz
    , tdData    :: a}
