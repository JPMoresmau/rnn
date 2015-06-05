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
evalSteps rnn = {-# SCC "evalSteps" #-} mapAccumL evalStep rnn

randomNetwork
    :: (RNNEval a sz,Monad m,RandomGen g)
    => sz
    -> FullSize sz
    -> RandT g m a
randomNetwork sz fsz = do
    s <- getRandom
    return $ fromVector sz $ randomVector s Gaussian $ fsz sz
