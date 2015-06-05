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
-- |
--
-----------------------------------------------------------------------------

module AI.Network.RNN.Util where



-- | It's a simple, differentiable sigmoid function.
sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}
