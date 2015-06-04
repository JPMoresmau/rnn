{-# LANGUAGE RecordWildCards, PatternGuards, BangPatterns #-}
-----------------------------------------------------------------------------
--
-- Module      :  AI.Network.RNN
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

module AI.Network.RNN
 ( RNNDimensions(..)
 , RNNetwork
 , evalStep
 , evalSteps
 , createRandomNetwork
 , randNetwork
 , createNetwork
 , createNetworkFromArray
 , networkToArray
 , createNetworkFromVector
 , networkToVector
 , networkDimensions

  , textToTrainData
  , dataToText
  , generate

  , RNNData(..)
  , buildNetworkData
  , stopf
 ) where

import AI.Network.RNN.Data
import AI.Network.RNN.Genetic
import AI.Network.RNN.RNN

