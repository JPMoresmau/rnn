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
 ( RNNEval (..)
 , RNNDimensions(..)
 , RNNetwork
 , evalSteps
 , totalDataLength
 , randNetwork
 , createNetwork
 , createNetworkFromArray
 , networkToArray
 , createNetworkFromVector
 , networkDimensions

 , LSTMNetwork
 , lstmFullSize
 , lstmList
 , cost
 , learnGradientDescent

  , textToTrainData
  , dataToText
  , generate
  , textToTrainDataB
  , dataToTextB
  , generateB

  , RNNData(..)
  , buildNetworkData
  , stopf
  , mutateNetwork

  , TrainData(..)
 ) where

import AI.Network.RNN.Data
import AI.Network.RNN.Genetic
import AI.Network.RNN.LSTM
import AI.Network.RNN.RNN
import AI.Network.RNN.Types

